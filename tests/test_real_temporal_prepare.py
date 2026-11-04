"""Real TEST opt-in, geometry provenance, GT selection and complete chronology."""
import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import finish_record, run_record, sha256, write_json
from scripts.real_eval.night_prepare import audit, prepare, _select, _rank_view
from facade_change.segmentation.data import PreparedData
from scripts.real_eval.night_common import load_frame, load_prepared
from scripts.real_eval.real_gt_metrics import derive_labels
from scripts.real_eval.robustness_common import read_json


def fixture(root):
    dataset, run, snapshot, alignments = (root / name for name in ("dataset", "run", "snapshot", "alignments"))
    for directory in (dataset, run, snapshot, alignments):
        directory.mkdir()
    images, coco_images, annotations, crops, pairs = [], [], [], [], []
    categories = [{"id": 1, "name": "WALL"}, {"id": 2, "name": "CRACK"}]
    groups = [("train", "train-wide", 16), ("train", "train-small", 8),
              ("val", "val-facade", 8), ("test", "test-facade", 8)]
    runs = []
    for part, building, damage_width in groups:
        group = []
        for frame in range(3):
            identifier = len(images) + 1
            rgb = np.zeros((16, 24, 3), np.uint8)
            rgb[:] = [identifier * 8, 50 + frame * 5, 100]
            path = dataset / f"image-{identifier}.png"
            Image.fromarray(rgb).save(path)
            image = dict(image_id=identifier, file_name=path.name, image_path=str(path), sha256=sha256(path),
                         width=24, height=16, building_id=building, view_id=building + "-view", year=2000 + frame,
                         split=part, metadata_status="reviewed", temporal_eligible=True)
            images.append(image)
            group.append(image)
            coco_images.append(dict(id=identifier, file_name=path.name, width=24, height=16))
            # Rightmost 8 pixels are not annotated and must remain unknown.
            annotations.append(dict(id=len(annotations)+1, image_id=identifier, category_id=1, iscrowd=0,
                                    segmentation=[[0, 0, 16, 0, 16, 16, 0, 16]], area=256, bbox=[0, 0, 16, 16]))
            if frame == 1:  # Damage appears, then is repaired: union != sum of observations.
                annotations.append(dict(id=len(annotations)+1, image_id=identifier, category_id=2, iscrowd=0,
                                        segmentation=[[0, 0, damage_width, 0, damage_width, 8, 0, 8]],
                                        area=damage_width*8, bbox=[0, 0, damage_width, 8]))
            crops.append(dict(crop_id=f"crop-{identifier}", image_id=identifier, image_sha256=image["sha256"],
                              building_id=building, view_id=image["view_id"], split=part,
                              x=0, y=0, width=24, height=16, pad_right=0, pad_bottom=8))
        for source in group[1:]:
            reference = group[0]
            pair_id = f"{reference['image_id']}-{source['image_id']}"
            pairs.append(dict(pair_id=pair_id, reference_id=reference["image_id"], source_id=source["image_id"],
                              building_id=building, view_id=reference["view_id"], split=part))
            output = alignments / ("pair-" + pair_id)
            output.mkdir()
            record = run_record("pair_alignment", dict(method="sift"))
            record["observations"] = dict(reference=reference, source=source)
            record["quality_gate"] = dict(passed=True)
            write_json(output / "geometry.json", dict(source_to_reference=np.eye(3).tolist()))
            finish_record(output, record, "completed")
            runs.append(str(output))
    write_json(dataset / "manifest.json", dict(images=images, pairs=pairs, categories=categories))
    write_json(dataset / "split.json", dict(building_assignments={row["building_id"]: row["split"] for row in images}))
    write_json(dataset / "coco.json", dict(images=coco_images, annotations=annotations, categories=categories))
    (dataset / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in crops))
    write_json(dataset / "dataset.json", dict(kind="semantic_crop_index", crop_size=24, source_coco="coco.json",
                                            manifest="manifest.json", crops="crops.jsonl", split="split.json",
                                            source_coco_sha256=sha256(dataset / "coco.json"), categories=categories))
    config = dict(dataset=str(dataset / "dataset.json"), alignment_runs=runs)
    write_json(run / "resolved_config.json", config)
    write_json(run / "experiment_contract.json", dict(config=config))
    write_json(snapshot / "2026-10-17_source_units.json", [])
    write_json(snapshot / "2026-10-17_quick_summary.json", dict(source_run=str(run)))
    return dict(dataset=dataset, run=run, snapshot=snapshot, alignments=alignments, runs=runs, images=images)


@unittest.skipUnless(importlib.util.find_spec("cv2") and importlib.util.find_spec("pycocotools"), "OpenCV/COCO required")
class RealTemporalPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = fixture(self.root)

    def audit(self, name="audit", **kwargs):
        return audit(self.input["run"], self.input["snapshot"], self.root / name, **kwargs)

    def test_audit_is_metadata_only_includes_test_and_preserves_parent_ids(self):
        with patch.object(Image, "open", side_effect=AssertionError("decoded RGB or GT")):
            result = self.audit()
        self.assertEqual(result["coverage"]["eligible_facades"], 4)
        self.assertEqual(result["coverage"]["by_split"]["test"]["history_eligible_facades"], 1)
        self.assertFalse(result["rgb_loaded"])
        self.assertFalse(result["annotations_rasterized"])
        original = importlib.import_module("facade_change.research.2026-10-14_temporal_inputs")
        parents, _, _, _ = original._real(result["source_config"], {})
        self.assertEqual({row["sequence_id"] for row in parents},
                         {row["sequence_id"] for row in result["sequences"] if row["split"] != "test"})

    def test_missing_alignment_is_explicit_not_identity_fallback(self):
        selected_runs = self.input["runs"][:-2]
        result = self.audit(alignment_runs=selected_runs)
        self.assertEqual(result["coverage"]["by_split"]["test"]["eligible_facades"], 0)
        view = next(row for row in result["coverage"]["views"] if row["split"] == "test")
        self.assertEqual(view["status"], "no_accepted_source")
        self.assertEqual([pair["status"] for pair in view["pairs"]], ["missing_alignment"]*2)
        self.assertEqual(result["coverage"]["pair_status_counts"]["missing_alignment"], 2)

    def test_discovery_finds_terminal_runs_and_lists_running(self):
        ongoing = self.input["alignments"] / "ongoing"
        ongoing.mkdir()
        write_json(ongoing / "run.json", dict(kind="pair_alignment", status="running"))
        result = self.audit(alignment_runs=[], runs_root=self.input["alignments"])
        self.assertEqual(result["coverage"]["eligible_facades"], 4)
        self.assertEqual(result["coverage"]["excluded_alignment_runs"][0]["reason"], "alignment_still_running")

    def test_stale_discovered_metadata_and_invalid_geometry_do_not_block_other_facades(self):
        stale = Path(self.input["runs"][0]) / "run.json"
        record = read_json(stale)
        record["observations"]["source"]["year"] = 1900
        write_json(stale, record)
        broken = Path(self.input["runs"][1]) / "geometry.json"
        write_json(broken, dict(source_to_reference=np.zeros((3, 3)).tolist()))
        result = self.audit(alignment_runs=[], runs_root=self.input["alignments"])
        view = next(row for row in result["coverage"]["views"] if row["building_id"] == "train-wide")
        self.assertEqual(view["status"], "no_accepted_source")
        self.assertEqual(view["pairs"][0]["attempts"][0]["reason"], "incompatible_current_metadata")
        self.assertEqual(view["pairs"][1]["attempts"][0]["reason"], "invalid_discovered_geometry")
        self.assertEqual(result["coverage"]["eligible_facades"], 3)

    def test_prepare_balances_splits_preserves_chronology_and_unknown(self):
        self.audit()
        result = prepare(self.root / "audit", self.root / "prepared", max_facades=3, crops_per_facade=1,
                         tile_size=24, stride=24, min_known_pixels=10)
        self.assertEqual([row["split"] for row in result["sequences"]], ["test", "val", "train"])
        self.assertEqual(result["sequences"][-1]["building_id"], "train-wide")
        self.assertEqual(result["coverage"]["previous_pairs"], 6)
        self.assertEqual(result["coverage"]["earliest_pairs"], 6)
        self.assertTrue(result["prepared_data"]["test_annotations_rasterized"])
        self.assertEqual(result["dataset"], str(self.input["dataset"] / "dataset.json"))
        for sequence in result["sequences"]:
            self.assertEqual([frame["year"] for frame in sequence["frames"]], [2000, 2001, 2002])
            self.assertEqual([frame["frame"] for frame in sequence["frames"]], [0, 1, 2])
            stats = sequence["selection_statistics"]
            self.assertEqual(stats["temporal_changed_pixel_observations"], stats["changed_union_pixels"] * 2)
            self.assertEqual(stats["known_union_pixels"], 256)
            with np.load(self.root / "prepared" / sequence["frames"][1]["artifact_path"], allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"rgb", "support", "packed"})
                self.assertEqual(arrays["rgb"].dtype, np.uint8)
                self.assertEqual(arrays["support"].dtype, bool)
                self.assertEqual(arrays["packed"].dtype, np.uint16)
                self.assertFalse(arrays["packed"][:, 16:].any())
        gallery = (self.root / "prepared" / "gallery.html").read_text()
        self.assertIn("EARLIEST", gallery)
        self.assertIn("data:image/jpeg;base64,", gallery)
        self.assertEqual(read_json(self.root / "prepared" / "run.json")["status"], "completed")

    def test_prepare_detects_changed_metadata(self):
        self.audit()
        coco = self.input["dataset"] / "coco.json"
        document = read_json(coco)
        document["annotations"] = []
        write_json(coco, document)
        with self.assertRaisesRegex(ValueError, "COCO hash"):
            prepare(self.root / "audit", self.root / "prepared", tile_size=8, stride=8, min_known_pixels=4)

    def test_all_geometric_keeps_positive_stable_unknown_and_all_splits(self):
        self.audit()
        rasterizations = Counter()
        original = PreparedData._rasterize

        def count_rasterizations(data, image_id, height, width):
            rasterizations[image_id] += 1
            return original(data, image_id, height, width)

        with patch.object(PreparedData, "_rasterize", count_rasterizations), patch(
                "scripts.real_eval.night_prepare._select", side_effect=AssertionError("GT selection invoked")):
            result = prepare(self.root / "audit", self.root / "full", tile_size=8, stride=8,
                             selection_mode="all-geometric", gallery_limit=3)
        self.assertEqual(result["coverage"]["facades"], 4)
        self.assertEqual(result["coverage"]["sequences"], 24)
        self.assertEqual(result["coverage"]["frames"], 72)
        self.assertEqual(result["coverage"]["previous_pairs"], 48)
        self.assertEqual({s["split"] for s in result["sequences"]}, {"train", "val", "test"})
        self.assertFalse(result["selection_policy"]["gt_used_for_selection"])
        self.assertTrue(result["selection_policy"]["complete_audited_buildings"])
        self.assertIsNone(result["selection_policy"]["max_facades"])
        self.assertIsNone(result["selection_policy"]["min_known_pixels"])
        self.assertEqual(result["gallery_policy"]["shown_sequences"], 3)
        self.assertEqual(len(rasterizations), 12)
        self.assertEqual(set(rasterizations.values()), {2})  # once for inspection, once for all windows
        root, manifest = load_prepared(self.root / "full")
        types = Counter()
        for sequence in manifest["sequences"]:
            self.assertEqual([f["year"] for f in sequence["frames"]], [2000, 2001, 2002])
            frames = [load_frame(root, frame) for frame in sequence["frames"]]
            x, y, width, height = sequence["crop_xywh"]
            for frame_info, arrays in zip(sequence["frames"], frames):
                original_path = self.input["images"][frame_info["image_id"] - 1]["image_path"]
                with Image.open(original_path) as original_image:
                    expected_rgb = np.asarray(original_image)[y:y+height, x:x+width]
                np.testing.assert_array_equal(arrays["rgb"], expected_rgb)
                self.assertTrue(arrays["support"].all())
            labels = derive_labels(frames[0]["packed"], frames[1]["packed"],
                                   frames[0]["support"] & frames[1]["support"], 2)
            types["changed" if np.any(labels == 1) else "unknown" if np.all(labels == 255) else "stable"] += 1
        self.assertEqual(types, Counter(changed=5, stable=11, unknown=8))
        gallery = (self.root / "full" / "gallery.html").read_text()
        self.assertIn("Показано 3 из 24", gallery)
        self.assertIn("нет известной области", gallery)

    def test_all_geometric_selection_does_not_depend_on_annotations(self):
        self.audit()
        first = prepare(self.root / "audit", self.root / "full", tile_size=8, stride=8,
                        selection_mode="all-geometric", gallery_limit=0)
        coco = self.input["dataset"] / "coco.json"
        annotations = read_json(coco)
        annotations["annotations"] = []
        write_json(coco, annotations)
        dataset_path = self.input["dataset"] / "dataset.json"
        metadata = read_json(dataset_path)
        metadata["source_coco_sha256"] = sha256(coco)
        write_json(dataset_path, metadata)
        self.audit(name="unknown-audit")
        second = prepare(self.root / "unknown-audit", self.root / "unknown-full", tile_size=8, stride=8,
                         selection_mode="all-geometric", gallery_limit=0)
        self.assertEqual([s["sequence_id"] for s in first["sequences"]],
                         [s["sequence_id"] for s in second["sequences"]])
        self.assertTrue(all(s["selection_statistics"]["known_union_pixels"] == 0 for s in second["sequences"]))

    def test_all_geometric_rejects_selection_caps_and_retains_geometry_gate(self):
        audit_result = self.audit()
        for options in ({"max_facades": 1}, {"crops_per_facade": 1}, {"min_known_pixels": 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "all-geometric does not accept"):
                prepare(self.root / "audit", self.root / "full", selection_mode="all-geometric", **options)
        parent = audit_result["sequences"][0]
        support = np.ones((16, 24), bool)
        support[:, :8] = False
        empty = np.zeros((16, 24), np.uint16)
        with patch("scripts.real_eval.night_prepare._warp", return_value=dict(support=support, packed=empty)):
            candidates, _ = _rank_view(type("Data", (), {"categories": [1, 2]})(), parent, 8, 8, .8, None,
                                       "all-geometric")
        self.assertEqual(sum(c["status"] == "eligible" for c in candidates), 4)
        self.assertEqual({c["exclusion_reason"] for c in candidates if c["status"] == "excluded"},
                         {"low_common_support"})

    def test_prepare_detects_changed_rgb_and_saves_failure(self):
        self.audit()
        path = Path(self.input["images"][-1]["image_path"])
        Image.fromarray(np.full((16, 24, 3), 23, np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
            prepare(self.root / "audit", self.root / "prepared", tile_size=8, stride=8, min_known_pixels=4)
        self.assertEqual(read_json(self.root / "prepared" / "run.json")["status"], "failed")

    def test_no_repeated_output_and_invalid_geometry_rejected(self):
        self.audit()
        with self.assertRaises(FileExistsError):
            self.audit()
        geometry = Path(self.input["runs"][0]) / "geometry.json"
        write_json(geometry, dict(source_to_reference=np.zeros((3, 3)).tolist()))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.audit(name="audit2")

    def test_two_crops_do_not_duplicate_same_view(self):
        candidates = []
        for x, count in ((0, 100), (2, 99), (8, 80)):
            candidates.append(dict(sequence_id=f"p-x{x}", parent_sequence_id="p", building_id="b", view_id="v",
                                   split="test", status="eligible", crop_xywh=[x, 0, 8, 8],
                                   changed_union_pixels=count, known_union_pixels=128, selected=False))
        views = [dict(building_id="b", view_id="v", split="test", eligible_crops=3,
                      changed_union_pixels=200, known_union_pixels=256)]
        selected, _ = _select(candidates, views, 1, 2, None)
        self.assertEqual([row["crop_xywh"][0] for row in selected], [0, 8])


if __name__ == "__main__":
    unittest.main()
