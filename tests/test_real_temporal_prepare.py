"""Real TEST opt-in, geometry provenance, GT selection and complete chronology."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.io import finish_record, run_record, sha256, write_json
from scripts.real_eval.night_prepare import audit, prepare, _select
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
