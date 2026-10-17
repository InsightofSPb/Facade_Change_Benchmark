"""Review preparation preserves frozen data, unknown labels, and GT-only selection."""
import builtins
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real_eval import robustness_prepare as prep
from scripts.real_eval.robustness_common import atomic_json, load_pair, load_prepared, load_selection, sha256


def fixture(base):
    run, snapshot = base / "source", base / "snapshot"
    (run / "maps").mkdir(parents=True)
    snapshot.mkdir()
    sequences, frames, records, reads = {}, {}, [], []
    for number, building in enumerate(("A", "B<script>alert(1)</script>")):
        identifier = f"real-{number:024x}-x0-y0"
        sequences[identifier] = dict(sequence_id=identifier, building_id=building, view_id=f"view{number}",
                                     domain="real", split="val", crop_xywh=[0, 0, 12, 8],
                                     frames=[dict(image_id=number*10+i, year=2000+i) for i in range(4)])
        for i in range(4):
            rgb = np.full((8, 12, 3), 80+i*10+number*20, np.uint8)
            bits = np.ones((8, 12), np.uint16)
            if i >= 1:
                bits[1:4, 2:5] = 3
            if i == 3:
                bits[:] = 0
            bits[6:, 9:] = 0
            support = np.ones((8, 12), bool)
            support[:, -1] = False
            frames[(identifier, i)] = dict(rgb=rgb, packed=bits, support=support)
        for i in (1, 2, 3):
            old, new = frames[(identifier, i-1)], frames[(identifier, i)]
            unit_id = f"rgb-{number}-{i}"
            path = run / "maps" / (unit_id + ".npz")
            np.savez_compressed(path, reference=old["rgb"], source=new["rgb"], support=old["support"] & new["support"],
                                scores=np.array([{"poison": "do not read score arrays"}], object),
                                labels=np.array([{"poison": "never use placeholder GT"}], object))
            records.append(dict(unit_id=unit_id, sequence_id=identifier, building_id=building, frame=i, domain="real", split="val",
                                method="rgb_diff", variant="frozen", mode="previous", target="step_event", status="completed",
                                artifact_path=str(path.relative_to(run)), artifact_sha256=sha256(path)))
    records.append(dict(records[0], unit_id="failed", sequence_id="real-ffffffffffffffffffffffff-x0-y0", status="failed", artifact_path="missing.npz"))
    for domain, split in (("real", "test"), ("real", "train"), ("synthetic", "val")):
        records.append(dict(records[0], unit_id="FORBIDDEN-"+split+domain, sequence_id="FORBIDDEN", domain=domain, split=split,
                            artifact_path="missing.npz"))
    config = {"methods": ["rgb_diff", "msdzip_abs"], "dataset": "FAKE_DATASET_DO_NOT_OPEN"}
    contract = {"config": config, "inputs": {"files_sha256": {}}, "sequence_ids": list(sequences) + [records[6]["sequence_id"]]}
    atomic_json(run / "resolved_config.json", config)
    atomic_json(run / "experiment_contract.json", contract)
    atomic_json(snapshot / "2026-10-17_source_units.json", records)
    atomic_json(snapshot / "2026-10-17_quick_summary.json", {"source_run": str(run), "calibration": [{"threshold": .5}]})

    def adapter(got_config, selected, got_contract):
        assert got_config == config and got_contract == contract
        assert set(selected) == set(sequences)
        def loader(sequence, index):
            reads.append((sequence["sequence_id"], index))
            return frames[(sequence["sequence_id"], index)]
        return sequences, loader, {"categories": [{"id": 3, "name": "wall"}, {"id": 9, "name": "crack</script>"}],
                                   "category_to_channel": {"3": 0, "9": 1}}
    return run, snapshot, adapter, frames, records, reads


def tree_hashes(root):
    return {str(path.relative_to(root)): sha256(path) for path in root.rglob('*') if path.is_file()}


class PreparationTests(unittest.TestCase):
    def prepare(self, run, snapshot, out, adapter):
        with contextlib.redirect_stdout(io.StringIO()):
            return prep.prepare(run, snapshot, out, input_adapter=adapter)

    def test_all_val_pairs_gt_and_frozen_inputs_no_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, snapshot, adapter, frames, records, reads = fixture(base)
            before_run, before_snapshot = tree_hashes(run), tree_hashes(snapshot)
            original_frames = {key: {name: value.copy() for name, value in item.items()} for key, item in frames.items()}
            original_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name == "torch" or name.startswith(("torch.", "transformers", "facade_change.methods")):
                    raise AssertionError("Preparation imported GPU/model stack: " + name)
                return original_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded):
                manifest = self.prepare(run, snapshot, base / "prepared", adapter)
            self.assertEqual(manifest["summary"], dict(pairs=7, ready_pairs=6, facades=2, recommended_pairs=2))
            self.assertEqual(len(reads), 8)  # Each warped frame used once per chronology.
            self.assertEqual(before_run, tree_hashes(run))
            self.assertEqual(before_snapshot, tree_hashes(snapshot))
            for key, old in original_frames.items():
                for name, value in old.items():
                    np.testing.assert_array_equal(value, frames[key][name])
            self.assertTrue(all("FORBIDDEN" not in row['sequence_id'] for row in manifest['pairs']))
            self.assertEqual(len(manifest['source_records']), len(records))  # Metadata provenance retained.
            for row in manifest["pairs"]:
                if row["status"] != "ready":
                    continue
                arrays = load_pair(base / "prepared", row)
                old, new = frames[(row["sequence_id"], row["frame"]-1)], frames[(row["sequence_id"], row["frame"])]
                np.testing.assert_array_equal(arrays["reference"], old["rgb"])
                np.testing.assert_array_equal(arrays["source"], new["rgb"])
                self.assertTrue(np.all(arrays['labels'][~arrays['support']] == 255))
                for name in ("exposure", "shadow"):
                    np.testing.assert_array_equal(arrays[name][~arrays['support']], arrays['source'][~arrays['support']])
                if row["frame"] == 1:
                    self.assertEqual(row["positive_pixels"], 9)
                    self.assertTrue(row["recommended"])
                elif row["frame"] == 2:
                    self.assertEqual(row["positive_pixels"], 0)
                    self.assertGreater(row["negative_pixels"], 0)
                else:
                    self.assertTrue(np.all(arrays["labels"] == 255))
                    self.assertFalse(row["recommended"])
            self.assertEqual(manifest['source_summary']['calibration'], [{'threshold': .5}])

    def test_repeat_is_deterministic_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, snapshot, adapter, *_ = fixture(base)
            first = self.prepare(run, snapshot, base / "one", adapter)
            second = self.prepare(run, snapshot, base / "two", adapter)
            self.assertEqual(first, second)
            self.assertEqual(sha256(base / "one" / "manifest.json"), sha256(base / "two" / "manifest.json"))
            with self.assertRaises(FileExistsError):
                self.prepare(run, snapshot, base / "one", adapter)
            with self.assertRaises(FileExistsError):
                self.prepare(run, snapshot, run / "child", adapter)

    def test_gallery_binds_explicit_review_selection_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, snapshot, adapter, *_ = fixture(base)
            self.prepare(run, snapshot, base / "prepared", adapter)
            root, manifest = load_prepared(base / "prepared")
            page = (root / "gallery.html").read_text()
            self.assertIn(sha256(root / 'manifest.json'), page)
            self.assertIn('loading="lazy"', page)
            self.assertNotIn('value="pair-', page.split('type="application/json">')[-1].split('</script>')[0])
            self.assertNotIn('B<script>alert(1)</script>', page)
            self.assertIn('B&lt;script&gt;alert(1)&lt;/script&gt;', page)
            self.assertNotIn(' checked', page)
            self.assertNotIn('https://', page)
            ready = [row for row in manifest['pairs'] if row['status'] == 'ready']
            selection = dict(schema_version=1, kind='real_robustness_selection', reviewed=True,
                             manifest_sha256=sha256(root / 'manifest.json'), pair_ids=[row['pair_id'] for row in ready])
            path = base / 'reviewed_selection.json'
            atomic_json(path, selection)
            self.assertEqual(len(load_selection(path, root, manifest)[1]), 6)
            for change in ({'pair_ids': []}, {'reviewed': False}, {'manifest_sha256': 'wrong'},
                           {'pair_ids': [next(row['pair_id'] for row in manifest['pairs'] if row['status'] == 'excluded')]}):
                atomic_json(path, {**selection, **change})
                with self.assertRaises(ValueError):
                    load_selection(path, root, manifest)

    def test_provenance_and_rgb_mismatch_rejected(self):
        for corruption in ('config', 'rgb', 'hash', 'traversal', 'split'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                run, snapshot, adapter, frames, records, _ = fixture(base)
                if corruption == 'config':
                    atomic_json(run / 'resolved_config.json', {'different': True})
                elif corruption == 'rgb':
                    frames[next(iter(frames))]['rgb'][0, 0, 0] += 1
                elif corruption in ('hash', 'traversal'):
                    records[0]['artifact_sha256' if corruption == 'hash' else 'artifact_path'] = 'bad' if corruption == 'hash' else '../escape.npz'
                    atomic_json(snapshot / '2026-10-17_source_units.json', records)
                else:
                    inner = adapter
                    def adapter(config, selected, contract):
                        sequences, loader, provenance = inner(config, selected, contract)
                        sequences[next(iter(sequences))]['split'] = 'test'
                        return sequences, loader, provenance
                with self.assertRaises(ValueError):
                    self.prepare(run, snapshot, base / 'prepared', adapter)
                self.assertFalse((base / 'prepared' / 'manifest.json').exists())

    def test_concurrent_metadata_change_blocks_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, snapshot, original, *_ = fixture(base)
            def adapter(config, selected, contract):
                result = original(config, selected, contract)
                with (snapshot / '2026-10-17_quick_summary.json').open('a') as stream:
                    stream.write('\n')
                return result
            with self.assertRaisesRegex(ValueError, 'метаданные изменились'):
                self.prepare(run, snapshot, base / 'prepared', adapter)
            self.assertFalse((base / 'prepared' / 'manifest.json').exists())
            self.assertEqual(json.loads((base / 'prepared' / 'run.json').read_text())['status'], 'failed')

    def test_recommendation_balances_facades_without_scores(self):
        rows = []
        for building in 'ABCDEF':
            for i in range(4):
                rows.append(dict(pair_id=building+str(i), building_id=building, status='ready', positive_pixels=100-i,
                                 changed_channels=[i % 2], reference_year=2000, source_year=2001+(i == 3),
                                 sequence_id=building+str(i), view_id=building, crop_xywh=[i*8, 0, 16, 16]))
        chosen = prep.recommend(rows)
        self.assertEqual(len(chosen), 12)
        for building in 'ABCDEF':
            self.assertEqual(sum(x.startswith(building) for x in chosen), 2)
        self.assertEqual(chosen, prep.recommend(list(reversed(rows))))
        rows.append(dict(rows[0], pair_id='no-event', positive_pixels=0))
        self.assertEqual(chosen, prep.recommend(rows))


if __name__ == '__main__':
    unittest.main()
