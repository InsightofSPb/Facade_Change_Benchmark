"""CPU integration tests: no model loading, GT isolation, persistence and resume."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.real_eval import robustness_run as runner
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, read_json, sha256


REPO = Path(__file__).resolve().parents[1]


def fixture(root, methods=("rgb_diff", "geoscd")):
    source, prepared, snapshot = [root / name for name in ("source", "prepared", "snapshot")]
    for path in (source / "maps", source / "method_provenance", prepared / "pairs", snapshot):
        path.mkdir(parents=True)
    records, pairs = [], []
    for number in (1, 2):
        support = np.ones((3, 4), bool)
        support[-1, -1] = False
        arrays = {"reference": np.full((3, 4, 3), 10*number, np.uint8),
                  "source": np.full((3, 4, 3), 100+number, np.uint8), "support": support,
                  "labels": np.zeros((3, 4), np.uint8),
                  "reference_bits": np.ones((3, 4), np.uint16),
                  "source_bits": np.full((3, 4), 3, np.uint16),
                  "exposure": np.full((3, 4, 3), 75+number, np.uint8),
                  "shadow": np.full((3, 4, 3), 65+number, np.uint8)}
        arrays["labels"][0, 0] = 1
        arrays["labels"][~support] = 255
        path = prepared / "pairs" / f"pair-{number}.npz"
        atomic_npz(path, arrays)
        pair = {"pair_id": f"pair-{number}", "status": "ready", "sequence_id": f"sequence-{number}",
                "frame": 1, "building_id": f"facade-{number}", "artifact_path": str(path.relative_to(prepared)),
                "artifact_sha256": sha256(path), "source_unit_id": f"rgb-{number}"}
        pairs.append(pair)
        scores = np.full(support.shape, .25, np.float32)
        scores[~support] = np.nan
        original = {k: arrays[k] for k in ("reference", "source", "support")}
        original.update(scores=scores, labels=np.full(support.shape, 255, np.uint8))
        path = source / "maps" / f"rgb-{number}.npz"
        atomic_npz(path, original)
        records.append({"unit_id": f"rgb-{number}", "method": "rgb_diff", "variant": "frozen",
                        "mode": "previous", "domain": "real", "split": "val", "frame": 1,
                        "sequence_id": pair["sequence_id"], "building_id": pair["building_id"],
                        "status": "completed", "artifact_path": str(path.relative_to(source)),
                        "artifact_sha256": sha256(path), "method_metadata": metadata("rgb_diff")})
    for method in methods:
        atomic_json(source / "method_provenance" / f"full_{method}.json", metadata(method))
    config = {"device": "cpu", "method_options": {method: {} for method in methods}}
    atomic_json(source / "resolved_config.json", config)
    manifest = {"schema_version": 1, "kind": "real_robustness_prepared", "source_run": str(source),
                "snapshot": str(snapshot), "repo_path": str(REPO), "source_config": config,
                "metadata_sha256": {str(source / "resolved_config.json"): sha256(source / "resolved_config.json")},
                "conditions": ["exposure", "shadow"], "pairs": pairs, "source_records": records,
                "categories": [{"id": 0, "name": "fixture"}]}
    atomic_json(prepared / "manifest.json", manifest)
    selection = root / "selection.json"
    atomic_json(selection, {"schema_version": 1, "reviewed": True,
                           "manifest_sha256": sha256(prepared / "manifest.json"),
                           "pair_ids": [pair["pair_id"] for pair in pairs]})
    return source, prepared, selection


def metadata(method):
    return {"method": method, "fixture": True,
            "output_kind": "native_mask" if method in runner.NATIVE_METHODS else "score"}


class FakeFactory:
    def __init__(self, fail=None, bad=False, interrupt=False):
        self.calls, self.activations, self.releases = [], [], 0
        self.fail, self.bad, self.interrupt = fail, bad, interrupt

    def __call__(self, **kwargs):
        owner = self

        class FakePool:
            stderr_path = None
            process = None
            last_worker_retained = False

            def activate(self, method, options):
                owner.activations.append((method, copy.deepcopy(options)))
                return metadata(method)

            def score(self, method, options, reference, source, support):
                # Strict method signature means GT/selection cannot be passed here.
                owner.calls.append((method, reference.copy(), source.copy(), support.copy()))
                if owner.interrupt:
                    owner.interrupt = False
                    raise KeyboardInterrupt()
                if owner.fail == method:
                    owner.fail = None
                    self.last_worker_retained = True
                    raise RuntimeError("Inference worker failed during score: " + runner.RECOVERABLE_SCORE_ERROR + "; stderr log: fixture")
                scores = np.full(support.shape, .25, np.float32)
                native = np.indices(support.shape).sum(axis=0) % 2 == 0
                if method in runner.NATIVE_METHODS:
                    scores = native.astype(np.float32)
                scores[~support] = np.nan
                if owner.bad:
                    scores[support] = np.nan
                result = {"scores": scores, "metadata": metadata(method)}
                if method in runner.NATIVE_METHODS:
                    result["native_prediction"] = native
                return result

            def release(self):
                owner.releases += 1

        return FakePool()


class RobustnessRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.prepared, self.selection = fixture(self.root)
        self.out = self.root / "output"
        self.silent = contextlib.redirect_stdout(io.StringIO())
        self.silent.__enter__()
        self.gpu = patch.object(runner, "_snapshot_gpu")
        self.gpu.start()

    def tearDown(self):
        self.gpu.stop()
        self.silent.__exit__(None, None, None)
        self.temp.cleanup()

    def execute(self, factory, **kwargs):
        return runner.run(self.prepared, self.selection, self.out, methods=["rgb_diff", "geoscd"],
                          pool_factory=factory, **kwargs)

    def test_full_run_native_reuse_gt_isolation_and_resume(self):
        source_before = {str(p): sha256(p) for p in self.source.rglob("*") if p.is_file()}
        factory = FakeFactory()
        summary = self.execute(factory)
        self.assertEqual(summary["completed"], 20)
        self.assertEqual(len(factory.calls), 18)
        self.assertEqual([x[0] for x in factory.activations], ["rgb_diff", "geoscd"])
        self.assertEqual(factory.releases, 2)
        rows = summary["results"]
        h0 = next(r for r in rows if r["hypothesis"] == "H0")
        with np.load(self.out / h0["artifact_path"]) as arrays:
            self.assertTrue(np.all(arrays["labels"][arrays["support"]] == 0))
            np.testing.assert_array_equal(arrays["reference_bits"], arrays["source_bits"])
            self.assertTrue(np.all(arrays["reference"] == 101))
        h1 = next(r for r in rows if r["hypothesis"] == "H1" and r["condition"] == "original")
        self.assertEqual(h1["origin"], "reused_source")
        with np.load(self.out / h1["artifact_path"]) as arrays:
            self.assertEqual(arrays["labels"][0, 0], 1)
        factory2 = FakeFactory()
        summary2 = self.execute(factory2, resume=True)
        self.assertEqual(summary2["completed"], 20)
        self.assertEqual(factory2.calls, [])
        self.assertEqual(len(list((self.out / "attempts").glob("*.json"))), 20)
        self.assertEqual(source_before, {str(p): sha256(p) for p in self.source.rglob("*") if p.is_file()})

    def test_recoverable_geoscd_failure_retry_only_one_task(self):
        factory = FakeFactory(fail="geoscd")
        summary = self.execute(factory)
        self.assertEqual((summary["completed"], summary["failed"]), (19, 1))
        self.assertEqual([x[0] for x in factory.activations], ["rgb_diff", "geoscd"])
        failing = next(r for r in summary["results"] if r["status"] == "failed")
        self.assertNotIn("artifact_path", failing)
        self.assertFalse((self.out / "maps" / f"{failing['unit_id']}.npz").exists())
        skipped = FakeFactory()
        self.assertEqual(self.execute(skipped, resume=True)["failed"], 1)
        self.assertEqual(skipped.calls, [])
        retried = FakeFactory()
        self.assertEqual(self.execute(retried, resume=True, retry_failed=True)["completed"], 20)
        self.assertEqual(len(retried.calls), 1)
        self.assertEqual(len(list((self.out / "attempts").glob(f"{failing['unit_id']}-*.json"))), 2)

    def test_selection_change_and_unreviewed_selection_rejected(self):
        self.execute(FakeFactory())
        value = read_json(self.selection)
        value["pair_ids"] = ["pair-2"]
        atomic_json(self.selection, value)
        with self.assertRaisesRegex(ValueError, "контракт"):
            self.execute(FakeFactory(), resume=True)
        value["reviewed"] = False
        atomic_json(self.selection, value)
        with self.assertRaisesRegex(ValueError, "визуально"):
            self.execute(FakeFactory(), resume=True)

    def test_modified_artifact_fails_before_model_loading(self):
        manifest = read_json(self.prepared / "manifest.json")
        path = self.prepared / manifest["pairs"][0]["artifact_path"]
        path.write_bytes(b"modified")
        factory = FakeFactory()
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.execute(factory)
        self.assertEqual(factory.activations, [])

    def test_missing_gt_array_is_rejected_before_model_loading(self):
        manifest = read_json(self.prepared / "manifest.json")
        pair = manifest["pairs"][0]
        path = self.prepared / pair["artifact_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files if name != "labels"}
        atomic_npz(path, arrays)
        pair["artifact_sha256"] = sha256(path)
        atomic_json(self.prepared / "manifest.json", manifest)
        selection = read_json(self.selection)
        selection["manifest_sha256"] = sha256(self.prepared / "manifest.json")
        atomic_json(self.selection, selection)
        factory = FakeFactory()
        with self.assertRaisesRegex(ValueError, "missing required arrays"):
            self.execute(factory)
        self.assertEqual(factory.activations, [])

    def test_geoscd_overlay_reuses_only_verified_original(self):
        overlay = self.root / "geo"
        (overlay / "units").mkdir(parents=True)
        (overlay / "maps").mkdir()
        atomic_json(overlay / "run.json", {"contract": {"source_run": str(self.source)}})
        manifest = read_json(self.prepared / "manifest.json")
        original = manifest["source_records"][0]
        with np.load(self.source / original["artifact_path"], allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["native_prediction"] = np.ones(arrays["support"].shape, bool)
        arrays["scores"] = np.where(arrays["support"], 1, np.nan).astype(np.float32)
        path = overlay / "maps" / "geo.npz"
        atomic_npz(path, arrays)
        record = {**original, "unit_id": "geo", "method": "geoscd", "artifact_path": "maps/geo.npz",
                  "artifact_sha256": sha256(path), "method_metadata": metadata("geoscd")}
        atomic_json(overlay / "units" / "geo.json", record)
        factory = FakeFactory()
        summary = self.execute(factory, geoscd_run=overlay)
        self.assertEqual(len(factory.calls), 17)
        self.assertEqual(sum(row.get("origin") == "reused_geoscd" for row in summary["results"]), 1)
        # Mutated provenance cannot be reused or bypassed by silently recomputing.
        record["method_metadata"]["fixture"] = False
        atomic_json(overlay / "units" / "geo.json", record)
        self.out = self.root / "out-mutated"
        with self.assertRaisesRegex(ValueError, "происхождение"):
            self.execute(FakeFactory(), geoscd_run=overlay)

    def test_msdzip_uses_recorded_checkpoint_and_dataset_fingerprint(self):
        checkpoint = self.root / "abs.pt"
        checkpoint.write_bytes(b"never loaded in CPU fixture")
        provenance = {**metadata("msdzip_abs"), "checkpoint_path": str(checkpoint),
                      "checkpoint_sha256": sha256(checkpoint), "checkpoint_dataset_verified": True,
                      "input_sha256": {"parent_run": "frozen-dataset"}}
        atomic_json(self.source / "method_provenance" / "full_msdzip_abs.json", provenance)
        manifest = read_json(self.prepared / "manifest.json")
        contract = runner._method_contract("msdzip_abs", manifest, self.source, None)
        self.assertEqual(contract["options"]["checkpoint_path"], str(checkpoint))
        self.assertEqual(contract["options"]["dataset_fingerprint"], provenance["input_sha256"])
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "отсутствует или изменился"):
            runner._method_contract("msdzip_abs", manifest, self.source, None)

    def test_invalid_scores_fail_method_not_zero_and_interrupt_preserves_cache(self):
        summary = self.execute(FakeFactory(bad=True))
        self.assertEqual(summary["status"], "completed_with_failures")
        self.assertTrue(summary["method_failures"])
        self.assertTrue(summary["unprocessed"] > 0)
        # A distinct output can be resumed after SIGINT without redoing committed maps.
        self.out = self.root / "interrupted"
        summary = self.execute(FakeFactory(interrupt=True))
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["completed"], 1)
        summary = self.execute(FakeFactory(), resume=True)
        self.assertEqual(summary["completed"], 20)

    def test_missing_msdzip_checkpoint_rejected_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            runner.run(self.prepared, self.selection, self.out, methods=["msdzip_abs"], pool_factory=FakeFactory())

    def test_completed_map_corruption_is_not_silently_recomputed(self):
        summary = self.execute(FakeFactory())
        path = self.out / summary["results"][0]["artifact_path"]
        path.write_bytes(b"corrupt")
        factory = FakeFactory()
        with self.assertRaisesRegex(RuntimeError, "сохранённый результат"):
            self.execute(factory, resume=True)
        self.assertEqual(factory.calls, [])

    def test_real_rgb_worker_end_to_end_without_models_or_gpu(self):
        manifest = read_json(self.prepared / "manifest.json")
        manifest["source_records"] = []
        atomic_json(self.prepared / "manifest.json", manifest)
        selection = read_json(self.selection)
        selection.update(manifest_sha256=sha256(self.prepared / "manifest.json"), pair_ids=["pair-1"])
        atomic_json(self.selection, selection)
        atomic_json(self.source / "method_provenance" / "full_rgb_diff.json",
                    {"method": "rgb_diff", "output_kind": "score"})
        summary = runner.run(self.prepared, self.selection, self.out, methods=["rgb_diff"])
        self.assertEqual(summary["completed"], 5)
        self.assertEqual(summary["status"], "completed")
        row = next(row for row in summary["results"] if row["hypothesis"] == "H0" and row["condition"] == "exposure")
        with np.load(self.out / row["artifact_path"], allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays["scores"][arrays["support"]], 25/255, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
