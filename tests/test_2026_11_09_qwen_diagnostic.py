"""CPU-only bbox feedback, saved-coordinate rendering and two-request diagnostics."""
import copy
import importlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from facade_change.agentic.config import defaults
from facade_change.agentic.reasoning import run_reasoning, validate_final
from facade_change.io import read_json, sha256, write_json
from tests import test_2026_11_08_agentic_reasoning as fixtures


diagnostic = importlib.import_module("scripts.2026-11-09_qwen_diagnostic")


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ToolReasoningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    @staticmethod
    def reply(bbox):
        def make(request):
            value = fixtures.final_reply(request)
            value["localization"] = None if bbox is None else {
                "bbox_xyxy": bbox, "coordinate_system": "region_crop_pixels", "evidence_id": value["evidence_ids"][0]}
            return value
        return make

    def test_zero_area_retry_contains_coordinates_reason_bounds_and_never_repairs_box(self):
        bbox = [160, 128, 160, 128]
        for explicit_null in (False, True):
            with self.subTest(explicit_null=explicit_null):
                pool = fixtures.FakePool([self.reply(bbox), self.reply(None if explicit_null else bbox)])
                service, out = self.fixture.service("retry-" + str(explicit_null), pool)
                result = run_reasoning("pair", self.fixture.case, service, pool, out)
                messages = read_json(out / "messages.json")
                feedback = next(b["text"] for m in messages if m["role"] == "user" for b in m["content"]
                                if b["type"] == "text" and "Your reply was invalid" in b["text"])
                self.assertIn(str(bbox), feedback)
                self.assertIn("width=0", feedback)
                self.assertIn("height=0", feedback)
                self.assertIn("[0, 0, 4, 4]", feedback)
                self.assertEqual(len(pool.requests), 2)
                originals = [json.loads(m["content"][0]["text"]) for m in messages if m["role"] == "assistant"]
                self.assertEqual(originals[0]["localization"]["bbox_xyxy"], bbox)
                if explicit_null:
                    self.assertEqual(result["status"], "completed")
                    self.assertIsNone(result["localization"])
                else:
                    self.assertEqual(result["reason"], "invalid_reply_limit")
                    self.assertEqual(originals[1]["localization"]["bbox_xyxy"], bbox)

    def test_bbox_feedback_uses_declared_grid_and_rejects_nonfinite_reversed_and_outside(self):
        service, _ = self.fixture.service()
        packet = service.base_pair()
        ids = [r["evidence_id"] for r in packet["evidence"]]
        for coordinate, bounds in (("region_crop_pixels", [0, 0, 4, 4]),
                                   ("observation_native", [0, 0, 8, 8]), ("anchor_native", [0, 0, 4, 4])):
            for bbox in ([2, 2, 2, 2], [3, 3, 1, 1], [0, 0, 20, 20], [0, 0, float("nan"), 2]):
                value = fixtures.final_reply({"messages": []}, identifiers=ids)
                value["localization"] = {"evidence_id": ids[0], "coordinate_system": coordinate, "bbox_xyxy": bbox}
                with self.subTest(coordinate=coordinate, bbox=bbox), self.assertRaises(ValueError) as raised:
                    validate_final(value, self.fixture.case, set(ids), service.evidence)
                self.assertIn(coordinate, str(raised.exception))
                self.assertIn(str(bounds), str(raised.exception))

    def test_rendering_maps_native_and_anchor_coordinates_and_preserves_rgb(self):
        service, out = self.fixture.service()
        record = service.base_pair()["evidence"][0]
        native = {"coordinate_system": "observation_native", "bbox_xyxy": [2, 1, 4, 3],
                  "evidence_id": record["evidence_id"], "observation_id": "reference"}
        points, zero = diagnostic.bbox_on_rgb(native, record, [4, 4])
        np.testing.assert_array_equal(points, [[0, 0], [2, 0], [2, 2], [0, 2]])
        self.assertFalse(zero)
        original = copy.deepcopy(record)
        original["artifact_coordinates"]["rgb"] = {"coordinate_system": "observation_native", "observation_id": "reference"}
        points, _ = diagnostic.bbox_on_rgb(native, original, [8, 8])
        np.testing.assert_array_equal(points, [[2, 1], [4, 1], [4, 3], [2, 3]])
        anchor = copy.deepcopy(record)
        anchor["geometry"]["bbox_xyxy"] = [-3, -2, 1, 2]
        points, _ = diagnostic.bbox_on_rgb({"coordinate_system": "anchor_native", "bbox_xyxy": [-3, -2, -1, 1]}, anchor, [4, 4])
        np.testing.assert_array_equal(points, [[0, 0], [2, 0], [2, 3], [0, 3]])
        with self.assertRaisesRegex(ValueError, "owner"):
            diagnostic.bbox_on_rgb({**native, "observation_id": "source"}, record, [4, 4])
        image = next(a["path"] for a in record["artifacts"] if a["role"] == "rgb")
        before = sha256(image)
        zero_location = {**native, "bbox_xyxy": [3, 2, 3, 2]}
        row = diagnostic.draw_bbox({"decision": "change", "localization": zero_location},
            {record["evidence_id"]: record}, out / "zero.png", mode="agent", origin="test", inputs={})
        self.assertEqual(row["status"], "rendered", row)
        self.assertEqual(row["validation"], "invalid: zero-area bbox")
        np.testing.assert_array_equal(row["polygon_on_rgb"], [[1, 1]] * 4)
        self.assertEqual(sha256(image), before)
        with Image.open(row["visualization"]) as picture:
            self.assertEqual(picture.getpixel((1, row["rgb_offset_in_figure"][1] + 1)), (255, 0, 0))
        broken = copy.deepcopy(record)
        del broken["geometry"]["source_to_crop"]
        row = diagnostic.draw_bbox({"localization": native}, {record["evidence_id"]: broken},
                                  out / "missing.png", mode="pair", origin="test", inputs={})
        self.assertEqual(row["status"], "not_rendered")
        self.assertIn("transform", row["reason"])
        self.assertFalse((out / "missing.png").exists())

    def prepare_saved_run(self):
        source = self.root / "saved"
        analysis = source / "analysis"
        analysis.mkdir(parents=True)
        config = defaults()
        config["models"]["qwen"]["python"] = sys.executable
        write_json(analysis / "config.json", config)
        write_json(self.root / "config.json", config)
        for name, modes in (("synthetic-2020", ["pair"]), ("synthetic-2025", ["pair", "fixed_context", "agent"])):
            case = {**self.fixture.case, "case_id": name}
            directory = analysis / "cases" / name
            directory.mkdir(parents=True)
            write_json(directory / "case.json", case)
            for mode in modes:
                box = [2, 2, 2, 2] if mode == "agent" else [0, 0, 2, 2]
                pool = fixtures.FakePool([self.reply(box), self.reply(box)])
                service, out = self.fixture.service(str(directory / "modes" / mode), pool, case)
                run_reasoning(mode, case, service, pool, out)
        return source

    def check_diagnostic(self, oom):
        source = self.prepare_saved_run()
        before = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
        workers, calls = [], []

        class Worker:
            def __init__(self, role, options, log_dir, deadline):
                self.options, self.provenance = options, {"model": "injected CPU worker", "resources": []}
                self.log_path = Path(log_dir) / "test-worker.log"
                self.process = SimpleNamespace(pid=1234 + len(workers))
                self.closed = False
                workers.append(self)

            def request(self, payload, deadline):
                calls.append((self, copy.deepcopy(payload)))
                if oom and len(calls) == 1:
                    return {"status": "error", "error": {"code": "out_of_memory", "message": "test OOM"},
                            "usage": {"model_calls": 1}}
                return {"status": "ok", "text": "Неподсказанный тестовый ответ", "provenance": self.provenance,
                        "usage": {"input_tokens": 13, "output_tokens": 7, "images": 2, "model_calls": 1},
                        "generation": {"actual_output_tokens": 7, "max_new_tokens": payload["max_new_tokens"],
                                       "reached_max_new_tokens": False}}

            def close(self):
                self.closed = True

        output = self.root / "diagnostic"
        with patch("facade_change.agentic.models._Worker", Worker):
            result = diagnostic.run(source, self.root / "config.json", output)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 3 if oom else 2)
        self.assertEqual(len(workers), 2 if oom else 1)
        self.assertIs(calls[-1][0], calls[-2][0])
        self.assertTrue(all(w.closed for w in workers))
        self.assertEqual(result["queries"][0]["usage"]["retries"], int(oom))
        self.assertEqual(result["queries"][1]["usage"]["retries"], 0)
        first = read_json(output / "reference-source/request.json")
        second = read_json(output / "reference-reference/request.json")
        for request in (first, second):
            self.assertEqual(len(request["messages"]), 1)
            self.assertEqual(request["messages"][0]["role"], "user")
            content = request["messages"][0]["content"]
            self.assertEqual([b["text"] for b in content if b["type"] == "text"], [diagnostic.INSTRUCTION, "A", "B"])
            self.assertEqual(len([b for b in content if b["type"] == "image"]), 2)
        paths = lambda r: [b["path"] for b in r["messages"][0]["content"] if b["type"] == "image"]
        self.assertNotEqual(*paths(first))
        self.assertEqual(paths(second), [paths(first)[0]] * 2)
        self.assertEqual(len(result["visualizations"]), 4)
        self.assertTrue(all(row["status"] == "rendered" for row in result["visualizations"]), result["visualizations"])
        self.assertEqual(before, {str(p): sha256(p) for p in source.rglob("*") if p.is_file()})
        self.assertEqual(read_json(output / "run.json")["status"], "completed")
        self.assertEqual(read_json(output / "reference-source/response.json")["generation"]["actual_output_tokens"], 7)
        with self.assertRaises(FileExistsError):
            diagnostic.run(source, self.root / "config.json", output)

    def test_two_independent_requests_reuse_worker_and_leave_saved_run_immutable(self):
        self.check_diagnostic(oom=False)

    def test_only_existing_oom_retry_is_counted_and_recovered_worker_is_reused(self):
        self.check_diagnostic(oom=True)

    def test_changed_saved_rgb_is_rejected_before_starting_worker(self):
        source = self.prepare_saved_run()
        with Image.open(self.fixture.source["crop_path"]) as image:
            changed = np.asarray(image).copy()
        changed[0, 0, 0] ^= 1
        Image.fromarray(changed).save(self.fixture.source["crop_path"])
        with patch("facade_change.agentic.models._Worker") as worker:
            with self.assertRaisesRegex(ValueError, "Saved input changed"):
                diagnostic.run(source, self.root / "config.json", self.root / "changed")
            worker.assert_not_called()
        self.assertEqual(read_json(self.root / "changed/run.json")["status"], "failed")
