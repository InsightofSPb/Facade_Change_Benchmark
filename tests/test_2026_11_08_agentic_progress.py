"""Terminal diagnostics are live stderr output, not model or JSON protocol changes."""
import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from facade_change.agentic import cli
from facade_change.agentic.acceptance import fixture
from facade_change.agentic.config import defaults, write_new_config
from facade_change.io import read_json, write_json
from tests import test_2026_11_08_agentic_cli as fixtures


class FlushedText(io.StringIO):
    """Observe actual flushes while the model call remains blocked."""
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.flushes = 0

    def write(self, text):
        with self.condition:
            return super().write(text)

    def flush(self):
        with self.condition:
            self.flushes += 1
            self.condition.notify_all()
            return super().flush()

    def snapshot(self):
        with self.condition:
            return self.getvalue()

    def wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not predicate(self.getvalue()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True


class TerminalProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = defaults()
        self.config.update(store_root=str(self.root / "store"), cache_root=str(self.root / "cache"),
                           knowledge_root=str(self.root / "knowledge"), methods=["ssim"], modes=["pair"])
        self.config_path = write_new_config(self.root / "config.json", self.config)
        pairs = fixture(self.root / "fixture")
        self.case_path = self.root / "cases.json"
        write_json(self.case_path, {"cases": pairs[:1]})

    def arguments(self, name, quiet=False):
        return ["run", "--config", str(self.config_path), "--cases", str(self.case_path),
                "--out", str(self.root / name), *(["--quiet"] if quiet else [])]

    def invoke(self, arguments, pool=fixtures.ContractPool):
        output, progress = io.StringIO(), FlushedText()
        with patch("facade_change.agentic.models.ModelPool", pool), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress):
            code = cli.main(arguments)
        return code, json.loads(output.getvalue()), progress.snapshot(), progress.flushes

    @staticmethod
    def semantic_summary(result):
        # Each invocation has a new result directory and its own measured time.
        return {key: value for key, value in result.items() if key not in {"elapsed_seconds", "report_path"}}

    def test_quiet_is_available_on_all_long_running_commands(self):
        for command in ("run", "smoke", "check", "history"):
            arguments = [command, "--out", str(self.root / command), "--quiet"]
            if command == "history":
                arguments += ["--histories", str(self.root / "histories"), "--track-id", "track",
                              "--available-at", "observation_date"]
            with self.subTest(command=command):
                self.assertTrue(cli.parser().parse_args(arguments).quiet)

    def test_timeout_does_not_present_unobserved_tokens_as_measured_zeros(self):
        output = io.StringIO()
        progress = cli.TerminalProgress(stream=output)
        self.addCleanup(progress.close)
        progress("pool_model_returned", role="qwen", op="generate", status="error",
                 error="Worker reply timed out", elapsed_seconds=30.,
                 usage={"input_tokens": 0, "output_tokens": 0, "incomplete": True,
                        "unobserved_worker_cost": True})
        text = output.getvalue()
        self.assertIn("Worker reply timed out", text)
        self.assertIn("учёт неполный", text)
        self.assertNotIn("input_tokens=0", text)
        self.assertNotIn("output_tokens=0", text)

    def test_model_wait_progress_is_flushed_before_completion_and_stdout_is_one_json(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingPool(fixtures.ContractPool):
            def call(self, role, request):
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("Test synchronization timed out")
                return super().call(role, request)

        output, progress = io.StringIO(), FlushedText()
        codes, failures = [], []

        def run():
            try:
                codes.append(cli.main(self.arguments("live")))
            except BaseException as error:
                failures.append(error)

        with patch("facade_change.agentic.models.ModelPool", BlockingPool), \
                patch.object(cli, "HEARTBEAT_SECONDS", .02), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(10), failures)
                self.assertEqual(output.getvalue(), "", "No final JSON before model completion")
                self.assertTrue(progress.wait_for(lambda text: any(
                    "ожидание:" in line and "qwen" in line.lower() and "прошло" in line
                    for line in text.splitlines())), "A waiting heartbeat must arrive before release")
                before_completion = progress.snapshot()
                self.assertIn("synthetic-2020", before_completion)
                self.assertIn("pair", before_completion)
                self.assertGreater(progress.flushes, 1)
                self.assertTrue(thread.is_alive())
                self.assertNotIn("%", before_completion)
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(codes, [0])
        result = json.loads(output.getvalue())
        self.assertEqual(result["errors"], [])
        text = progress.snapshot()
        self.assertIn("input_tokens=40", text)
        self.assertIn("output_tokens=30", text)
        self.assertIn("completed=1", text)
        self.assertIn("abstained=0", text)
        self.assertIn("error=0", text)
        self.assertIn(str(self.root / "live/report/report.html"), text)

    def test_validation_error_is_concrete_and_quiet_preserves_result_and_exit_code(self):
        marker = "FULL_MODEL_RESPONSE_MUST_STAY_IN_FILES"

        class InvalidPool(fixtures.ContractPool):
            def call(self, role, request):
                response = super().call(role, request)
                value = json.loads(response["text"])
                value.update(decision="invalid-decision", visible_difference=marker)
                return {**response, "text": json.dumps(value)}

        self.config["budgets"]["max_invalid_replies"] = 0
        write_json(self.config_path, self.config)
        code, result, progress, flushes = self.invoke(self.arguments("invalid"), InvalidPool)
        quiet_code, quiet_result, quiet_progress, _ = self.invoke(self.arguments("quiet", quiet=True), InvalidPool)
        self.assertEqual((code, quiet_code), (1, 1))
        self.assertIn("invalid_reply_limit", progress)
        self.assertIn("Unknown decision", progress)
        self.assertIn("error=1", progress)
        self.assertNotIn(marker, progress)
        self.assertGreater(flushes, 0)
        self.assertEqual(quiet_progress, "")
        self.assertEqual(self.semantic_summary(result), self.semantic_summary(quiet_result))
        self.assertTrue(Path(result["report_path"]).is_file())
        self.assertTrue(Path(quiet_result["report_path"]).is_file())
        root = self.root / "invalid/cases/synthetic-2020/modes/pair"
        quiet_root = self.root / "quiet/cases/synthetic-2020/modes/pair"
        saved, quiet_saved = read_json(root / "result.json"), read_json(quiet_root / "result.json")
        for key in ("status", "decision", "reason", "error", "limitations"):
            self.assertEqual(saved[key], quiet_saved[key])
        self.assertIn(marker, (root / "messages.json").read_text())

    def test_abstention_reports_decision_and_budget_reason_without_calling_model(self):
        requests = []

        class RecordingPool(fixtures.ContractPool):
            def call(self, role, request):
                requests.append(request)
                return super().call(role, request)

        self.config["budgets"]["max_images"] = 2  # Two RGB plus support need three images.
        write_json(self.config_path, self.config)
        code, result, progress, _ = self.invoke(self.arguments("abstained"), RecordingPool)
        self.assertEqual(code, 0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(requests, [])
        self.assertIn("insufficient_data", progress)
        self.assertIn("image_budget_exhausted", progress)
        self.assertIn("completed=0", progress)
        self.assertIn("abstained=1", progress)
        self.assertIn("error=0", progress)

    def test_check_progress_preserves_json_and_quiet(self):
        class CheckPool(fixtures.ContractPool):
            def call(self, role, request):
                self.last_request = request
                return {"status": "ok", "provenance": {"model": "explicit-check-test-double"}, "usage": {}}

        command = ["check", "--config", str(self.config_path), "--roles", "qwen", "--out"]
        code, result, progress, _ = self.invoke(command + [str(self.root / "check")], CheckPool)
        quiet_code, quiet_result, quiet_progress, _ = self.invoke(
            command + [str(self.root / "check-quiet"), "--quiet"], CheckPool)
        self.assertEqual((code, quiet_code), (0, 0))
        self.assertEqual(self.semantic_summary(result), self.semantic_summary(quiet_result))
        self.assertEqual(result["status"], "completed")
        self.assertIn("qwen", progress)
        self.assertEqual(quiet_progress, "")

    def test_history_legacy_progress_cannot_contaminate_final_json(self):
        # Existing history helpers can print their own progress to stdout.
        # Replace only import/selection; the downstream case pipeline is real.
        def import_history(*args, **kwargs):
            print("Legacy history verification progress", flush=True)
            return {"status": "ok"}

        command = ["history", "--config", str(self.config_path), "--histories", str(self.root / "archive"),
                   "--track-id", "fixture-track", "--available-at", "observation_date", "--out"]
        pairs = read_json(self.case_path)["cases"]
        with patch("facade_change.agentic.store.RegionStore.import_histories", side_effect=import_history), \
                patch("facade_change.agentic.pipeline.pairs_from_history", return_value=pairs):
            code, result, progress, _ = self.invoke(command + [str(self.root / "history")])
            quiet_code, quiet_result, quiet_progress, _ = self.invoke(
                command + [str(self.root / "history-quiet"), "--quiet"])
        self.assertEqual((code, quiet_code), (0, 0))
        self.assertEqual(result["cases"], 1)
        self.assertIn("Legacy history verification progress", progress)
        self.assertEqual(quiet_progress, "")
        self.assertEqual(self.semantic_summary(result), self.semantic_summary(quiet_result))


if __name__ == "__main__":
    unittest.main()
