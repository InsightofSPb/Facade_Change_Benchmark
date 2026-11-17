"""Offline WWW action/answer audit. Never executes a worker or rewrites a saved run."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json
from facade_change.agentic.runner import new_run, complete_run

runner = importlib.import_module("facade_change.agentic.2026-11-15_www_runner")
curation = importlib.import_module("facade_change.agentic.2026-11-15_www_curation")


class MissingSupport(ValueError):
    pass


class RecordedCoverage:
    """A saved count can establish nonempty support, never its per-pixel footprint."""
    def __init__(self, initial, question):
        self.shape = tuple(question["frame"]["shape"])
        self.count = initial["common_pixels"]
        if type(self.count) is not int or not 0 <= self.count <= self.shape[0] * self.shape[1]:
            raise ValueError("Invalid recorded common-support count")

    def any(self):
        return self.count > 0

    def __getitem__(self, key):
        raise MissingSupport("Per-pixel support is absent from this bundle; box support cannot be replayed")


def baseline_module(ref, expected):
    if not re.fullmatch(r"[0-9a-f]{7,40}", ref):
        raise ValueError("Baseline must be an explicit git commit SHA")
    path = "facade_change/agentic/2026-11-15_www_runner.py"
    raw = subprocess.check_output(["git", "show", ref + ":" + path], cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Baseline source does not match the saved execution contract")
    module = types.ModuleType("facade_change.agentic.saved_www_baseline")
    module.__package__, module.__file__ = "facade_change.agentic", runner.__file__
    exec(compile(raw, ref + ":" + path, "exec"), module.__dict__)
    return module


def context(call, question, exposure, folder, read, fingerprints):
    messages = call["request"]["messages"]
    control = json.loads(messages[-1]["content"][0]["text"])
    initial = json.loads(messages[1]["content"][0]["text"])
    if any(initial[k] != question[k] for k in ("question_id", "question", "task", "task_dates")):
        raise ValueError("Saved request differs from the frozen question")
    shown = set()
    for message in messages:
        if message["role"] != "user":
            continue
        for block in message["content"]:
            if block["type"] == "image":
                shown.add(block.get("evidence_id"))
            elif block["type"] == "text":
                try:
                    metadata = json.loads(block["text"])
                except ValueError:
                    continue
                if isinstance(metadata, dict) and "evidence_id" in metadata:
                    shown.add(metadata["evidence_id"])
    evidence = {key: row for key, row in exposure["evidence"].items() if key in shown}
    local = sorted((r for r in evidence.values() if r["origin"] == "local_photo"), key=lambda r: (r["date"], r["observation_id"]))
    dates = sorted(set(question["task_dates"]))
    compared = [next(r["observation_id"] for r in local if r["date"] == date) for date in (dates[0], dates[-1])]
    prepared, common = {"compared_observations": compared}, RecordedCoverage(initial, question)
    preparation = folder.parent / "preparation.json"
    support_path = folder.parent / "common_support.npy"
    if preparation.is_file() and support_path.is_file():
        prepared = read(preparation)
        fingerprints.add(support_path, prepared["common_support_sha256"])
        common = np.load(support_path, allow_pickle=False)
        if common.dtype != bool or list(common.shape) != question["frame"]["shape"]:
            raise ValueError("Saved support has the wrong grid or dtype")
    support = {"evidence_id": "support-" + digest([question["question_id"], compared])[:20],
        "geometry": {"crop_shape": list(common.shape), "anchor_sha256": question["frame"]["anchor_sha256"],
                     "bbox_xyxy": question["frame"]["anchor_bbox_xyxy"]}}
    return control, evidence, prepared, common, support


def check(module, raw, question, control, evidence, prepared, common, support, protocol):
    result = {"status": "rejected", "normalizations": [], "parsed": None, "normalized": None, "error": None}
    try:
        value = module.parse_reply(raw, result["normalizations"])
        result["parsed"] = value
        final = value.get("type") == "final" or "final" in value or value.get("stage") == "answer" and "decision" in value
        if hasattr(module, "normalize_stage_reply"):
            value, _ = module.normalize_stage_reply(value, control["stage"], result["normalizations"])
        elif control["stage"] == "answer" or final:
            value = module.normalize_final(value, result["normalizations"])
        else:
            value = module.normalize_action(value, result["normalizations"])
        result["normalized"] = value
        if value.get("type") == "final":
            if "selected_abstention_reason" in control and value.get("decision") not in {"insufficient_data", "incomparable"}:
                raise ValueError("The selected abstention requires insufficient_data or incomparable")
            result["answer"] = module.validate_answer(value, question, evidence, prepared, common, support, protocol)
            result["status"] = "validated_final" if isinstance(common, np.ndarray) else "public_fields_passed_using_recorded_coverage"
        elif control["stage"] == "choose_action" and value.get("type") in {"retrieve", "answer", "abstain"}:
            if value["type"] == "retrieve":
                if control["allowed_action"] != "retrieve_or_decide":
                    raise ValueError("Retrieval not allowed at this saved stage/budget")
                if len(value["query"]) > protocol["max_query_chars"] or value["observation_id"] not in [None, *control["task_scope"]["local_observation_ids"]]:
                    raise ValueError("Invalid query length or unshown local query image")
            if value["type"] == "answer" and question["task"] == "archival_qa" and not any(r["origin"] == "archive" for r in evidence.values()):
                raise ValueError("No archive evidence exposed for archival_qa; answer does not dispatch a search")
            result["status"] = "accepted_action"
            if final:
                result["legacy_final_fell_through_as_action"] = True
        else:
            raise ValueError("Expected a final answer in answer stage")
    except (MissingSupport, FileNotFoundError) as error:
        result.update(status="support_validation_unavailable", error=str(error))
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def replay(source, out, baseline_ref=None):
    source = Path(source).resolve()
    if Path(out).resolve().is_relative_to(source):
        raise ValueError("Use a new output outside the preserved source run")
    fingerprints = Fingerprints()
    def read(path):
        fingerprints.add(path)
        return read_json(path)
    contract = read(source / "contract.json")
    fingerprints.add(source / "questions.json", contract["questions_sha256"])
    questions = {q["question_id"]: q for q in runner.public_questions(source / "questions.json")["questions"]}
    baseline = baseline_module(baseline_ref, contract["implementation_sha256"]["2026-11-15_www_runner.py"]) if baseline_ref else None
    journal = new_run(out, {"source": str(source), "source_contract_sha256": sha256(source / "contract.json"),
        "baseline_ref": baseline_ref, "replay_runner_sha256": sha256(runner.__file__), "no_inference": True}, "www_action_replay")
    cases, calls = [], []
    try:
        for folder in sorted((source / "questions").glob("*/*")):
            if not (folder / "calls").is_dir():
                continue
            identifier = folder.parent.name
            print("Replay: " + identifier + " / " + folder.name, file=sys.stderr, flush=True)
            question, exposure = questions[identifier], read(folder / "exposure.json")
            saved = read(folder / "result.json") if (folder / "result.json").is_file() else {"status": "unfinished"}
            ledger = read(folder / "costs.json")
            transcript = read(folder / "messages.json")
            assistant_texts = [b["text"] for m in transcript if m["role"] == "assistant" for b in m["content"] if b["type"] == "text"]
            case_calls = []
            for request_number, path in enumerate(sorted((folder / "calls").glob("*-qwen.json")), 1):
                call = read(path)
                reply_path = folder / ("reply-%03d.json" % request_number)
                saved_reply = read(reply_path) if reply_path.is_file() else {}
                if saved_reply:
                    fingerprints.add(path, saved_reply["source_call_sha256"])
                if call["response"].get("status") != "ok":
                    calls.append({"source_call": str(path), "status": "worker_error", "response": call["response"]})
                    continue
                control, evidence, prepared, common, support = context(call, question, exposure, folder, read, fingerprints)
                raw = call["response"]["text"]
                request_messages = call["request"]["messages"]
                # Stage system prompts may change; all user/evidence/assistant history must persist exactly.
                if request_messages[1:] != transcript[1:len(request_messages)] or raw not in assistant_texts:
                    raise ValueError("Saved request/response does not match messages.json")
                assistant_texts.remove(raw)
                row = {"question_id": identifier, "condition": folder.name, "source_call": str(path), "sha256": sha256(path),
                    "raw_text": raw, "control": control, "generation": call["response"].get("generation"),
                    "saved_parser": saved_reply, "exposed_at_call": list(evidence),
                    "current": check(runner, raw, question, control, evidence, prepared, common, support, contract["protocol"])}
                if baseline:
                    row["baseline"] = check(baseline, raw, question, control, evidence, prepared, common, support, contract["protocol"])
                parsed = row["current"]["parsed"] or {}
                row["raw_discriminators"] = {k: parsed[k] for k in ("stage", "action", "type") if k in parsed}
                calls.append(row)
                case_calls.append(row)
            # The archive was not searched merely because a model emitted the word query.
            text = (saved.get("answer") or {})
            claims = [s for s in [text.get("answer_text", ""), *text.get("limitations", [])]
                      if "архив" in s.lower() and any(w in s.lower() for w in ("отсутств", "нет ", "не найден", "не содерж"))]
            cases.append({"question_id": identifier, "condition": folder.name, "task": question["task"],
                "saved_status": saved["status"], "saved_error": saved.get("error"), "qwen_calls": len(case_calls),
                "saved_transcript_verified": True,
                "search_allowed_at_choice": [r["control"]["allowed_action"] == "retrieve_or_decide" for r in case_calls if r["control"]["stage"] == "choose_action"],
                "raw_choices": [r["raw_discriminators"] for r in case_calls],
                "saved_accepted_actions": [r["saved_parser"]["normalized_action"]["type"] for r in case_calls
                    if r["saved_parser"].get("normalized_action") and not r["saved_parser"].get("validation_error")],
                "saved_parser_errors": [r["saved_parser"]["validation_error"] for r in case_calls if r["saved_parser"].get("validation_error")],
                "tool_files": len(list(folder.glob("retrieval-*.json"))), "exposure_tools": len(exposure["tools"]),
                "retrieval_rounds": ledger.get("retrieval_rounds"),
                "archive_absence_claims_for_human_review": claims if not exposure["tools"] else [],
                "call_results": [{"source_call": r["source_call"], **r["current"]} for r in case_calls]})
        summary = {"status": "completed", "manifest_questions": len(questions), "saved_cases": len(cases), "calls": len(calls),
            "saved_statuses": dict(Counter(r["saved_status"] for r in cases)),
            "current_replay": dict(Counter(r["current"]["status"] for r in calls if "current" in r)),
            "baseline_replay": dict(Counter(r["baseline"]["status"] for r in calls if "baseline" in r)),
            "transport_rejections_now_accepted": sum(r.get("baseline", {}).get("status") == "rejected" and r.get("current", {}).get("status") == "accepted_action" for r in calls),
            "observed_tools": sum(r["exposure_tools"] for r in cases),
            "archive_absence_claim_cases": sum(bool(r["archive_absence_claims_for_human_review"]) for r in cases),
            "models_executed": False, "not_a_counterfactual_run_or_quality_score": True,
            "geometry_scope": "Real support masks when present; otherwise recorded coverage only and box support explicitly unavailable"}
        with redirect_stdout(sys.stderr):
            fingerprints.verify()
        write_json(journal.out / "replay.private.json", {"access": "private_evaluation", "summary": summary,
            "cases": cases, "calls": calls, "input_sha256": fingerprints.files})
        curation.csv_rows(journal.out / "cases.private.csv", [{k: v for k, v in c.items() if k != "call_results"} for c in cases])
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as error:
        complete_run(journal, "failed", error=str(error))
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-ref", help="Optional commit whose runner SHA must equal the saved contract")
    args = parser.parse_args()
    print(json.dumps(replay(args.run, args.out, args.baseline_ref), ensure_ascii=False))
