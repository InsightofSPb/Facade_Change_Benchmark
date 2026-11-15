"""CPU replay of saved WWW replies, never a new inference run or scientific score.

Uses saved public questions, exposure, preparation and support. No model, alignment,
retrieval, reference labels or source-image regeneration. Outputs go to a new run.
"""
import argparse
from contextlib import redirect_stdout
import importlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_change.history_features import Fingerprints, digest
from facade_change.io import read_json, sha256, write_json
from facade_change.agentic.runner import complete_run, new_run

runner = importlib.import_module("facade_change.agentic.2026-11-15_www_runner")


def replay(source, out):
    source = Path(source).resolve()
    if Path(out).resolve().is_relative_to(source):
        raise ValueError("Replay output must be separate from the preserved source run")
    fingerprints = Fingerprints()

    def read(path):
        fingerprints.add(path)
        return read_json(path)

    questions = {q["question_id"]: q for q in read(source / "questions.json")["questions"]}
    contract = read(source / "contract.json")
    journal = new_run(out, {"source_run": str(source), "contract_sha256": sha256(source / "contract.json"),
        "scope": "engineering_reply_replay_only; not rescoring the interrupted run"}, "www_reply_replay")
    rows = []
    try:
        for call_path in sorted((source / "questions").glob("*/*/calls/*-qwen.json")):
            call = read(call_path)
            if call["response"].get("status") != "ok":
                continue
            folder = call_path.parent.parent
            question = questions[folder.parent.name]
            prepared = read(folder.parent / "preparation.json")
            evidence = read(folder / "exposure.json")["evidence"]
            common_path = folder.parent / "common_support.npy"
            fingerprints.add(common_path, prepared["common_support_sha256"])
            common = np.load(common_path, allow_pickle=False)
            if list(common.shape) != question["frame"]["shape"] or common.dtype != bool:
                raise ValueError("Saved common support differs from the public question grid")
            support = {"evidence_id": "support-" + digest([question["question_id"], prepared["compared_observations"]])[:20],
                "geometry": {"crop_shape": list(common.shape), "anchor_sha256": question["frame"]["anchor_sha256"],
                             "bbox_xyxy": question["frame"]["anchor_bbox_xyxy"]}}
            row = {"source_call": str(call_path.relative_to(source)), "source_call_sha256": sha256(call_path),
                "question_id": question["question_id"], "condition": folder.name, "raw_text": call["response"]["text"],
                "normalizations": [], "parsed": None, "status": "invalid", "error": None,
                "generation": call["response"].get("generation")}
            try:
                row["parsed"] = runner.parse_reply(row["raw_text"], row["normalizations"])
                row["answer"] = runner.validate_answer(row["parsed"], question, evidence, prepared, common, support, contract["protocol"])
                row["status"] = "valid_structure"
            except (OSError, ValueError, TypeError, KeyError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            if (folder / "final_raw.json").is_file() and row["parsed"] == read(folder / "final_raw.json"):
                prior = read(folder / "result.json")
                row["original_valid_answer_unchanged"] = prior["answer"] == row.get("answer")
            rows.append(row)
        with redirect_stdout(sys.stderr):
            fingerprints.verify()
        write_json(journal.out / "replay.private.json", {"kind": "private_engineering_replay_diagnostics", "access": "private",
            "not_scientific_scores": True, "image_truth_assessed": False, "input_sha256": fingerprints.files, "cases": rows})
        summary = {"status": "completed", "calls": len(rows), "valid_structure": sum(r["status"] == "valid_structure" for r in rows),
            "invalid": sum(r["status"] == "invalid" for r in rows),
            "original_valid_answers_unchanged": sum(r.get("original_valid_answer_unchanged", False) for r in rows),
            "models_executed": False, "quality_claim": None}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Read-only saved WWW run directory")
    parser.add_argument("--out", required=True, help="New engineering diagnostics directory")
    args = parser.parse_args()
    print(json.dumps(replay(args.run, args.out), ensure_ascii=False))
