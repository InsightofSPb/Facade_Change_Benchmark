"""Small application orchestration over the project's existing RunStore."""
from __future__ import annotations

import copy
import importlib
import json
import re
import subprocess
import time
from pathlib import Path

from ..history_features import digest
from ..io import finish_record, read_json, sha256, write_json
from .config import ROOT


def source_snapshot():
    paths = sorted(p for directory in (ROOT / "facade_change", ROOT / "data_analysis", ROOT / "configs")
                   for p in directory.rglob("*") if p.is_file() and p.suffix in {".py", ".json"})
    result = {"files": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}
    try:
        result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        result["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        result["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["commit"] = None
    result["source_sha256"] = digest(result["files"])
    return result


def new_run(out, config, kind):
    module = importlib.import_module("facade_change.2026-10-14_temporal_experiment")
    journal = module.RunStore(out, config, resume=False)
    journal.record["kind"] = kind
    write_json(journal.out / "run.json", journal.record)
    write_json(journal.out / "config.json", config)
    snapshot = source_snapshot()
    write_json(journal.out / "source_snapshot.json", snapshot)
    journal.contract({"config": config, "source": snapshot, "protocol": "agentic-facade-v1"})
    return journal


def complete_run(journal, status, **values):
    journal.finish(status, **values)
    finish_record(journal.out, journal.record, status)


def _case_id(spec):
    from .store import reject_forbidden
    reject_forbidden(spec)
    if spec.get("case_id"):
        value = spec["case_id"]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
            raise ValueError("case_id must be a safe bounded identifier")
        return value
    identity = copy.deepcopy(spec)
    for role in ("reference", "source"):
        row = identity.get(role)
        if isinstance(row, dict):
            for key in ("path", "crop_path", "original_path", "support_path"):
                if row.get(key):
                    row[key + "_sha256"] = sha256(Path(row[key]).expanduser())
    if identity.get("manifest"):
        identity["manifest_sha256"] = sha256(identity["manifest"])
    return digest(identity)[:24]


def run_cases(config, pairs, out, *, pool_factory=None):
    """Prepare once; free detector workers before starting any VLM worker.

    A failed case remains in the fixed cohort and report. No model error can
    silently become a negative detection. Empty modes explicitly means a
    detector/history-only run, never a simulated Qwen result.
    """
    from .models import ModelPool
    from .pipeline import prepare_case, resolve_pair_spec
    from .reasoning import run_reasoning
    from .report import render_report
    from .retrieval import KnowledgeBase, SearchIndex
    from .store import RegionStore
    from .tools import ToolService

    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Provide at least one explicit pair")
    pairs = [resolve_pair_spec(item) for item in pairs]
    identifiers = [_case_id(item) for item in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate pair IDs in fixed cohort")
    config = copy.deepcopy(config)
    journal = new_run(out, config, "agentic_facade_analysis")
    started = time.monotonic()
    case_records, manifest_cases, errors = [], [], []
    try:
        with RegionStore(config["store_root"]) as regions:
            for identifier, supplied in zip(identifiers, pairs):
                spec = {**supplied, "case_id": identifier}
                case_dir = journal.out / "cases" / identifier
                journal.event("case_started", case_id=identifier)
                try:
                    case = prepare_case(config, spec, case_dir, regions)
                    case["case_id"] = identifier
                    write_json(case_dir / "case.json", case)
                    journal.event("case_prepared", case_id=identifier,
                                  artifact_sha256=sha256(case_dir / "case.json"))
                    if case.get("status") in {"error", "failed", "completed_with_errors"}:
                        errors.append({"case_id": identifier, "stage": "prepare", "status": case["status"],
                                       "limitations": case.get("limitations", [])})
                except Exception as exc:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    case = {"case_id": identifier, "building_id": supplied.get("building_id", "unknown"),
                            "view_id": supplied.get("view_id"), "split": supplied.get("split", "unassigned"),
                            "status": "error", "comparability": "unresolved",
                            "baseline": {"status": "error", "decision": "insufficient_data"},
                            "limitations": [f"{type(exc).__name__}: {exc}"]}
                    write_json(case_dir / "case.json", case)
                    errors.append({"case_id": identifier, "stage": "prepare", "error": str(exc)})
                    journal.event("case_failed", case_id=identifier, error=str(exc))
                case_records.append(case)
                # Freeze the cohort independently of success, failure or produced observation IDs.
                manifest_cases.append({"case_id": identifier, "building_id": supplied["building_id"],
                    "view_id": supplied["view_id"], "split": supplied.get("split", "unassigned"),
                    "reference_date": supplied["reference"].get("date"), "source_date": supplied["source"].get("date"),
                    "reference_sha256": supplied["reference"]["sha256"],
                    "source_sha256": supplied["source"]["sha256"],
                    "input_spec_sha256": digest(supplied)})
            manifest = {"schema_version": 1, "availability_policy": config["availability_policy"],
                        "cases": manifest_cases}
            write_json(journal.out / "input_manifest.json", manifest)
            index = SearchIndex(regions)
            knowledge = KnowledgeBase(config["knowledge_root"])
            # No supplied textbooks means no fabricated textbook facts.
            knowledge.seed_taxonomy(ROOT / "configs/semantic_class_groups.json", available_at="2026-10-11")
            model_pool = (pool_factory or ModelPool)(config["models"], journal.out / "worker_logs",
                gpu_mode=config["gpu_mode"], timeout=config["worker_timeout"],
                cache_dir=Path(config["cache_root"]) / "models")
            try:
                for case in case_records:
                    if case.get("status") in {"error", "failed"} or not case.get("source_observation_id"):
                        continue
                    for mode in config["modes"]:
                        mode_started = time.monotonic()
                        mode_out = journal.out / "cases" / case["case_id"] / "modes" / mode
                        mode_out.mkdir(parents=True, exist_ok=False)
                        service = ToolService(regions, index, knowledge, model_pool, case, mode_out,
                                              method_evidence=case.get("method_evidence", case.get("evidence", [])),
                                              embedding_role=config.get("retrieval_encoder"))
                        try:
                            result = run_reasoning(mode, case, service, model_pool, mode_out,
                                budgets=config["budgets"], event=journal.event)
                            journal.event("mode_completed", case_id=case["case_id"], mode=mode,
                                          status=result.get("status"), decision=result.get("decision"))
                            if result.get("status") in {"error", "failed"}:
                                errors.append({"case_id": case["case_id"], "mode": mode,
                                               "status": result["status"], "limitations": result.get("limitations", [])})
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            errors.append({"case_id": case["case_id"], "mode": mode, "error": error})
                            write_json(mode_out / "result.json", {"status": "error", "mode": mode,
                                "case_id": case["case_id"], "region_id": case.get("region_id"),
                                "decision": "insufficient_data", "evidence_ids": [],
                                "limitations": [error]})
                            journal.event("mode_failed", case_id=case["case_id"], mode=mode, error=error)
                        finally:
                            # Same initial live-model state for every mode. Durable input-bound caches remain explicit.
                            model_pool.close()
                            costs_path = mode_out / "costs.json"
                            costs = read_json(costs_path) if costs_path.is_file() else {}
                            costs["reasoning_elapsed_seconds"] = costs.get("elapsed_seconds")
                            costs["elapsed_seconds"] = time.monotonic() - mode_started
                            costs["lifecycle_policy"] = "owned model workers closed between modes; measured cache hits retained"
                            costs["scope"] = "incremental interpretation including tools and worker lifecycle; shared preparation recorded on case"
                            write_json(costs_path, costs)
            finally:
                model_pool.close()
        summary = {"cases": len(case_records), "modes": config["modes"], "errors": errors,
                   "elapsed_seconds": time.monotonic() - started,
                   "real_vlm_requested": bool(config["modes"]),
                   "cost_scope": "shared preparation once; each mode's incremental interpretation and tools",
                   "cost_regime": "workers unloaded between modes; explicit content-bound caches may be warm",
                   "scope": "Configured model execution; quality requires separate labelled evaluation"}
        write_json(journal.out / "summary.json", summary)
        render_report(journal.out)
        complete_run(journal, "completed_with_errors" if errors else "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        journal.close()
