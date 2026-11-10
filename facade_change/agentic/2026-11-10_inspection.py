"""Supplied-region adaptive visual inspection: preflight/run/evaluate/demo."""
from __future__ import annotations

import argparse
import contextlib
import copy
import html
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from ..history_features import digest
from ..io import read_json, sha256, write_json
from ..methods.remote import RemotePool
from .config import load_config
from .evaluation import _csv, box_iou
from .models import ModelPool
from .pipeline import prepare_case, _support
from .runner import new_run, complete_run
from .store import PrivateEvidenceError, RegionStore


data = importlib.import_module(".2026-11-10_inspection_data", __package__)
policy_code = importlib.import_module(".2026-11-10_inspection_policy", __package__)


def preflight(questions, histories, config_path, protocol_path, out):
    config, protocol = load_config(config_path), read_json(protocol_path)
    journal = new_run(out, dict(config=config, protocol=protocol), "facade_inspection_preflight")
    try:
        resolved = data.resolve_inputs(questions, histories, config, protocol)
        write_json(journal.out / "eligibility.json", {k: v for k, v in resolved.items() if k not in {"public", "resolved"}})
        write_json(journal.out / "rgb_inventory.json", [r["inventory"] for r in resolved["resolved"]])
        summary = dict(status="completed" if resolved["status"] == "eligible" else "ineligible",
            **resolved["counts"], eligibility=resolved["eligibility"], scope=data.SCOPE,
            inference_started=False, model_loading=False,
            requirement="Correct and re-review the public export for any ineligible question; no automatic filtering")
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed" if resolved["status"] == "eligible" else "failed", summary=summary)
        return summary
    except BaseException as exc:
        write_json(journal.out / "validation_error.json", dict(error=f"{type(exc).__name__}: {exc}"))
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def prepare_inventory(resolved, config, folder, store, detector_factory=RemotePool):
    started = time.monotonic()
    case = prepare_case(config, resolved["pair"], folder, store, detector_factory)
    if case["status"] not in {"completed", "completed_with_errors"}:
        raise ValueError("Prepared case failed: " + str(case.get("limitations")))
    maps = [m for m in case["maps"] if m["method"] == "anychange"]
    if len(maps) != 1 or maps[0]["status"] != "completed" or not maps[0].get("native_prediction_path"):
        raise ValueError("AnyChange preparation failed or native binary map missing: " + str(maps))
    native = maps[0]
    if native["metadata"].get("output_kind") != "native_mask":
        raise ValueError("AnyChange must return its original native binary union mask")
    inventory = copy.deepcopy(resolved["inventory"])
    if inventory["current_ids"] != [case["reference_observation_id"], case["source_observation_id"]]:
        raise ValueError("Prepared observation IDs differ from exact public/canonical join")
    # This path distinguishes observation chronology from system knowledge time.
    # Legacy prepare_case and WWW retain their original source-date cutoff behavior.
    case.update(cutoff=resolved["question"]["knowledge_cutoff"],
                knowledge_cutoff=resolved["question"]["knowledge_cutoff"],
                observation_cutoff=case["source_date"], protocol_version=data.VERSION)
    for oid in inventory["history_ids"]:
        store.add_observation(inventory["observations"][oid])
    support = _support(case["support_path"], tuple(case["geometry"]["shape"]))
    mask = _support(native["native_prediction_path"], support.shape)
    proposed, components = data.proposals(mask, support)
    preview = np.zeros((*mask.shape, 3), np.uint8)
    for component in proposed["selected"]:
        preview[components == component["component_label"]] = [255, 150, 30]
    preview = Image.fromarray(preview)
    draw = ImageDraw.Draw(preview)
    for component in proposed["selected"]:
        x0, y0, x1, y1 = component["bbox_xyxy"]
        draw.rectangle([x0, y0, x1-1, y1-1], outline=(255, 255, 255))
    preview_path = folder / "proposal_preview.png"
    preview.save(preview_path)
    write_json(folder / "proposals.json", proposed)
    inventory.update(proposals=proposed, proposal_preview_path=str(preview_path), proposal_preview_sha256=sha256(preview_path),
        method=dict(method="anychange", cache_key=native["cache_key"], source_unit_id=native["source_unit_id"],
                    source_artifact_sha256=native["source_artifact_sha256"], metadata=native["metadata"],
                    native_map_path=native["native_prediction_path"], native_map_sha256=sha256(native["native_prediction_path"])))
    case["preparation_costs"].update(inspection_total_seconds=time.monotonic()-started,
        cold_loading_seconds=None, warm_inference_seconds=None,
        measurement_reason="existing detector adapter reports combined scoring/loading time only",
        candidate_generation_included=True, cache_source_scoring_seconds=native.get("original_scoring_seconds"))
    case["inventory_sha256"] = digest(inventory)
    write_json(folder / "inventory.json", inventory)
    write_json(folder / "case.json", case)
    return case, inventory


def prediction_wrapper(question_sha, rows):
    return dict(schema_version=1, kind="facade_qa_predictions", model_manifest_sha256=question_sha,
                ranking_source=dict(kind="single_box", source="saved fresh Qwen judge result; no padding at K=3/5/10"), cases=rows)


def render_report(root, records, title="Bounded supplied-region inspection"):
    """A compact static report with direct links to all persisted evidence/traces."""
    root = Path(root)
    def link(path, label):
        if not Path(path).is_file():
            return html.escape(label + " (unavailable)")
        relative = os.path.relpath(path, root)
        return '<a href="' + html.escape(relative, quote=True) + '">' + html.escape(label) + '</a>'
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title) + '</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}img{max-height:220px;max-width:320px}article{border-top:1px solid #aaa;padding:1rem 0}pre{white-space:pre-wrap}</style>',
        '<h1>' + html.escape(title) + '</h1><p>' + data.SCOPE +
        '. No whole-facade discovery or compression benefit is established. Geometric coverage is not visibility.</p>']
    for record in records:
        chunks.append('<article><h2>' + html.escape(record["question_id"] + " / " + record.get("policy", "preparation")) + '</h2>')
        chunks.append('<pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre>')
        for key in ("case_path", "inventory_path", "result_path", "trace_path", "judge_path", "costs_path"):
            if record.get(key):
                chunks.append(link(record[key], key) + ' ')
        for path, caption in record.get("panels", []):
            chunks.append('<figure><img src="' + html.escape(os.path.relpath(path, root), quote=True) +
                          '"><figcaption>' + html.escape(caption) + '</figcaption></figure>')
        chunks.append('</article>')
    chunks.append('</html>')
    (root / "report.html").write_text("\n".join(chunks), encoding="utf-8")


def run(questions, histories, config_path, protocol_path, out, *, pool_factory=ModelPool,
        detector_factory=RemotePool, check_resources=True):
    supplied_config, protocol = load_config(config_path), data.validate_protocol(read_json(protocol_path))
    config = copy.deepcopy(supplied_config)
    config.update(store_root=str(Path(out).resolve() / "region-store"), methods=["anychange"],
                  diagnostic_baseline=None, fusion=None, retrieval_encoder=None)
    journal = new_run(out, dict(supplied_config=supplied_config, execution_config=config, protocol=protocol), "facade_inspection")
    wrappers = {p: [] for p in protocol["policies"]}
    outcomes, reports, prepared = [], [], []
    try:
        resolved = data.resolve_inputs(questions, histories, supplied_config, protocol, check_resources=check_resources)
        write_json(journal.out / "protocol.json", protocol)
        write_json(journal.out / "eligibility.json", {k: v for k, v in resolved.items() if k not in {"public", "resolved"}})
        write_json(journal.out / "input_manifest.json", dict(questions_path=str(Path(questions).resolve()),
            questions_sha256=sha256(questions), input_sha256=resolved["input_sha256"],
            cohort=resolved["public"]["questions"], scope=data.SCOPE, public_protocol="supplied_region_qa"))
        if resolved["status"] != "eligible":
            raise ValueError("Ineligible fixed cohort; see eligibility.json. Correct the reviewed export before inference")
        if protocol["policies"] == ["heuristic", "agent"] and len(resolved["resolved"]) != 2:
            raise ValueError("The C/D smoke profile requires exactly two approved public questions")
        with RegionStore(config["store_root"]) as store:
            for selected in resolved["resolved"]:
                question = selected["question"]
                folder = journal.out / "cases" / question["question_id"]
                journal.event("inspection_preparation_started", question_id=question["question_id"])
                try:
                    case, inventory = prepare_inventory(selected, config, folder, store, detector_factory)
                    prepared.append((selected, case, inventory, None))
                except PrivateEvidenceError:
                    raise
                except Exception as exc:
                    prepared.append((selected, None, None, f"{type(exc).__name__}: {exc}"))
                journal.event("inspection_preparation_finished", question_id=question["question_id"], error=prepared[-1][-1])
            # Every inventory is frozen before any selector or judge runs.
            for selected, case, inventory, error in prepared:
                question = selected["question"]
                folder = journal.out / "cases" / question["question_id"]
                for policy in protocol["policies"]:
                    destination = folder / "policies" / policy
                    destination.mkdir(parents=True)
                    pool = None
                    started = time.monotonic()
                    try:
                        if error:
                            raise ValueError("Technical preparation failure: " + error)
                        service = data.InspectionTools(store, case, destination, inventory, protocol["max_native_crop_pixels"])
                        pool = pool_factory(config["models"], destination / "worker_logs", gpu_mode="sequential",
                            timeout=config["worker_timeout"], cache_dir=Path(config["cache_root"]) / "models")
                        if hasattr(pool, "progress"):
                            pool.progress = journal.event
                        result = policy_code.run_policy(policy, question, case, service, pool, destination, protocol,
                                                        event=journal.event, started_at=started)
                    except PrivateEvidenceError:
                        raise
                    except Exception as exc:
                        result = dict(schema_version=1, case_id=question["question_id"], region_id=question["region_id"],
                            status="error", decision=None, localization=None, policy=policy,
                            error=f"{type(exc).__name__}: {exc}", reason="preparation_error" if error else "technical_error")
                        write_json(destination / "result.json", result)
                    finally:
                        if pool:
                            pool.close()
                    elapsed = time.monotonic()-started
                    if elapsed > protocol["caps"]["max_seconds"]:
                        result.update(status="error", decision=None, localization=None, reason="budget_limited",
                                      error="policy_case_time_budget_exhausted_including_worker_lifecycle")
                        write_json(destination / "result.json", result)
                        state_path = destination / "state.json"
                        if state_path.exists():
                            state = read_json(state_path)
                            state.update(result=result, lifecycle_budget_exhausted=True)
                            write_json(state_path, state)
                        journal.event("inspection_lifecycle_budget_exhausted", question_id=question["question_id"], policy=policy, elapsed_seconds=elapsed)
                    costs_path = destination / "costs.json"
                    costs = read_json(costs_path) if costs_path.exists() else dict(incomplete=True, reason="inference_not_started", model_requests=0)
                    costs.update(incremental_policy_seconds=elapsed,
                        preparation_seconds=case["preparation_costs"]["inspection_total_seconds"] if case else None,
                        total_policy_case_seconds=(elapsed + case["preparation_costs"]["inspection_total_seconds"]) if case else None,
                        scope="preparation recorded once on case; incremental policy includes worker lifecycle; shared caches remain explicit")
                    write_json(costs_path, costs)
                    wrappers[policy].append(dict(question_id=question["question_id"], result_path=str(destination / "result.json"),
                                                 result_sha256=sha256(destination / "result.json")))
                    outcomes.append(dict(question_id=question["question_id"], policy=policy, status=result["status"],
                                         error=result.get("error"), inventory_sha256=case.get("inventory_sha256") if case else None))
                    panels = [(o["rgb_path"], "Compared RGB " + o["date"]) for o in question["observations"]]
                    if inventory:
                        panels.append((inventory["proposal_preview_path"], "Selected AnyChange components; area heuristic, not confidence"))
                    for image_path in sorted((destination / "crops").glob("*/rgb.png")):
                        geometry = read_json(image_path.parent / "geometry.json")
                        oid = geometry["observation_id"]
                        panels.append((str(image_path), inventory["observations"][oid]["date"] + " / " + oid + " / " + str(geometry["requested_bbox_xyxy"])))
                    reports.append(dict(question_id=question["question_id"], policy=policy, status=result["status"],
                        case_path=str(folder / "case.json"), inventory_path=str(folder / "inventory.json"),
                        result_path=str(destination / "result.json"), trace_path=str(destination / "trace.jsonl"),
                        judge_path=str(destination / "judge_messages.json"), costs_path=str(costs_path), panels=panels))
        for policy, rows in wrappers.items():
            write_json(journal.out / (policy + "-predictions.json"), prediction_wrapper(sha256(questions), rows))
        render_report(journal.out, reports)
        counts = {p: {status: sum(r["policy"] == p and r["status"] == status for r in outcomes)
                      for status in ("completed", "abstained", "error", "missing")} for p in protocol["policies"]}
        summary = dict(status="completed_with_errors" if any(r["status"] == "error" for r in outcomes) else "completed",
            scope=data.SCOPE, questions=len(resolved["public"]["questions"]), policies=protocol["policies"],
            outcomes=outcomes, counters=counts, report_path=str(journal.out / "report.html"),
            scientific_benefit="not established; owner-reviewed real smoke and pilot required")
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, summary["status"], summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def _at(group, threshold=.25):
    return next(r for r in group["localization"] if r["k"] == 1 and r["iou_threshold"] == threshold)


def _f1(row):
    denominator = 2*row["matched"] + row["unmatched_predictions"] + row["missed"]
    return 2*row["matched"]/denominator if denominator else None


def compare_evaluations(evaluations):
    first = next(iter(evaluations.values()))
    # Eligible-event counts depend only on reviewed GT, never on prediction success.
    positive = sorted(b for b, group in first["buildings"].items() if _at(group)["eligible_events"] > 0)
    stable_only = sorted(b for b, group in first["buildings"].items()
                         if _at(group)["eligible_events"] == 0 and group["coverage"]["assessable_stable_cases"] > 0)
    rows, means = [], {}
    for threshold in (.25, .5):
        for building in sorted(first["buildings"]):
            row = dict(building_id=building, iou_threshold=threshold, k=1, primary_member=building in positive,
                       stable_only=building in stable_only)
            for policy, evaluation in evaluations.items():
                counters = _at(evaluation["buildings"][building], threshold)
                if counters["eligible_events"] != _at(first["buildings"][building], threshold)["eligible_events"]:
                    raise ValueError("Evaluator changed the frozen eligible event cohort")
                row[policy + "_f1"] = _f1(counters)
                for key in ("matched", "missed", "unmatched_predictions", "eligible_events", "region_recall", "region_precision"):
                    row[policy + "_" + key] = counters[key]
            row["agent_minus_heuristic"] = (row["agent_f1"]-row["heuristic_f1"]
                if row.get("agent_f1") is not None and row.get("heuristic_f1") is not None else None)
            rows.append(row)
        for policy in evaluations:
            values = [r[policy + "_f1"] for r in rows if r["iou_threshold"] == threshold and r["primary_member"]]
            if any(v is None for v in values):
                raise ValueError("Positive-event building F1 must remain defined including missing predictions")
            means[f"{policy}@{threshold}"] = sum(values)/len(values) if values else None
    a, h = means.get("agent@0.25"), means.get("heuristic@0.25")
    return dict(primary=dict(name="D minus C mean building event F1", k=1, iou_threshold=.25,
        positive_event_buildings=positive, agent= a, heuristic=h, difference=a-h if a is not None and h is not None else None),
        stable_only_buildings=stable_only, mean_building_f1=means, buildings=rows,
        coverage={p: e["pooled"]["coverage"] for p, e in evaluations.items()},
        evidence_access="A pair only; B fixed history; identical frozen evidence universe/actions/caps only for C/D",
        statistical_unit="building; no crop-level significance claims")


def proposal_diagnostics(evaluation, run_root):
    """Diagnostic coverage only; matching is taken exclusively from saved QA counts."""
    at = _at(evaluation["pooled"])
    matched = {m["event_id"] for m in at["matches"]}
    rows = []
    for event in evaluation["pooled"]["events"]:
        covered, unknown = False, False
        for representation in event["representations"]:
            path = run_root / "cases" / representation["question_id"] / "proposals.json"
            if not path.is_file():
                unknown = True
                continue
            proposals = read_json(path)["selected"]
            covered |= any(box_iou(p["bbox_xyxy"], representation["bbox_xyxy"]) >= .25 for p in proposals)
        hit = event["event_id"] in matched
        rows.append(dict(event_id=event["event_id"], proposal_covered=covered if covered or not unknown else None,
                         missed_despite_proposal=covered and not hit,
                         recovered_outside_proposals=hit and not covered if not unknown else None))
    return dict(definition="Any selected proposal bbox at IoU>=0.25 against any reviewed event representation; empty-map fallback is not a proposal",
                events=rows, eligible_events=len(rows), proposal_covered=sum(r["proposal_covered"] is True for r in rows),
                unavailable=sum(r["proposal_covered"] is None for r in rows),
                missed_despite_proposal=sum(r["missed_despite_proposal"] for r in rows),
                recovered_outside_proposals=sum(r["recovered_outside_proposals"] is True for r in rows))


def evaluate(questions, answers, run_path, out):
    evaluator = importlib.import_module(".2026-11-10_qa_evaluation", __package__)
    root = Path(run_path).resolve()
    run_record = read_json(root / "run.json")
    saved_hashes = run_record.get("artifact_sha256", {})
    if run_record.get("kind") != "facade_inspection" or not run_record.get("status", "").startswith("completed"):
        raise ValueError("Evaluation requires a completed inspection run, including completed_with_errors")
    # Prediction failures stay per-case. Cohort/protocol/inventory corruption is fatal.
    protected = {"input_manifest.json", "protocol.json", "source_snapshot.json"}
    protected.update(p for p in saved_hashes if p.endswith(("/inventory.json", "/proposals.json", "/case.json"))
                     or "crops" in Path(p).parts or "maps" in Path(p).parts or p.endswith("/proposal_preview.png"))
    for relative in protected:
        if relative not in saved_hashes or sha256(root / relative) != saved_hashes[relative]:
            raise ValueError("Frozen inspection provenance changed: " + relative)
    inputs = read_json(root / "input_manifest.json")
    if inputs["questions_sha256"] != sha256(questions) or inputs["cohort"] != read_json(questions)["questions"]:
        raise ValueError("Evaluation must use the exact frozen public cohort")
    protocol = data.validate_protocol(read_json(root / "protocol.json"))
    if protocol["questions_sha256"] != sha256(questions):
        raise ValueError("Run protocol/public manifest mismatch")
    journal = new_run(out, dict(run=str(root), questions_sha256=sha256(questions), protocol=protocol), "facade_inspection_evaluation")
    try:
        evaluations = {}
        for policy in protocol["policies"]:
            wrapper = root / (policy + "-predictions.json")
            if not wrapper.is_file():
                # Retain the entire approved cohort when a policy never produced predictions.
                wrapper = journal.out / (policy + "-predictions.json")
                write_json(wrapper, prediction_wrapper(sha256(questions), []))
            else:
                if sha256(wrapper) != saved_hashes.get(wrapper.name):
                    raise ValueError("Saved prediction wrapper changed: " + wrapper.name)
                write_json(journal.out / (policy + "-predictions.json"), read_json(wrapper))
            evaluator.evaluate(questions, answers, wrapper, journal.out / policy)
            evaluations[policy] = read_json(journal.out / policy / "evaluation.json")
        comparison = compare_evaluations(evaluations)
        comparison.update(version=data.VERSION, scope=data.SCOPE, public_protocol="supplied_region_qa",
            proposal_diagnostics={p: proposal_diagnostics(e, root) for p, e in evaluations.items()},
            source_run=str(root), questions_sha256=sha256(questions),
            inspection_protocol_sha256=sha256(root / "protocol.json"),
            evaluation_sha256={p: sha256(journal.out / p / "evaluation.json") for p in evaluations})
        write_json(journal.out / "2026-11-10-comparison.json", comparison)
        _csv(journal.out / "2026-11-10-comparison.csv", comparison["buildings"])
        records = [dict(question_id="fixed-cohort", policy=p, **e["pooled"]["coverage"],
                        result_path=str(journal.out / p / "evaluation.json"), trace_path=str(root / "report.html"))
                   for p, e in evaluations.items()]
        render_report(journal.out, records, "PRIVATE — inspection comparison; see JSON/CSV for building F1")
        summary = dict(status="completed", primary=comparison["primary"], report_path=str(journal.out / "report.html"))
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = cli.add_subparsers(dest="task", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name, allow_abbrev=False)
        for flag in ("questions", "histories", "config", "protocol", "out"):
            command.add_argument("--" + flag, required=True)
    command = commands.add_parser("evaluate", allow_abbrev=False)
    for flag in ("questions", "answers", "run", "out"):
        command.add_argument("--" + flag, required=True)
    commands.add_parser("demo", allow_abbrev=False).add_argument("--out", required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        # Existing fingerprint progress is diagnostic; stdout remains one JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            if args.task == "demo":
                result = importlib.import_module(".2026-11-10_inspection_demo", __package__).demo(args.out)
            elif args.task == "evaluate":
                result = evaluate(args.questions, args.answers, args.run, args.out)
            else:
                result = (preflight if args.task == "preflight" else run)(args.questions, args.histories, args.config, args.protocol, args.out)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "completed" else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"inspection: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
