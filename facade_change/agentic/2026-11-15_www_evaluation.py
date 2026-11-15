"""Private, fixed-cohort WWW endpoints. No labels are available to the runner."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import secrets

from ..history_features import Fingerprints, digest
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .evaluation import _verify_recorded_artifacts
from .report import _copy
from .runner import complete_run, new_run
from .store import PrivateEvidenceError, reject_forbidden, temporal_value

qa = importlib.import_module(".2026-11-10_questions", __package__)
evaluation = importlib.import_module(".2026-11-10_qa_evaluation", __package__)
runner = importlib.import_module(".2026-11-15_www_runner", __package__)
curation = importlib.import_module(".2026-11-15_www_curation", __package__)
ratio = evaluation.ratio


def plan(questions_path, answers_path, inventory_path, config, out, *, archive_index=None,
         allow_empty_archive=False, protocol=None, synthetic=False):
    """Freeze reviewed stratum/cohort membership before any model outcomes exist."""
    fingerprints = Fingerprints()
    public, labels = evaluation.approved_cohort(questions_path, answers_path, fingerprints)
    inv = qa.read_bound(inventory_path, fingerprints)
    private = qa.read_bound(answers_path, fingerprints)
    review = qa.read_bound(private["review_source"]["path"], fingerprints, private["review_source"]["sha256"])
    originals = qa._load_candidates(review["candidates_path"], fingerprints)
    ids = {r["question_id"] for r in public["questions"]}
    if any(r["source_provenance"]["history_manifest_sha256"] != inv.get("history_manifest_sha256")
           for r in originals["candidates"] if r["question_id"] in ids):
        raise ValueError("Approved questions and exposure inventory use different canonical history revisions")
    views = {(r["building_id"], r["view_id"]): r for r in inv["views"]}
    protocol, budgets = runner.protocol_config(config, protocol)
    archive = runner.archive.ArchiveSearch(archive_index, config, fingerprints, synthetic=synthetic) if archive_index else None
    if archive is None and not allow_empty_archive:
        raise ValueError("Explicit empty-archive plumbing flag required; no scientific retrieval comparison")
    members = []
    for question in public["questions"]:
        view = views.get((question["building_id"], question["view_id"]), {})
        if not view.get("eligible") or view.get("cohort") not in {"development", "evaluation"}:
            raise ValueError("Cohort includes an unaudited/ineligible building; review inventory/exposure before freezing")
        stratum = labels[question["question_id"]].get("archival_stratum")
        if type(stratum) is not bool:
            raise ValueError("Set reviewed archival_stratum=true/false before running models; no outcome-based stratum selection")
        members.append({"question_id": question["question_id"], "building_id": question["building_id"],
                        "cohort": view["cohort"], "archival_stratum": stratum})
        if archive:
            archive.eligible(question)
    contract = runner.execution_contract(questions_path, config, archive.sha256 if archive else None,
                                         runner.CONDITIONS, protocol, budgets, synthetic)
    journal = new_run(out, {"questions_sha256": sha256(questions_path), "answers_sha256": sha256(answers_path),
        "inventory_sha256": sha256(inventory_path), "execution_contract_sha256": digest(contract)}, "www_cohort_freeze")
    try:
        payload = qa._private("www_cohort_plan", created_at=datetime.now(timezone.utc).isoformat(),
            questions_sha256=sha256(questions_path), answers_sha256=sha256(answers_path), inventory_sha256=sha256(inventory_path),
            execution_contract=contract, cases=members, input_sha256=fingerprints.files)
        write_json(journal.out / "plan.private.json", payload)
        fingerprints.verify()
        summary = {"status": "completed", "questions": len(members), "cohorts": dict(Counter(r["cohort"] for r in members)),
            "archival_stratum_questions": sum(r["archival_stratum"] for r in members), "models_executed": False}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def structured_correct(question, label, result):
    """A deterministic structured endpoint, explicitly not free-text entailment."""
    status, answer = result.get("status"), result.get("answer") or {}
    if label["assessment"] == "insufficient_evidence":
        return status == "abstained" and answer.get("decision") in {"insufficient_data", "incomparable"}
    if status != "completed":
        return False
    decision = answer.get("decision")
    if label["assessment"] in {"confirmed_change", "confirmed_stability"}:
        return decision == {"confirmed_change": "change", "confirmed_stability": "no_change"}[label["assessment"]]
    if label["assessment"] == "earliest_presence":
        allowed = {label["earliest_observation_id"], *label.get("equivalent_observation_ids", [])}
        ids = answer.get("observation_ids", [])
        interval = answer.get("date_interval") or {}
        return (decision == "earliest_presence" and len(ids) == 1 and ids[0] in allowed
                and interval.get("start") == interval.get("end") == label["present_by"]
                and interval.get("precision") == ("year" if len(label["present_by"]) == 4 else "day"))
    actual = {k: answer.get(k) for k in ("decision", "observation_ids", "date_interval")}
    return actual in label.get("acceptable_structured_answers", [])


def aggregate(rows):
    counts = Counter(r["status"] for r in rows)
    buildings = {}
    for building in sorted({r["building_id"] for r in rows}):
        members = [r for r in rows if r["building_id"] == building]
        buildings[building] = {"questions": len(members), "structured_accuracy": ratio(sum(r["correct"] for r in members), len(members)),
            "response_coverage": ratio(sum(r["status"] in {"completed", "abstained"} for r in members), len(members))}
    supported = [r for r in rows if r["known_archive_support_count"]]
    return {"questions": len(rows), "structured_correct": sum(r["correct"] for r in rows),
        "structured_accuracy": ratio(sum(r["correct"] for r in rows), len(rows)),
        "building_macro_accuracy": ratio(sum(r["structured_accuracy"] for r in buildings.values()), len(buildings)),
        "independent_buildings": len(buildings), "buildings": buildings, "outcome_counts": dict(counts),
        "response_coverage": ratio(counts["completed"] + counts["abstained"], len(rows)),
        "decision_coverage": ratio(counts["completed"], len(rows)),
        "legitimate_abstentions": sum(r["legitimate_abstention"] for r in rows),
        "known_support_questions": len(supported), "known_support_hit_rate": ratio(sum(r["known_support_hits"] > 0 for r in supported), len(supported)),
        "known_support_recall": ratio(sum(r["known_support_hits"] for r in supported), sum(r["known_archive_support_count"] for r in supported)),
        "citation_id_validity": ratio(sum(r["valid_citations"] for r in rows), sum(r["citations"] for r in rows)),
        "cost_totals": {k: sum(r["costs"].get(k, 0) for r in rows) for k in (
            "model_calls", "qwen_requests", "retrieval_rounds", "input_tokens", "output_tokens", "images", "repeated_images",
            "cache_hits", "model_loads", "worker_starts", "retries", "elapsed_seconds")},
        "incomplete_cost_cases": sum(bool(r["costs"].get("incomplete")) for r in rows)}


def annotation_diagnostics(answers_path, labels, fingerprints):
    """Audit annotation-derived drafting signals against reviewed cases, outside the primary endpoint."""
    private = qa.read_bound(answers_path, fingerprints)
    reviewed = qa.read_bound(private["review_source"]["path"], fingerprints, private["review_source"]["sha256"])
    candidates = qa._load_candidates(reviewed["candidates_path"], fingerprints)
    damage_classes = set(candidates["source"].get("damage_categories", []))
    rows = []
    for case in candidates["candidates"]:
        if case["question_id"] not in labels:
            continue
        label = labels[case["question_id"]]
        facts = curation.proposals(case, fingerprints, candidates["source"].get("annotation_targets"))["positive_annotation_facts"]
        positive = {r["observation_id"] for r in facts if r["category"] in damage_classes}
        first = next((r["observation_id"] for r in case["observations"] if r["observation_id"] in positive), None)
        rows.append({"question_id": case["question_id"], "candidate_family": case["candidate_type"],
            "reviewed_assessment": label["assessment"], "positive_annotation_facts": facts,
            "first_annotated_damage_observation_id": first,
            "earliest_annotation_matches_reviewed_visible_presence": first in {label.get("earliest_observation_id"),
                *label.get("equivalent_observation_ids", [])} if first is not None and label["assessment"] == "earliest_presence" else None})
    assessable = [r for r in rows if r["reviewed_assessment"] in {"confirmed_change", "confirmed_stability"}]
    flagged = [r for r in assessable if r["candidate_family"] == "possible_change"]
    changes = [r for r in assessable if r["reviewed_assessment"] == "confirmed_change"]
    hits = sum(r["reviewed_assessment"] == "confirmed_change" for r in flagged)
    earliest = [r for r in rows if r["reviewed_assessment"] == "earliest_presence"]
    return qa._private("www_annotation_drafting_diagnostics", cases=rows,
        scope="Separate annotation-derived drafting audit, not an answer-model condition or temporal gold source",
        unknown_policy="Unannotated pixels remain unknown; equal masks do not prove stability",
        reviewed_assessable_cases=len(assessable), change_proposal_precision=ratio(hits, len(flagged)),
        change_proposal_recall=ratio(hits, len(changes)), earliest_reviewed_cases=len(earliest),
        earliest_annotated_matches_reviewed_rate=ratio(sum(r["earliest_annotation_matches_reviewed_visible_presence"] is True for r in earliest), len(earliest)))


def score(questions_path, answers_path, run_path, plan_path, out):
    fingerprints = Fingerprints()
    public, labels = evaluation.approved_cohort(questions_path, answers_path, fingerprints)
    plan_doc = qa.read_bound(plan_path, fingerprints)
    _verify_recorded_artifacts(Path(plan_path).resolve().parent, fingerprints.files)
    root = Path(run_path).resolve()
    _verify_recorded_artifacts(root, fingerprints.files)
    record = qa.read_bound(root / "run.json", fingerprints)
    contract = qa.read_bound(root / "contract.json", fingerprints)
    if record.get("status") not in {"completed", "completed_with_errors", "failed", "interrupted"}:
        raise ValueError("Score only a terminal immutable inference run")
    if (plan_doc.get("kind") != "www_cohort_plan" or plan_doc["questions_sha256"] != sha256(questions_path)
            or plan_doc["answers_sha256"] != sha256(answers_path) or contract != plan_doc["execution_contract"]):
        raise ValueError("Evaluation questions/labels/model/corpus/protocol differ from the predeclared plan")
    if temporal_value(plan_doc["created_at"]) > temporal_value(record["started_utc"]):
        raise ValueError("Cohort/archival stratum was not frozen before model outcomes")
    members = {r["question_id"]: r for r in plan_doc["cases"]}
    questions = {r["question_id"]: r for r in public["questions"]}
    if len(members) != len(plan_doc["cases"]) or set(members) != set(questions):
        raise ValueError("Plan must retain the exact fixed cohort")
    if (root / "results.json").is_file():
        payload = qa.read_bound(root / "results.json", fingerprints)
        if payload.get("questions_sha256") != sha256(questions_path) or payload.get("contract_sha256") != digest(contract):
            raise ValueError("Saved results do not bind this execution contract")
        saved = payload["results"]
    else:
        saved = [qa.read_bound(p, fingerprints) for p in sorted((root / "questions").glob("*/*/result.json"))]
    reject_forbidden(saved, private_only=True)
    results = {}
    for result in saved:
        key = (result["question_id"], result["condition"])
        if key in results or key[0] not in questions or key[1] not in runner.CONDITIONS:
            raise ValueError("Duplicate/unknown saved question-condition; no result-based cohort selection")
        results[key] = result
    journal = new_run(out, {"questions_sha256": sha256(questions_path), "answers_sha256": sha256(answers_path),
        "plan_sha256": sha256(plan_path), "inference_run": str(root), "primary_endpoint": "reviewed_task_specific_structured_answer"}, "www_offline_metrics")
    try:
        rows, normalized, blind, blind_key, panels, sources = [], {}, [], [], [], {}
        (journal.out / "assets").mkdir()
        blind_salt = secrets.token_hex(16)
        gold_panels = []
        gold_root = journal.out / "reviewed-overlays.private"
        gold_root.mkdir()
        for identifier, question in questions.items():
            validation = journal.out / "validation" / identifier
            validation.mkdir(parents=True)
            prepared, local, common, support = runner._prepare(question, validation, fingerprints, contract["protocol"])
            label = labels[identifier]
            gold_dir = gold_root / identifier
            gold_dir.mkdir()
            images = []
            for number, event in enumerate(label.get("events", [])):
                path = runner._overlay(gold_dir, event, local[prepared["selected_observations"][-1]], number, reviewed=True)
                images.append({"src": path.relative_to(gold_root).as_posix(), "label": "PRIVATE REVIEWED / " + event["event_id"]})
            if images:
                gold_panels.append({"title": identifier, "caption": "Reviewed events on saved region_crop_pixels; geometric support is not localization accuracy.", "panels": images})
            for condition in runner.CONDITIONS:
                result = results.get((identifier, condition), {"status": "missing", "answer": None, "costs": {"incomplete": True}}).copy()
                exposed, tool_ids = {}, set()
                folder = root / "questions" / identifier / condition
                try:
                    if (folder / "exposure.json").is_file():
                        exposure = qa.read_bound(folder / "exposure.json", fingerprints)
                        reject_forbidden(exposure)
                        exposed = exposure["evidence"]
                        tool_ids = {k for k, v in exposed.items() if v.get("origin") == "archive"}
                    if result["status"] in {"completed", "abstained"}:
                        raw = qa.read_bound(folder / "final_raw.json", fingerprints)
                        answer = runner.validate_answer(raw, question, exposed, prepared, common, support, contract["protocol"])
                        if answer != result["answer"] or answer["status"] != result["status"]:
                            raise ValueError("Saved answer differs from its validated raw final JSON")
                except PrivateEvidenceError:
                    raise
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    result.update(status="error", answer=None, error="Saved answer validation: " + str(exc))
                if result["status"] not in {"completed", "abstained", "error", "failed", "missing"}:
                    result.update(status="error", answer=None, error="Unknown saved execution status")
                answer = result.get("answer") or {}
                normalized[identifier, condition] = result
                gold_support = set(label.get("supporting_evidence_ids", [])) - set(local)
                citations = answer.get("citations", [])
                row = {"question_id": identifier, "building_id": question["building_id"], "family": question["task"],
                    "assessment": label["assessment"], "condition": condition, **{k: members[identifier][k] for k in ("cohort", "archival_stratum")},
                    "status": result["status"], "correct": structured_correct(question, label, result), "error": result.get("error"),
                    "legitimate_abstention": label["assessment"] == "insufficient_evidence" and result["status"] == "abstained",
                    "known_archive_support_count": len(gold_support), "known_support_hits": len(gold_support & tool_ids),
                    "known_cited_support_hits": len(gold_support & set(citations)), "known_support_exhaustive": label.get("known_support_exhaustive", False),
                    "citations": len(citations), "valid_citations": len(set(citations) & set(exposed)), "costs": result.get("costs", {})}
                rows.append(row)
                blind_id = digest([blind_salt, identifier, condition])[:16]
                blind.append({"blind_id": blind_id, "question": question["question"], "answer_text": answer.get("answer_text", ""),
                    "citation_ids": citations, "factual_support": "", "contradiction": "", "image_source_relevance": "", "reviewer": "", "notes": ""})
                blind_key.append({"blind_id": blind_id, "question_id": identifier, "condition": condition})
                section_panels = []
                for key in citations:
                    evidence = exposed[key]
                    if evidence.get("image_path"):
                        fingerprints.add(evidence["image_path"], evidence["image_sha256"])
                        section_panels.append({"src": _copy(journal.out, evidence["image_path"], journal.out, sources), "label": key})
                source_text = [{"evidence_id": k, "source_page_url": exposed[k].get("source_page_url"),
                    "shown_source_excerpt": runner.archive.source_excerpt(exposed[k], exposed[k]["text_selection"]["used_characters"])[0]
                        if exposed[k].get("origin") == "archive" else None,
                    "observation_date": exposed[k].get("observation_date", exposed[k].get("date"))} for k in citations]
                panels.append({"title": blind_id, "caption": json.dumps({"question": question["question"],
                    "answer": answer.get("answer_text"), "sources": source_text}, ensure_ascii=False), "panels": section_panels})
        summaries, paired = {}, {}
        for cohort in ("development", "evaluation"):
            for stratum in ("all", "archival"):
                subset = [r for r in rows if r["cohort"] == cohort and (stratum == "all" or r["archival_stratum"])]
                key = cohort + "/" + stratum
                summaries[key] = {}
                for condition in runner.CONDITIONS:
                    selected = [r for r in subset if r["condition"] == condition]
                    summaries[key][condition] = {**aggregate(selected), "families": {
                        family: aggregate([r for r in selected if r["family"] == family]) for family in sorted({r["family"] for r in selected})}}
                paired[key] = []
                for a, b in zip(runner.CONDITIONS, runner.CONDITIONS[1:]):
                    left, right = summaries[key][a]["buildings"], summaries[key][b]["buildings"]
                    differences = {name: right[name]["structured_accuracy"] - left[name]["structured_accuracy"] for name in left}
                    paired[key].append({"contrast": b + " minus " + a, "buildings": differences,
                        "independent_buildings": len(differences), "mean_paired_difference": ratio(sum(differences.values()), len(differences)),
                        "uncertainty": "not estimated; no significance claim"})
        localization, localization_strata = {}, {}
        for condition in runner.CONDITIONS:
            wrapped = []
            for identifier in questions:
                result = normalized[identifier, condition]
                answer = result.get("answer") or {}
                wrapped.append({"question_id": identifier, "status": result["status"], "decision": answer.get("decision"),
                    "error": result.get("error"), "earliest_observation_id": (answer.get("observation_ids") or [None])[0],
                    "regions": [{**box, "prediction_id": identifier + f":box-{i+1}", "rank": i+1}
                        for i, box in enumerate(answer.get("boxes", []))] if answer.get("decision") == "change" else []})
            prediction = journal.out / ("predictions-" + condition + ".json")
            wrapper = {"schema_version": 1, "kind": "facade_qa_predictions", "model_manifest_sha256": sha256(questions_path),
                "ranking_source": {"kind": "saved_rank", "source": "saved model box order before private labels"}, "cases": wrapped}
            write_json(prediction, wrapper)
            evaluation.evaluate(questions_path, answers_path, prediction, journal.out / ("localization-" + condition), thresholds=(.25, .5))
            evaluated = read_json(journal.out / ("localization-" + condition) / "evaluation.json")
            # Preserve physical event matching and fixed-domain pixel diagnostics from qa evaluate.
            localization[condition] = [{**r, "physical_event_f1": ratio(2*r["matched"], r["eligible_events"] + r["available_predictions"])}
                                       for r in evaluated["pooled"]["localization"]]
            checked = evaluation._predictions(wrapper, questions, journal.out, fingerprints)
            localization_strata[condition] = {}
            for cohort in ("development", "evaluation"):
                for stratum in ("all", "archival"):
                    ids = [key for key in questions if members[key]["cohort"] == cohort and (stratum == "all" or members[key]["archival_stratum"])]
                    values = evaluation._aggregate({k: questions[k] for k in ids}, {k: labels[k] for k in ids}, {k: checked[k] for k in ids}, (.25, .5))
                    localization_strata[condition][cohort + "/" + stratum] = [{**r, "physical_event_f1": ratio(2*r["matched"], r["eligible_events"] + r["available_predictions"])} for r in values["localization"]]
        annotation_report = annotation_diagnostics(answers_path, labels, fingerprints)
        annotation_report["synthetic_fixture"] = contract["synthetic_fixture"]
        report = qa._private("www_metrics", primary_endpoint="task-specific structured correctness; free-text semantics require human review",
            synthetic_fixture=contract["synthetic_fixture"],
            denominator="all predeclared question-condition rows including errors and missing", summaries=summaries,
            paired_building_differences=paired, localization=localization, localization_by_cohort_and_stratum=localization_strata, cases=rows,
            citation_scope="ID exposure validity and known-support hit/recall; not entailment; other sources are not automatically irrelevant",
            archive_scope=(read_json(root / "summary.json").get("comparison_scope") if (root / "summary.json").is_file()
                           else "plumbing_no_archive" if contract["archive_index_sha256"] is None else "frozen_archive_incomplete_run"),
            input_sha256=fingerprints.files)
        write_json(journal.out / "metrics.private.json", report)
        write_json(journal.out / "annotation-diagnostics.private.json", annotation_report)
        write_static_gallery(gold_root, "PRIVATE — reviewed localization, never an inference input", gold_panels)
        curation.csv_rows(journal.out / "structured-cases.private.csv", rows)
        curation.csv_rows(journal.out / "blind-review.private.csv", sorted(blind, key=lambda r: r["blind_id"]))
        write_json(journal.out / "blind-key.private.json", qa._private("www_blind_review_key", cases=blind_key))
        write_static_gallery(journal.out, "Blind factual-support review", sorted(panels, key=lambda r: r["title"]), notes=[
            "Judge answer/source meaning in blind-review.private.csv. Condition mapping is separately private.",
            "A valid citation ID does not prove factual support. No gold overlays appear on this page."])
        fingerprints.verify()
        summary = {"status": "completed", "questions": len(questions), "conditions": list(runner.CONDITIONS),
            "cohort_condition_rows": len(rows), "report_path": str(journal.out / "gallery.html"),
            "synthetic_fixture": contract["synthetic_fixture"],
            "metrics": str(journal.out / "metrics.private.json"), "statistical_unit": "building"}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
