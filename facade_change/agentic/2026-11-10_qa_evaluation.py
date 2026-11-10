"""Fixed-cohort, offline localization/observability metrics for reviewed facade QA."""
from __future__ import annotations

from collections import Counter, deque
import copy
import importlib
import json
import math
from pathlib import Path

from ..history_features import Fingerprints
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .coordinates import coordinate_mismatch
from .evaluation import _binary, _csv, _pixel_result, _verify_recorded_artifacts, box_iou
from .runner import complete_run, new_run
from .store import PrivateEvidenceError, reject_forbidden


K_VALUES = (1, 3, 5, 10)


def ratio(a, b):
    return a / b if b else None


def _file(path, fingerprints, expected=None):
    path = Path(path).expanduser().resolve()
    fingerprints.add(path, expected)
    return read_json(path)


def _ranking(ranking):
    if (not isinstance(ranking, dict) or not isinstance(ranking.get("kind"), str)
            or ranking["kind"] not in {"single_box", "saved_rank", "score_descending"}
            or not isinstance(ranking.get("source"), str) or not ranking["source"].strip()):
        raise ValueError("Document a saved, non-GT ranking source (single_box/saved_rank/score_descending)")
    return ranking


def _prediction_case(row, case, root, fingerprints, ranking):
    """Only this case's data is parsed here; cohort/ranking/GT checks stay outside."""
    identifier = row["question_id"]
    reject_forbidden(row)
    if "result_path" in row:
        path = Path(row["result_path"]).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        row["result_path"] = str(path)
        saved = _file(path, fingerprints, row.get("result_sha256"))
        reject_forbidden(saved, private_only=True)
        reject_forbidden(saved)
        if not isinstance(saved, dict):
            raise ValueError("Saved result must be a JSON object")
        if "regions" in saved and not isinstance(saved["regions"], list):
            raise ValueError("Saved regions must be a ranked list")
        row.update({k: saved.get(k) for k in ("status", "decision", "error", "earliest_observation_id")})
        location = saved.get("localization")
        if location is not None and not isinstance(location, dict):
            raise ValueError("Saved localization must be an object or null")
        row["regions"] = ([{**location, "prediction_id": identifier + ":single", "rank": 1}]
                          if location is not None and saved.get("decision") == "change" else [])
    status, decision = row.get("status"), row.get("decision")
    regions = row.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("Saved regions must be a ranked list")
    if ranking["kind"] == "single_box" and len(regions) > 1:
        raise ValueError("Do not manufacture a multi-region ranking from one VLM box")
    if status not in {"completed", "abstained", "error", "failed", "missing"}:
        raise ValueError("Unknown prediction status")
    if ((status == "abstained" and (decision not in {"insufficient_data", "incomparable"} or regions))
            or (status == "completed" and decision not in {"change", "no_change", "earliest_presence"})
            or (status == "completed" and regions and decision != "change")):
        raise ValueError("Inconsistent status/decision/regions")
    row.update(status=status, ranking_source=ranking, regions=[], supplied_prediction_count=len(regions))
    if status != "completed":
        return row
    ids, ranks = set(), set()
    for prediction in regions:
        if not isinstance(prediction, dict):
            raise ValueError("Each saved region must be an object")
        prediction = copy.deepcopy(prediction)
        name = prediction.get("prediction_id", identifier + ":single" if ranking["kind"] == "single_box" else None)
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Predictions require unique saved IDs")
        ids.add(name)
        if ranking["kind"] == "score_descending":
            score = prediction.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Saved prediction score must be finite")
        else:
            rank = 1 if ranking["kind"] == "single_box" else prediction.get("rank")
            if type(rank) is not int or rank < 1 or rank in ranks:
                raise ValueError("Saved model ranks must be distinct positive integers")
            ranks.add(rank)
            prediction["rank"] = rank
        prediction.update(prediction_id=name, question_id=identifier)
        prediction["coordinate_issue"] = coordinate_mismatch(prediction, case["frame"], region_id=case["region_id"])
        try:
            bbox = prediction.get("bbox_xyxy")
            if any(isinstance(v, bool) for v in bbox):
                raise ValueError("Boolean bbox coordinate")
            box_iou(bbox, bbox)
            if not prediction["coordinate_issue"]:
                height, width = case["frame"]["shape"]
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                    raise ValueError("Prediction box exceeds its coordinate grid")
        except (ValueError, TypeError) as exc:
            prediction["coordinate_issue"] = "invalid_box: " + str(exc)
        prediction.pop("pixel_diagnostic", None)
        for key in ("mask", "support"):
            if prediction.get(key + "_path"):
                try:
                    path = Path(prediction[key + "_path"]).expanduser()
                    path = (path if path.is_absolute() else root / path).resolve()
                    prediction[key + "_path"] = str(path)
                    fingerprints.add(path, prediction.get(key + "_sha256"))
                    if _binary(path.parent, str(path)).shape != tuple(case["frame"]["shape"]):
                        raise ValueError("Prediction pixel artifact does not share the question grid")
                except (OSError, ValueError, TypeError, EOFError) as exc:
                    prediction["pixel_diagnostic"] = {"status": "error", "reason": "prediction_pixel_artifact_error",
                        "artifact": key, "path": prediction[key + "_path"], "error": str(exc)}
        row["regions"].append(prediction)
    key = (lambda p: (-p["score"], p["prediction_id"])) if ranking["kind"] == "score_descending" else (lambda p: (p["rank"], p["prediction_id"]))
    row["regions"].sort(key=key)
    return row


def _predictions(document, questions, root, fingerprints):
    if document.get("kind") != "facade_qa_predictions" or not isinstance(document.get("cases"), list):
        raise ValueError("Expected a saved facade_qa_predictions manifest with a cases list")
    reject_forbidden(document, private_only=True)
    reject_forbidden({key: value for key, value in document.items() if key != "cases"})
    if "ranking_source" in document:
        _ranking(document["ranking_source"])
    cases = {}
    for supplied in document["cases"]:
        identifier = supplied.get("question_id") if isinstance(supplied, dict) else None
        if not isinstance(identifier, str) or identifier not in questions or identifier in cases:
            raise ValueError("Prediction cohort has unknown/duplicate question IDs")
        row = copy.deepcopy(supplied)
        ranking = _ranking(row.get("ranking_source", document.get("ranking_source")))
        if "result_path" in row and ranking["kind"] != "single_box":
            raise ValueError("Existing agentic result.json supplies at most one box")
        try:
            cases[identifier] = _prediction_case(row, questions[identifier], root, fingerprints, ranking)
        except PrivateEvidenceError:
            raise  # Never downgrade label leakage to a per-case technical failure.
        except (OSError, ValueError, TypeError) as exc:
            cases[identifier] = {"question_id": identifier, "status": "missing" if isinstance(exc, FileNotFoundError) else "error",
                "decision": None, "regions": [], "supplied_prediction_count": 0, "ranking_source": ranking,
                "error": f"{identifier} ({row.get('result_path', 'inline result')}): {type(exc).__name__}: {exc}"}
    for identifier in questions:
        cases.setdefault(identifier, {"question_id": identifier, "status": "missing", "decision": None,
                                      "regions": [], "supplied_prediction_count": 0, "ranking_source": None})
    return cases


def _events(questions, labels):
    events = {}
    for identifier, label in labels.items():
        if label["assessment"] != "confirmed_change":
            continue
        case = questions[identifier]
        for event in label["events"]:
            name = event["event_id"]
            identity = (case["building_id"], tuple(case["task_dates"]))
            entry = events.setdefault(name, {"event_id": name, "identity": identity, "representations": [], "mask_area": None})
            if entry["identity"] != identity:
                raise ValueError("Reviewed event ID spans buildings/temporal intervals; use distinct physical episode IDs")
            entry["representations"].append({**event, "question_id": identifier})
    for event in events.values():
        areas = [r["mask_area"] for r in event["representations"] if r.get("mask_area")]
        if not areas:
            event["area_status"] = "unavailable_box_only"
        elif all(area == areas[0] for area in areas):
            event.update(mask_area=areas[0], area_status="available")
        else:
            event["area_status"] = "unavailable_conflicting_crop_area_references"
    return events


def match_one_to_one(predictions, events, threshold):
    """Deterministic maximum-cardinality matching; no GT-derived prediction ordering."""
    edges, representations = {}, {}
    for index, prediction in enumerate(predictions):
        adjacent = []
        if not prediction.get("coordinate_issue"):
            for name, event in sorted(events.items()):
                options = []
                for truth in event["representations"]:
                    if truth["question_id"] != prediction["question_id"] or coordinate_mismatch(prediction, truth):
                        continue
                    overlap = box_iou(prediction["bbox_xyxy"], truth["bbox_xyxy"])
                    if overlap >= threshold:
                        options.append((overlap, truth))
                if options:
                    overlap, truth = max(options, key=lambda value: value[0])
                    adjacent.append((-overlap, name))
                    representations[index, name] = (overlap, truth)
        edges[index] = [name for _, name in sorted(adjacent)]
    assigned, reverse = {}, {}
    for start in range(len(predictions)):
        queue, visited, parent, finish = deque([start]), {start}, {}, None
        while queue and finish is None:
            index = queue.popleft()
            for name in edges[index]:
                if name in parent:
                    continue
                parent[name] = index
                if name not in assigned:
                    finish = name
                    break
                other = assigned[name]
                if other not in visited:
                    visited.add(other)
                    queue.append(other)
        while finish is not None:
            index = parent[finish]
            prior = reverse.get(index)
            assigned[finish], reverse[index] = index, finish
            finish = prior
    matches = []
    for name, index in sorted(assigned.items()):
        overlap, truth = representations[index, name]
        matches.append({"event_id": name, "question_id": predictions[index]["question_id"],
                        "prediction_id": predictions[index]["prediction_id"], "iou": overlap,
                        "pixels": predictions[index].get("pixel_diagnostic") or _pixel_result(
                            predictions[index], truth, Path("/"), Path("/"),
                            region_id=truth.get("region_id"), support_policy="fixed_target")})
    return matches


def _aggregate(questions, labels, predictions, thresholds):
    events = _events(questions, labels)
    cases = []
    for identifier, case in questions.items():
        label, prediction = labels[identifier], predictions[identifier]
        status = prediction["status"]
        assessment = label["assessment"]
        cases.append({"question_id": identifier, "building_id": case["building_id"], "assessment": assessment,
            "status": status, "decision": prediction.get("decision"), "available_predictions": len(prediction["regions"]),
            "error": prediction.get("error"),
            "supplied_predictions": prediction["supplied_prediction_count"], "ranking_source": prediction["ranking_source"],
            "coordinate_issues": sum(bool(p.get("coordinate_issue")) for p in prediction["regions"]),
            "coordinate_diagnostics": [{"prediction_id": p["prediction_id"], "reason": p["coordinate_issue"]}
                                       for p in prediction["regions"] if p.get("coordinate_issue")],
            "pixel_diagnostics": [{"prediction_id": p["prediction_id"], **p["pixel_diagnostic"]}
                                  for p in prediction["regions"] if p.get("pixel_diagnostic")],
            "false_alarm": assessment == "confirmed_stability" and status == "completed" and prediction.get("decision") == "change",
            "justified_abstention": assessment == "insufficient_evidence" and status == "abstained",
            "unsupported_confident_decision": assessment == "insufficient_evidence" and status == "completed",
            "earliest_evidence_correct": (prediction.get("earliest_observation_id") == label.get("earliest_observation_id")
                                          and status == "completed" and prediction.get("decision") == "earliest_presence")
                                         if assessment == "earliest_presence" else None})
    stable = [c for c in cases if c["assessment"] == "confirmed_stability"]
    uncertain = [c for c in cases if c["assessment"] == "insufficient_evidence"]
    earliest = [c for c in cases if c["assessment"] == "earliest_presence"]
    counts = Counter(c["status"] for c in cases)
    coverage = {"cohort_cases": len(cases), "completed": counts["completed"], "abstained": counts["abstained"],
        "missing": counts["missing"], "technical_errors": counts["error"] + counts["failed"],
        "response_coverage": ratio(counts["completed"] + counts["abstained"], len(cases)),
        "decision_coverage": ratio(counts["completed"], len(cases)),
        "coordinate_issue_predictions": sum(c["coordinate_issues"] for c in cases),
        "assessable_stable_cases": len(stable), "false_alarm_cases": sum(c["false_alarm"] for c in stable),
        "false_alarm_rate": ratio(sum(c["false_alarm"] for c in stable), len(stable)),
        "unassessable_cases": len(uncertain), "justified_abstentions": sum(c["justified_abstention"] for c in uncertain),
        "unsupported_confident_decisions": sum(c["unsupported_confident_decision"] for c in uncertain),
        "earliest_presence_cases": len(earliest), "earliest_evidence_correct": sum(c["earliest_evidence_correct"] for c in earliest),
        "earliest_evidence_recall": ratio(sum(c["earliest_evidence_correct"] for c in earliest), len(earliest))}
    metrics = []
    for threshold in thresholds:
        for k in K_VALUES:
            selected = [p for identifier in sorted(questions) if labels[identifier]["assessment"] in {"confirmed_change", "confirmed_stability"}
                        for p in predictions[identifier]["regions"][:k]]
            matches = match_one_to_one(selected, events, threshold)
            matched_ids = {r["event_id"] for r in matches}
            row = {"iou_threshold": threshold, "k": k, "eligible_events": len(events), "matched": len(matches),
                "missed": len(events)-len(matches), "available_predictions": len(selected),
                "unmatched_predictions": len(selected)-len(matches), "region_recall": ratio(len(matches), len(events)),
                "region_precision": ratio(len(matches), len(selected)), "matches": matches,
                "prediction_counts_by_question": {identifier: {"available": len(p["regions"]), "used": min(k, len(p["regions"]))}
                    for identifier, p in predictions.items() if labels[identifier]["assessment"] in {"confirmed_change", "confirmed_stability"}}, "area_subsets": []}
            bases = sorted({e["mask_area"]["basis"] for e in events.values() if e["mask_area"]})
            for basis in bases or ["fixed_reference_region_geometric_support"]:
                for minimum in (.01, .05):
                    eligible = {name for name, event in events.items() if event["mask_area"]
                                and event["mask_area"]["basis"] == basis and event["mask_area"]["fraction"] >= minimum}
                    hits = len(eligible & matched_ids)
                    row["area_subsets"].append({"basis": basis, "minimum_fraction": minimum, "eligible_events": len(eligible),
                        "matched": hits, "missed": len(eligible)-hits, "region_recall": ratio(hits, len(eligible)),
                        "region_precision": None, "precision_scope": "reported on all eligible events only; smaller true events are not false alarms"})
            metrics.append(row)
    return {"coverage": coverage, "localization": metrics, "cases": cases,
            "area_availability": dict(Counter(e["area_status"] for e in events.values())), "events": list(events.values())}


def evaluate(questions_path, answers_path, predictions_path, out, thresholds=(.25, .5)):
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("IoU thresholds must be distinct finite numbers in (0, 1]")
    qa = importlib.import_module(".2026-11-10_questions", __package__)
    fingerprints = Fingerprints()
    _verify_recorded_artifacts(Path(questions_path).resolve().parent, fingerprints.files)
    _verify_recorded_artifacts(Path(answers_path).resolve().parent, fingerprints.files)
    public = qa.validate_public(_file(questions_path, fingerprints))
    private = _file(answers_path, fingerprints)
    saved = _file(predictions_path, fingerprints)
    model_sha = sha256(questions_path)
    if private.get("kind") != "facade_qa_answers" or private.get("access") != qa.PRIVATE:
        raise ValueError("Only approved private QA answers can be scored")
    review_source = private["review_source"]
    _verify_recorded_artifacts(Path(review_source["path"]).resolve().parent, fingerprints.files)
    reviewed = _file(review_source["path"], fingerprints, review_source["sha256"])
    if private.get("model_manifest_sha256") != model_sha or saved.get("model_manifest_sha256") != model_sha:
        raise ValueError("Answers and predictions must bind the identical fixed model-question cohort")
    questions = {r["question_id"]: r for r in public["questions"]}
    if len(private["cases"]) != len(questions) or {r["question_id"] for r in private["cases"]} != set(questions):
        raise ValueError("Private answers must cover the exact approved cohort once")
    fingerprints.add(reviewed["candidates_path"], reviewed["candidate_sha256"])
    candidates = {r["question_id"]: r for r in qa._load_candidates(reviewed["candidates_path"], fingerprints)["candidates"]}
    labels = {}
    for row in private["cases"]:
        if row["review"].get("status") != "approved" or not row["review"].get("reviewer"):
            raise ValueError("Unapproved cases cannot enter scored evaluation")
        if row["review"] != reviewed["states"].get(row["question_id"]):
            raise ValueError("Private labels disagree with the immutable review source")
        original = candidates[row["question_id"]]
        expected = {**{key: original[key] for key in qa.PUBLIC_KEYS},
                    "question": row["review"].get("reviewed_question", original["question"])}
        if questions[row["question_id"]] != expected:
            raise ValueError("Public question disagrees with its reviewed wording/coordinate provenance")
        label = row["reviewed_labels"]
        if label != row["review"]["label"]:
            raise ValueError("Private labels disagree with the approved review revision")
        for event in label.get("events", []):
            for key in ("mask", "support"):
                if event.get(key + "_path"):
                    fingerprints.add(event[key + "_path"], event[key + "_sha256"])
        # Reuse curation validation: the domain is the immutable earlier reference,
        # with checked ownership, grid, support hash and fixed mask-area denominator.
        if qa._checked_label(original, label, fingerprints) != label:
            raise ValueError("Reviewed labels disagree with their fixed reference support/mask provenance")
        labels[row["question_id"]] = label
    predictions = _predictions(saved, questions, Path(predictions_path).resolve().parent, fingerprints)
    protocol = {"k": list(K_VALUES), "iou_thresholds": list(thresholds), "matching": "deterministic_maximum_cardinality_one_to_one",
        "rank_policy": "top K per question from saved non-GT ranking; no padding of single-box outputs",
        "event_deduplication": "reviewed physical event_id shared across overlapping crops, within building and temporal interval",
        "threshold_status": "0.25 is an initial coarse-localization choice; 0.5 stricter diagnostic, neither universally validated",
        "area_denominator": "fixed earlier reference geometric support; never reduced using later visibility",
        "pixel_support_policy": "fixed reviewed reference domain; prediction must cover every required pixel",
        "empty_positive_cohort": "recall undefined", "statistical_unit": "building", "protocol": qa.PROTOCOL}
    journal = new_run(out, {"model_manifest_sha256": model_sha, "inputs": fingerprints.files, **protocol}, "facade_qa_offline_evaluation")
    try:
        pooled = _aggregate(questions, labels, predictions, thresholds)
        buildings = {}
        for building in sorted({row["building_id"] for row in questions.values()}):
            cohort = {key: value for key, value in questions.items() if value["building_id"] == building}
            buildings[building] = _aggregate(cohort, {key: labels[key] for key in cohort}, {key: predictions[key] for key in cohort}, thresholds)
        macro = []
        for index, row in enumerate(pooled["localization"]):
            defined = [b["localization"][index]["region_recall"] for b in buildings.values() if b["localization"][index]["region_recall"] is not None]
            macro.append({"iou_threshold": row["iou_threshold"], "k": row["k"], "eligible_buildings": len(defined),
                          "mean_building_region_recall": sum(defined)/len(defined) if defined else None})
        result = {"schema_version": 1, "access": qa.PRIVATE, "status": "completed", "protocol": protocol,
                  "pooled": pooled, "buildings": buildings, "building_macro": macro, "input_sha256": fingerprints.files}
        write_json(journal.out / "evaluation.json", result)
        _csv(journal.out / "cases.csv", pooled["cases"])
        _csv(journal.out / "localization.csv", pooled["localization"])
        sections = [{"title": "Весь фиксированный approved cohort", "caption": json.dumps(pooled["coverage"], ensure_ascii=False), "panels": []}]
        sections += [{"title": f"IoU {row['iou_threshold']}, K={row['k']}", "caption": json.dumps(
            {key: row[key] for key in ("eligible_events", "matched", "missed", "available_predictions", "unmatched_predictions", "region_recall", "region_precision", "area_subsets")}, ensure_ascii=False), "panels": []}
            for row in pooled["localization"]]
        sections += [{"title": "Здание " + name, "caption": json.dumps(value["coverage"], ensure_ascii=False), "panels": []} for name, value in buildings.items()]
        sections += [{"title": row["question_id"], "caption": json.dumps(row, ensure_ascii=False), "panels": []}
                     for row in pooled["cases"] if row["error"] or row["coordinate_diagnostics"] or row["pixel_diagnostics"]]
        write_static_gallery(journal.out, "PRIVATE — офлайн-оценка фасадных вопросов", sections, notes=[
            "Оценка supplied-region QA не является автоматическим поиском изменений на всём фасаде.",
            "Размер изменения определяется только проверенной временной маской, не площадью bbox.",
            "Ошибки исполнения не являются успешным воздержанием. Отсутствующие прогнозы остаются в когорте."])
        fingerprints.verify()
        summary = {"status": "completed", "questions": len(questions), "buildings": len(buildings),
                   "coverage": pooled["coverage"], "report_path": str(journal.out / "gallery.html")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
