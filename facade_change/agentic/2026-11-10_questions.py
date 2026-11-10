"""Offline, human-reviewed facade QA cohorts. Private labels never enter RegionStore."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..history_features import Fingerprints, _history_document, _inside, digest
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .config import ROOT
from .coordinates import REGION_CROP, normalize_coordinate_system
from .evaluation import _binary, _verify_recorded_artifacts, box_iou
from .report import _copy
from .runner import complete_run, new_run
from .store import reject_forbidden, temporal_value


VERSION = 1
PRIVATE = "private_evaluation"
PROTOCOL = "supplied_region_qa"
ARCHIVAL = {"potentially_useful", "verified_useful", "unavailable", "unknown"}
ASSESSMENTS = {"confirmed_change", "confirmed_stability", "insufficient_evidence", "earliest_presence"}
PUBLIC_KEYS = {"question_id", "question", "task", "building_id", "view_id", "region_id", "split",
               "protocol", "task_mode", "knowledge_cutoff", "task_dates", "frame", "observations"}
OBS_KEYS = {"observation_id", "date", "available_at", "date_status", "rgb_path", "rgb_sha256",
            "support_path", "support_sha256", "geometry"}


def read_bound(path, fingerprints, expected=None):
    fingerprints.add(path, expected)
    return read_json(path)


def _private(kind, **values):
    return {"schema_version": VERSION, "access": PRIVATE, "kind": kind, **values}


def _load_candidates(path, fingerprints):
    _verify_recorded_artifacts(Path(path).resolve().parent, fingerprints.files)
    document = read_bound(path, fingerprints)
    if document.get("kind") != "facade_qa_candidates" or document.get("access") != PRIVATE:
        raise ValueError("Expected an immutable private QA candidate manifest")
    return document


def _annotation_maps(path, manifest_sha, fingerprints):
    if path is None:
        return None, {}, []
    path = Path(path).resolve()
    document = read_bound(path, fingerprints)
    if document.get("kind") != "history_annotation_targets":
        raise ValueError("Annotations must be a saved history_annotation_targets/index.json")
    _verify_recorded_artifacts(path.parent, fingerprints.files)
    if manifest_sha not in document.get("input_sha256", {}).values():
        raise ValueError("Annotation export is not bound to this exact history manifest")
    categories = document["contract"]["categories"]
    expected = [{"category_id": row["id"], "channel": i} for i, row in enumerate(categories)]
    if document["contract"].get("category_to_channel") != expected or len(categories) > 16:
        raise ValueError("Annotation ontology/channel mapping is inconsistent")
    return path.parent, document["observations"], categories


def _review_view(out, candidates, decisions=None):
    """The review gallery/table is explicitly private and has no model-facing links."""
    decisions = decisions or {}
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    sources, sections, table = {}, [], []
    for case in candidates:
        review = decisions.get(case["question_id"], {})
        label = review.get("label") or {}
        panels = []
        for observation in case["observations"]:
            relative = _copy(out, observation["rgb_path"], out, sources)
            panels.append({"src": relative, "label": f"RGB {observation['date']} / {observation['observation_id']}"})
        for preview in case.get("annotation_previews", []):
            relative = _copy(out, preview["path"], out, sources)
            panels.append({"src": relative, "label": preview["caption"]})
        status = review.get("status", "candidate")
        sections.append({"title": case["question_id"] + " — " + case["question"], "panels": panels,
            "caption": json.dumps({"access": PRIVATE, "status": status, "candidate_type": case["candidate_type"],
                "provisional_answer": case["provisional_answer"], "issues": case["unresolved_issues"],
                "observability": case["observability"], "archival_status": case["archival_status"],
                "review": review}, ensure_ascii=False)})
        table.append({"question_id": case["question_id"], "status": status, "revision": review.get("revision", 0),
            "building_id": case["building_id"], "view_id": case["view_id"], "region_id": case["region_id"],
            "candidate_type": case["candidate_type"], "question": case["question"],
            "provisional_answer": case["provisional_answer"], "task_dates": " / ".join(case["task_dates"]),
            "observability": label.get("observability", case["observability"]),
            "archival_status": label.get("archival_status", case["archival_status"]),
            "reviewer": review.get("reviewer", ""), "private_notes": review.get("private_notes", "")})
    columns = list(table[0]) if table else ["question_id", "status"]
    with (out / "review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    write_static_gallery(out, "PRIVATE — кандидаты вопросов фасадной истории", sections, notes=[
        "Только ручное approved допускает случай к оценке. Аннотации не являются проверенным временным GT.",
        "Совпадение классов не доказывает стабильность. Отсутствие аннотаций не доказывает отсутствие повреждения."])


def generate(histories, out, *, annotation_targets=None, knowledge_cutoff, task_mode, available_at,
             damage_categories=None, splits=("train", "val", "test", "unassigned")):
    root = Path(histories).resolve()
    if task_mode not in {"retrospective", "as_of"} or temporal_value(knowledge_cutoff) is None:
        raise ValueError("Choose retrospective/as_of and an explicit knowledge cutoff")
    if available_at != "observation_date":
        if temporal_value(available_at) is None:
            raise ValueError("Explicit archive availability or observation_date assumption is required")
    fingerprints = Fingerprints()
    document, manifest_sha = _history_document(root, fingerprints)
    annotation_root, annotations, categories = _annotation_maps(annotation_targets, manifest_sha, fingerprints)
    taxonomy = read_bound(ROOT / "configs/semantic_class_groups.json", fingerprints)
    damage_categories = damage_categories or next(g["categories"] for g in taxonomy["groups"] if g["name"] == "structural_damage")
    channels = [i for i, row in enumerate(categories) if row["name"] in damage_categories]
    if annotation_targets and not channels:
        raise ValueError("No requested damage categories in the saved annotation ontology")
    damage_bits = sum(1 << i for i in channels)
    config = {"histories": str(root), "annotation_targets": str(annotation_targets) if annotation_targets else None,
              "knowledge_cutoff": knowledge_cutoff, "task_mode": task_mode, "available_at": available_at,
              "damage_categories": damage_categories, "protocol": PROTOCOL, "splits": list(splits)}
    journal = new_run(out, config, "facade_qa_candidates")
    candidates = []
    skipped = []
    try:
        (journal.out / "previews").mkdir()
        for track in document["tracks"]:
            if track["split"] not in splits:
                continue
            x, y, width, height = track["bbox_xywh"]
            frame = {"coordinate_system": REGION_CROP, "region_id": track["track_id"], "shape": [height, width],
                     "anchor_sha256": track["anchor_sha256"], "anchor_bbox_xyxy": [x, y, x+width, y+height]}
            observations, arrays, previews, annotation_evidence = [], {}, {}, {}
            for row in track["observations"]:
                date = str(row["year"])
                known_at = date if available_at == "observation_date" else available_at
                if max(temporal_value(date), temporal_value(known_at)) > temporal_value(knowledge_cutoff):
                    skipped.append({"observation_id": row["observation_id"], "reason": "after_knowledge_cutoff"})
                    continue
                rgb, support_path = _inside(root, row["rgb"]), _inside(root, row["support"])
                fingerprints.add(rgb, row["rgb_sha256"])
                fingerprints.add(support_path, row["support_sha256"])
                support = _binary(root, str(support_path))
                if support.shape != (height, width) or not support.any():
                    skipped.append({"observation_id": row["observation_id"], "reason": "no_usable_geometric_support"})
                    continue
                with Image.open(rgb) as image:
                    if image.size != (width, height) or image.mode != "RGB":
                        raise ValueError("History RGB grid/type mismatch")
                matrix = np.asarray(row["alignment"]["matrix"], dtype=float)
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
                    raise ValueError("Invalid saved history geometry")
                shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
                observation = {"observation_id": row["observation_id"], "date": date, "date_status": "confirmed",
                    "available_at": known_at, "rgb_path": str(rgb), "rgb_sha256": row["rgb_sha256"],
                    "support_path": str(support_path), "support_sha256": row["support_sha256"],
                    "geometry": {"source_to_crop": (shift @ matrix).tolist(), "crop_to_source": np.linalg.inv(shift @ matrix).tolist(),
                                 "crop_shape": [height, width], "anchor_sha256": track["anchor_sha256"]}}
                observations.append(observation)
                packed = None
                mapped = annotations.get(row["observation_id"])
                if mapped:
                    if mapped["track_id"] != track["track_id"] or mapped["image_sha256"] != row["image_sha256"]:
                        raise ValueError("Annotation observation does not own this history/grid")
                    artifact = _inside(annotation_root, mapped["artifact_path"])
                    fingerprints.add(artifact, mapped["artifact_sha256"])
                    with np.load(artifact, allow_pickle=False) as loaded:
                        packed = loaded["packed"].copy()
                        if (packed.dtype != np.uint16 or packed.shape != support.shape
                                or not np.array_equal(loaded["support"], support)):
                            raise ValueError("Annotation masks do not share the saved observation grid/support")
                    if np.any(packed.astype(np.uint32) >> len(categories)):
                        raise ValueError("Annotation bitset contains channels outside its saved ontology")
                    if np.any(packed[~support]):
                        raise ValueError("Annotation pixels extend outside saved geometric support")
                    annotation_evidence[row["observation_id"]] = {"path": str(artifact), "sha256": mapped["artifact_sha256"],
                        "known_pixels": int((packed != 0).sum()), "coordinate_system": REGION_CROP, "region_id": track["track_id"],
                        "categories_present": [c["name"] for i, c in enumerate(categories) if np.any(packed & (1 << i))]}
                    preview = journal.out / "previews" / (digest(row["observation_id"])[:20] + ".png")
                    colors = np.zeros((*packed.shape, 3), np.uint8)
                    colors[packed != 0] = [70, 100, 160]
                    colors[(packed & damage_bits) != 0] = [255, 120, 40]
                    Image.fromarray(colors).save(preview)
                    previews[row["observation_id"]] = {"path": str(preview), "sha256": sha256(preview),
                        "caption": f"Аннотации {date}: оранжевый — выбранные повреждения; чёрный — неизвестно, не отсутствие"}
                arrays[row["observation_id"]] = (packed, support)
            observations.sort(key=lambda row: (temporal_value(row["date"]), row["observation_id"]))
            if not observations:
                continue

            def candidate(selected, family, reasons, provisional):
                task = "earliest_evidence" if family == "earliest_damage" else "temporal_change"
                identifier = "qa-" + digest([manifest_sha, track["track_id"], [r["observation_id"] for r in selected], family])[:24]
                support = arrays[selected[0]["observation_id"]][1]
                candidates.append({"question_id": identifier, "protocol": PROTOCOL, "task": task,
                    **{key: track[key] for key in ("building_id", "view_id", "split")}, "region_id": track["track_id"],
                    "task_mode": task_mode, "knowledge_cutoff": knowledge_cutoff, "task_dates": [r["date"] for r in selected],
                    "question": ("Какое самое раннее датированное свидетельство повреждения этого участка доступно?"
                                 if task == "earliest_evidence" else "Что изменилось в этом участке между наблюдениями, если вообще изменилось?"),
                    "candidate_type": family, "provisional_answer": provisional, "generation_reasons": reasons,
                    "observations": selected, "frame": frame, "observability": "unknown", "status": "candidate",
                    "reference_support": {"path": selected[0]["support_path"], "sha256": selected[0]["support_sha256"],
                        "pixels": int(support.sum()), "shape": [height, width], "basis": "fixed_reference_region_geometric_support"},
                    "annotation_previews": [previews[r["observation_id"]] for r in selected if r["observation_id"] in previews],
                    "annotation_evidence": {r["observation_id"]: annotation_evidence[r["observation_id"]] for r in selected
                                            if r["observation_id"] in annotation_evidence},
                    "unresolved_issues": ["Требуется проверка RGB и физической видимости; аннотации не являются временным GT"],
                    "archival_status": "potentially_useful" if task == "earliest_evidence" else "unknown",
                    "source_provenance": {"history_manifest_sha256": manifest_sha, "annotation_targets": config["annotation_targets"],
                        "availability_assumption": available_at}})

            first = observations[0]
            packed, support = arrays[first["observation_id"]]
            if packed is not None and np.any((packed & damage_bits) != 0):
                candidate(observations, "earliest_damage", ["damage_annotation_in_earliest_usable_observation"],
                          f"Возможное повреждение уже присутствует к {first['date']}; время возникновения неизвестно. Требуется проверка.")
            for before, after in zip(observations, observations[1:]):
                if before["date"] == after["date"]:
                    continue
                a, sa = arrays[before["observation_id"]]
                b, sb = arrays[after["observation_id"]]
                if a is None or b is None:
                    family, reason = "uncertain_control", "missing_annotation_mapping"
                else:
                    known = sa & sb & (a != 0) & (b != 0)
                    changed = known & (a != b)
                    if changed.any():
                        family, reason = "possible_change", "spatial_annotation_difference_on_same_region"
                    elif known.any() and np.all(known[sa | sb]):
                        family, reason = "stable_control", "matching_annotations_do_not_prove_stability"
                    else:
                        family, reason = "uncertain_control", "incomplete_annotation_or_geometric_coverage"
                candidate([before, after], family, [reason], "Не проверено: аннотации дают только сигнал для ручного сравнения RGB.")
        result = _private("facade_qa_candidates", protocol=PROTOCOL, source=config,
                          candidates=candidates, skipped=skipped, input_sha256=fingerprints.files)
        write_json(journal.out / "candidates.json", result)
        template = _private("facade_qa_review_decisions", candidate_sha256=sha256(journal.out / "candidates.json"),
            decisions=[{"question_id": row["question_id"], "expected_revision": 0, "status": "candidate",
                        "reviewer": "", "private_notes": "", "label": None} for row in candidates])
        write_json(journal.out / "review_template.json", template)
        _review_view(journal.out, candidates)
        fingerprints.verify()
        summary = {"status": "completed", "candidates": len(candidates), "approved": 0,
                   "review_table": str(journal.out / "review.csv"), "report_path": str(journal.out / "gallery.html")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def _checked_label(case, label, fingerprints):
    if not isinstance(label, dict) or label.get("assessment") not in ASSESSMENTS:
        raise ValueError("Approved cases need an explicit reviewed assessment")
    result = copy.deepcopy(label)
    assessment = result["assessment"]
    allowed = ({"earliest_presence", "insufficient_evidence"} if case["task"] == "earliest_evidence"
               else {"confirmed_change", "confirmed_stability", "insufficient_evidence"})
    if assessment not in allowed:
        raise ValueError("Earliest presence is not a temporal-change label")
    if result.get("observability") not in {"visible", "partly_visible", "occluded", "missing_coverage", "poor_quality", "other", "unknown"}:
        raise ValueError("Record observability explicitly; unknown is permitted")
    if not isinstance(result.get("reference_answer"), str) or not result["reference_answer"].strip():
        raise ValueError("Reviewed reference answer is required")
    if not isinstance(result.get("limitations", []), list):
        raise ValueError("Limitations must be a list")
    if assessment == "insufficient_evidence" and not result.get("limitations"):
        raise ValueError("Insufficient evidence needs a documented limitation")
    if result["observability"] in {"occluded", "missing_coverage"} and assessment != "insufficient_evidence":
        raise ValueError("Fully hidden/uncovered physical state belongs in underlying_state; it is not an assessable verdict")
    if result.get("archival_status", "unknown") not in ARCHIVAL:
        raise ValueError("Unknown archival usefulness state")
    if result.get("archival_status") == "verified_useful" and not result.get("supporting_evidence_ids"):
        raise ValueError("Verified archival usefulness needs private supporting evidence IDs")
    if assessment == "earliest_presence":
        observation = next((r for r in case["observations"] if r["observation_id"] == result.get("earliest_observation_id")), None)
        if observation is None or result.get("present_by") != observation["date"] or result.get("onset") != "unknown":
            raise ValueError("Earliest presence needs a permitted dated observation; onset must remain unknown")
    events = result.setdefault("events", [])
    if not isinstance(events, list) or (assessment == "confirmed_change" and not events):
        raise ValueError("Confirmed localized changes require reviewed event identities and boxes")
    if assessment != "confirmed_change" and events:
        raise ValueError("Hidden underlying events belong in underlying_state, not eligible observed events")
    ids = set()
    for event in events:
        identity = event.get("event_id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Reviewed physical event IDs must be nonempty and unique per question")
        ids.add(identity)
        frame = normalize_coordinate_system(event.get("coordinate_system"))
        if frame != REGION_CROP or event.get("region_id") != case["region_id"]:
            raise ValueError("Review boxes must use this supplied region_crop_pixels grid; do not rename native boxes")
        if (not isinstance(event.get("bbox_xyxy"), list)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in event["bbox_xyxy"])):
            raise ValueError("Reviewed bbox needs four finite numeric coordinates, not booleans")
        box_iou(event.get("bbox_xyxy"), event.get("bbox_xyxy"))
        left, top, right, bottom = event["bbox_xyxy"]
        height, width = case["frame"]["shape"]
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError("Reviewed event box exceeds its reference region")
        event["mask_area"] = None
        if event.get("mask_path"):
            if event.get("mask_task") != "temporal_change":
                raise ValueError("Only reviewed temporal-change masks support area stratification")
            path = Path(event["mask_path"]).expanduser().resolve()
            fingerprints.add(path, event.get("mask_sha256"))
            mask = _binary(path.parent, str(path))
            reference = case["reference_support"]
            fingerprints.add(reference["path"], reference["sha256"])
            support = _binary(Path(reference["path"]).parent, reference["path"])
            if mask.shape != tuple(case["frame"]["shape"]) or support.shape != mask.shape or np.any(mask & ~support):
                raise ValueError("Event mask must share the fixed reference grid/support")
            if not mask.any() or int(support.sum()) != reference["pixels"]:
                raise ValueError("Positive event mask/immutable reference denominator required")
            ys, xs = np.nonzero(mask)
            if xs.min() < left or ys.min() < top or xs.max()+1 > right or ys.max()+1 > bottom:
                raise ValueError("Reviewed event mask extends outside its event box")
            event.update(mask_path=str(path), mask_sha256=sha256(path), support_path=reference["path"],
                         support_sha256=reference["sha256"])
            event["mask_area"] = {"changed_pixels": int(mask.sum()), "reference_pixels": reference["pixels"],
                "fraction": int(mask.sum()) / reference["pixels"], "shape": list(mask.shape),
                "coordinate_system": REGION_CROP, "region_id": case["region_id"], "basis": reference["basis"],
                "reference_support_sha256": reference["sha256"], "mask_sha256": event["mask_sha256"]}
    return result


def review(candidates_path, decisions_path, out, previous=None):
    fingerprints = Fingerprints()
    candidates = _load_candidates(candidates_path, fingerprints)
    candidate_sha = sha256(candidates_path)
    supplied = read_bound(decisions_path, fingerprints)
    if supplied.get("candidate_sha256") != candidate_sha:
        raise ValueError("Review decisions must bind the exact original candidate manifest")
    states, history = {}, []
    if previous:
        _verify_recorded_artifacts(Path(previous).resolve().parent, fingerprints.files)
        old = read_bound(previous, fingerprints)
        if old.get("kind") != "facade_qa_review" or old["candidate_sha256"] != candidate_sha:
            raise ValueError("Review revision belongs to another cohort")
        states = old["states"]
        history = [*old["history"], {"path": str(Path(previous).resolve()), "sha256": sha256(previous)}]
        for source in history:
            fingerprints.add(source["path"], source["sha256"])
    cases = {r["question_id"]: r for r in candidates["candidates"]}
    journal = new_run(out, {"candidates": str(Path(candidates_path).resolve()), "candidate_sha256": candidate_sha,
                           "previous": str(Path(previous).resolve()) if previous else None,
                           "decisions": str(Path(decisions_path).resolve())}, "facade_qa_review")
    try:
        touched = set()
        for decision in supplied["decisions"]:
            identifier = decision["question_id"]
            if identifier not in cases or identifier in touched:
                raise ValueError("Unknown or duplicate question in review decisions")
            touched.add(identifier)
            if not decision.get("reviewer") and decision.get("status") == "candidate":
                continue  # Unedited template row, not a human decision.
            if not decision.get("reviewer") or not decision.get("private_notes"):
                raise ValueError("Every review decision needs reviewer and rationale")
            revision = states.get(identifier, {}).get("revision", 0)
            if decision.get("expected_revision") != revision:
                raise ValueError("Stale review revision")
            status = decision.get("status")
            if status not in {"candidate", "approved", "rejected"}:
                raise ValueError("Review status must be candidate/approved/rejected")
            label = _checked_label(cases[identifier], decision.get("label"), fingerprints) if status == "approved" else None
            states[identifier] = {"status": status, "revision": revision+1, "reviewer": decision["reviewer"],
                "private_notes": decision["private_notes"], "reviewed_at": datetime.now(timezone.utc).isoformat(), "label": label}
            journal.event("qa_review_decision", question_id=identifier, **states[identifier])
        result = _private("facade_qa_review", candidate_sha256=candidate_sha,
                          candidates_path=str(Path(candidates_path).resolve()), history=history, states=states,
                          input_sha256=fingerprints.files)
        write_json(journal.out / "reviewed.json", result)
        _review_view(journal.out, candidates["candidates"], states)
        fingerprints.verify()
        summary = {"status": "completed", **{status: sum(r.get("status", "candidate") == status for r in
            [states.get(case["question_id"], {}) for case in candidates["candidates"]]) for status in ("candidate", "approved", "rejected")}}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def validate_public(document):
    reject_forbidden(document)
    if set(document) != {"schema_version", "kind", "protocol", "questions", "retrieval_contract"}:
        raise ValueError("Unexpected model manifest fields")
    if document["kind"] != "facade_qa_questions" or document["protocol"] != PROTOCOL:
        raise ValueError("Expected supplied-region QA questions")
    ids = set()
    for case in document["questions"]:
        if set(case) != PUBLIC_KEYS or case["question_id"] in ids:
            raise ValueError("Unknown/private fields or duplicate public question")
        ids.add(case["question_id"])
        if set(case["frame"]) != {"coordinate_system", "region_id", "shape", "anchor_sha256", "anchor_bbox_xyxy"}:
            raise ValueError("Unexpected public coordinate metadata")
        for observation in case["observations"]:
            if set(observation) != OBS_KEYS:
                raise ValueError("Unknown/private model observation fields")
            if set(observation["geometry"]) != {"source_to_crop", "crop_to_source", "crop_shape", "anchor_sha256"}:
                raise ValueError("Unexpected public geometry fields")
            if max(temporal_value(observation["date"]), temporal_value(observation["available_at"])) > temporal_value(case["knowledge_cutoff"]):
                raise ValueError("Model observation is unavailable at knowledge cutoff")
    return document


def export(review_path, out):
    fingerprints = Fingerprints()
    _verify_recorded_artifacts(Path(review_path).resolve().parent, fingerprints.files)
    reviewed = read_bound(review_path, fingerprints)
    if reviewed.get("kind") != "facade_qa_review":
        raise ValueError("Expected reviewed cohort")
    for item in reviewed["history"]:
        fingerprints.add(item["path"], item["sha256"])
    fingerprints.add(reviewed["candidates_path"], reviewed["candidate_sha256"])
    candidates = _load_candidates(reviewed["candidates_path"], fingerprints)
    journal = new_run(out, {"review": str(Path(review_path).resolve()), "review_sha256": sha256(review_path)}, "facade_qa_export")
    try:
        questions, labels = [], []
        for case in candidates["candidates"]:
            decision = reviewed["states"].get(case["question_id"], {})
            if decision.get("status") != "approved":
                continue
            label = _checked_label(case, decision["label"], fingerprints)
            for observation in case["observations"]:
                for name in ("rgb", "support"):
                    fingerprints.add(observation[name + "_path"], observation[name + "_sha256"])
            questions.append({key: copy.deepcopy(case[key]) for key in PUBLIC_KEYS})
            labels.append({"question_id": case["question_id"], "building_id": case["building_id"],
                           "review": decision, "reviewed_labels": label})
        public = validate_public({"schema_version": VERSION, "kind": "facade_qa_questions", "protocol": PROTOCOL,
            "questions": questions, "retrieval_contract": {"encoder": "jina-clip-v2", "units": ["dated_observation", "archival_record"],
                "comparison_variants": ["photos_only", "multimodal_retrieval", "graph_retrieval", "bounded_agentic_retrieval"],
                "implemented_systems": [], "cutoff_field": "knowledge_cutoff"}})
        write_json(journal.out / "questions.json", public)
        private = _private("facade_qa_answers", model_manifest_sha256=sha256(journal.out / "questions.json"),
            review_source={"path": str(Path(review_path).resolve()), "sha256": sha256(review_path)}, cases=labels)
        write_json(journal.out / "answers.private.json", private)
        fingerprints.verify()
        summary = {"status": "completed", "approved_questions": len(questions), "protocol": PROTOCOL,
                   "model_manifest": str(journal.out / "questions.json"), "private_labels": str(journal.out / "answers.private.json")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--histories", required=True)
    gen.add_argument("--annotation-targets")
    gen.add_argument("--knowledge-cutoff", required=True)
    gen.add_argument("--task-mode", choices=("retrospective", "as_of"), required=True)
    gen.add_argument("--available-at", required=True)
    gen.add_argument("--damage-categories", nargs="+")
    gen.add_argument("--splits", nargs="+", default=["train", "val", "test", "unassigned"])
    rev = commands.add_parser("review")
    rev.add_argument("--candidates", required=True)
    rev.add_argument("--decisions", required=True)
    rev.add_argument("--previous")
    exp = commands.add_parser("export")
    exp.add_argument("--review", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--iou", nargs="+", type=float, default=[.25, .5])
    demo = commands.add_parser("demo")
    for command in (gen, rev, exp, evaluate, demo):
        command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.action == "generate":
                result = generate(args.histories, args.out, annotation_targets=args.annotation_targets,
                    knowledge_cutoff=args.knowledge_cutoff, task_mode=args.task_mode, available_at=args.available_at,
                    damage_categories=args.damage_categories, splits=args.splits)
            elif args.action == "review":
                result = review(args.candidates, args.decisions, args.out, args.previous)
            elif args.action == "export":
                result = export(args.review, args.out)
            elif args.action == "evaluate":
                module = importlib.import_module(".2026-11-10_qa_evaluation", __package__)
                result = module.evaluate(args.questions, args.answers, args.predictions, args.out, args.iou)
            else:
                module = importlib.import_module(".2026-11-10_qa_demo", __package__)
                result = module.demo(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"agentic qa: {exc}", file=sys.stderr, flush=True)
        return 2
