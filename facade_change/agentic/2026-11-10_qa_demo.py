"""Clearly synthetic, CPU-only curation/evaluation acceptance example; no model calls."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import read_json, sha256, write_json
from .runner import complete_run, new_run


def fixture(root):
    root = Path(root)
    histories, annotations = root / "histories", root / "annotation-targets"
    histories.mkdir(parents=True)
    annotations.mkdir()
    (annotations / "maps").mkdir()
    original = root / "original"
    original.mkdir()
    groups = {"earliest": "b0", "change": "b1", "stable": "b2", "occluded": "b3",
              "missing": "b4", "error": "b5", "unsupported": "b6"}
    boxes = {"earliest": [[3, 1, 6, 4]], "change": [[3, 1, 6, 4], [7, 7, 9, 9]],
             "missing": [[2, 2, 4, 4]], "error": [[1, 1, 2, 2]], "stable": [], "occluded": [], "unsupported": []}
    images, packed = {}, {}
    for name in groups:
        for year in (2009, 2015):
            rgb = np.full((20, 20, 3), 180, np.uint8)
            labels = np.ones((20, 20), np.uint16)
            if name == "earliest" or year == 2015:
                for left, top, right, bottom in boxes[name]:
                    rgb[top:bottom, left:right] = [90, 60, 40]
                    labels[top:bottom, left:right] |= 2
            if name in {"occluded", "unsupported"} and year == 2015:
                rgb[:10, :10] = [50, 80, 100]
                labels[:10, :10] = 0  # Unknown annotations; synthetic reviewer separately labels occlusion.
            path = original / f"{name}-{year}.png"
            Image.fromarray(rgb).save(path)
            images[name, year], packed[name, year] = path, labels
    tracks, maps = [], {}
    for region in (*groups, "overlap"):
        name = "change" if region == "overlap" else region
        x = 2 if region == "overlap" else 0
        observations = []
        for year in (2009, 2015):
            identifier = region + "-" + str(year)
            rgb_path, support_path = histories / (identifier + ".png"), histories / (identifier + "-support.png")
            with Image.open(images[name, year]) as image:
                Image.fromarray(np.asarray(image)[:10, x:x+10]).save(rgb_path)
            Image.fromarray(np.full((10, 10), 255, np.uint8)).save(support_path)
            observations.append({"observation_id": identifier, "year": year, "image_id": name + "-" + str(year),
                "original_image_path": str(images[name, year]), "image_sha256": sha256(images[name, year]),
                "rgb": rgb_path.name, "rgb_sha256": sha256(rgb_path), "support": support_path.name,
                "support_sha256": sha256(support_path), "alignment": {"matrix": np.eye(3).tolist()}})
            artifact = annotations / "maps" / (identifier + ".npz")
            np.savez_compressed(artifact, packed=packed[name, year][:10, x:x+10], support=np.ones((10, 10), bool))
            maps[identifier] = {"observation_id": identifier, "track_id": region,
                "image_sha256": sha256(images[name, year]), "artifact_path": str(artifact.relative_to(annotations)),
                "artifact_sha256": sha256(artifact), "status": "synthetic_fixture_projection"}
        tracks.append({"track_id": region, "building_id": groups[name], "view_id": groups[name]+"-view", "split": "test",
                       "bbox_xywh": [x, 0, 10, 10], "anchor_sha256": sha256(images[name, 2009]),
                       "distinct_years": 2, "observations": observations})
    document = {"schema_version": 1, "kind": "canonical_crop_histories", "synthetic_fixture": True,
                "report": "reports/prepare-1", "tracks": tracks, "views": []}
    report = histories / document["report"]
    report.mkdir(parents=True)
    write_json(histories / "manifest.json", document)
    write_json(report / "manifest.json", document)
    write_json(report / "run.json", {"status": "completed", "synthetic_fixture": True,
                                    "manifest_sha256": sha256(histories / "manifest.json")})
    categories = [{"id": 4, "name": "ORNAMENT_INTACT"}, {"id": 9, "name": "CRACK"}]
    write_json(annotations / "index.json", {"schema_version": 1, "kind": "history_annotation_targets",
        "synthetic_fixture": True, "observations": maps, "events": [],
        "input_sha256": {str(histories / "manifest.json"): sha256(histories / "manifest.json")},
        "contract": {"categories": categories, "category_to_channel": [{"category_id": 4, "channel": 0}, {"category_id": 9, "channel": 1}]}})
    write_json(annotations / "run.json", {"status": "completed", "synthetic_fixture": True,
                                         "artifact_sha256": {"index.json": sha256(annotations / "index.json")}})
    # This photo is never included among the evaluated system's permitted observations.
    archival = root / "private-underlying-state.png"
    Image.fromarray(np.full((10, 10, 3), [100, 70, 40], np.uint8)).save(archival)
    return histories, annotations / "index.json", archival


def demo(out):
    qa = importlib.import_module(".2026-11-10_questions", __package__)
    evaluator = importlib.import_module(".2026-11-10_qa_evaluation", __package__)
    journal = new_run(out, {"synthetic_fixture": True, "models_executed": False}, "facade_qa_synthetic_demo")
    try:
        histories, annotations, archival = fixture(journal.out / "fixture")
        qa.generate(histories, journal.out / "candidates", annotation_targets=annotations,
                    knowledge_cutoff="2026", task_mode="retrospective", available_at="observation_date")
        candidates_path = journal.out / "candidates/candidates.json"
        candidates = read_json(candidates_path)["candidates"]
        decisions, selected = [], {}
        masks = journal.out / "private-event-masks"
        masks.mkdir()
        for case in candidates:
            region = case["region_id"]
            if region == "earliest" and case["task"] == "temporal_change":
                decisions.append({"question_id": case["question_id"], "expected_revision": 0, "status": "rejected",
                    "reviewer": "synthetic_fixture_not_human", "private_notes": "Synthetic redundant control excluded"})
                continue
            selected[region] = case
            label = {"assessment": "confirmed_change", "observability": "visible", "reference_answer": "Synthetic reviewed answer only",
                     "limitations": [], "archival_status": "unknown", "supporting_evidence_ids": [], "events": []}
            if region == "earliest":
                label.update(assessment="earliest_presence", earliest_observation_id=case["observations"][0]["observation_id"],
                    present_by="2009", onset="unknown", reference_answer="Повреждение уже присутствовало к 2009; начало неизвестно.",
                    archival_status="potentially_useful")
            elif region == "stable":
                label.update(assessment="confirmed_stability")
            elif region in {"occluded", "unsupported"}:
                label.update(assessment="insufficient_evidence", observability="occluded", limitations=["Synthetic truck hides the region"],
                    underlying_state={"assessment": "confirmed_change", "evidence_ids": ["fixture-private-archive"],
                        "source": {"path": str(archival), "sha256": sha256(archival)}, "not_permitted_to_system": True})
            else:
                events = {"change": [("e1", [3, 1, 6, 4]), ("e2", [7, 7, 9, 9])],
                          "overlap": [("e1", [1, 1, 4, 4]), ("e2", [5, 7, 7, 9])],
                          "missing": [("e3", [2, 2, 4, 4])], "error": [("e4", [1, 1, 2, 2])]}[region]
                for name, box in events:
                    event = {"event_id": name, "bbox_xyxy": box, "coordinate_system": "region_crop_pixels", "region_id": region}
                    if region not in {"overlap", "missing"}:
                        mask = np.zeros((10, 10), bool)
                        mask[box[1]:box[3], box[0]:box[2]] = True
                        path = masks / (name + ".npy")
                        np.save(path, mask, allow_pickle=False)
                        event.update(mask_path=str(path), mask_sha256=sha256(path), mask_task="temporal_change")
                    label["events"].append(event)
            decisions.append({"question_id": case["question_id"], "expected_revision": 0, "status": "approved",
                "reviewer": "synthetic_fixture_not_human", "private_notes": "Deterministic acceptance fixture; not real human evaluation", "label": label})
        decision_path = journal.out / "synthetic-decisions.private.json"
        write_json(decision_path, {"access": qa.PRIVATE, "candidate_sha256": sha256(candidates_path), "decisions": decisions})
        qa.review(candidates_path, decision_path, journal.out / "review")
        qa.export(journal.out / "review/reviewed.json", journal.out / "export")
        rows = []
        for region, case in selected.items():
            if region in {"missing", "overlap"}:
                continue
            row = {"question_id": case["question_id"], "status": "completed", "decision": "change", "regions": []}
            if region == "change":
                for index, box in enumerate(([3, 1, 6, 4], [3, 1, 6, 4], [0, 0, 10, 10]), 1):
                    row["regions"].append({"prediction_id": f"change-{index}", "rank": index, "bbox_xyxy": box,
                                           "coordinate_system": "region_crop_pixels", "region_id": region})
            elif region == "stable":
                row["regions"] = [{"prediction_id": "false-alarm", "rank": 1, "bbox_xyxy": [1, 1, 2, 2],
                                   "coordinate_system": "region_crop_pixels", "region_id": region}]
            elif region == "occluded":
                row.update(status="abstained", decision="insufficient_data")
            elif region == "unsupported":
                row.update(decision="no_change")
            elif region == "error":
                row.update(status="error", decision="insufficient_data", error="synthetic runtime error")
            else:
                row.update(decision="earliest_presence", earliest_observation_id=case["observations"][0]["observation_id"])
            rows.append(row)
        predictions_path = journal.out / "saved-predictions.json"
        write_json(predictions_path, {"schema_version": 1, "kind": "facade_qa_predictions", "synthetic_fixture": True,
            "model_manifest_sha256": sha256(journal.out / "export/questions.json"),
            "ranking_source": {"kind": "saved_rank", "source": "fixed synthetic saved order; not computed from GT"}, "cases": rows})
        evaluator.evaluate(journal.out / "export/questions.json", journal.out / "export/answers.private.json",
                           predictions_path, journal.out / "evaluation")
        result = read_json(journal.out / "evaluation/evaluation.json")
        at10 = next(r for r in result["pooled"]["localization"] if r["iou_threshold"] == .25 and r["k"] == 10)
        expected = {"eligible_events": 4, "matched": 1, "missed": 3, "available_predictions": 4, "unmatched_predictions": 3}
        if {key: at10[key] for key in expected} != expected:
            raise AssertionError("Hand-checkable fixture counts changed")
        summary = {"status": "completed", "synthetic_fixture": True, "models_executed": False, "expected_at_k10_iou025": expected,
                   "review_report": str(journal.out / "review/gallery.html"), "evaluation_report": str(journal.out / "evaluation/gallery.html")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
