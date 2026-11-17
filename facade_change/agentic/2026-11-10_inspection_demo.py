"""Procedural CPU acceptance fixture. All detector/Qwen replies are explicit stubs."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from ..history_features import digest
from ..io import read_json, sha256, write_json
from .config import defaults
from .runner import new_run, complete_run


inspection = importlib.import_module(".2026-11-10_inspection", __package__)
data = importlib.import_module(".2026-11-10_inspection_data", __package__)
qa = importlib.import_module(".2026-11-10_questions", __package__)


def fixture(root):
    root = Path(root)
    histories = root / "histories"
    histories.mkdir(parents=True)
    original = root / "original"
    original.mkdir()
    tracks = []
    yy, xx = np.mgrid[:768, :1024]
    for b in range(1, 5):
        images = {}
        for index, year in enumerate((2000, 2005, 2010, 2015)):
            # Recognizable one-pixel detail, unique bytes for each native image.
            rgb = np.stack([(xx % 2)*210 + 20, (yy % 2)*200 + 15,
                            (xx*3 + yy*5 + b*17 + index*23) % 256], axis=-1).astype(np.uint8)
            path = original / f"b{b}-{year}.png"
            Image.fromarray(rgb).save(path)
            images[year] = path
        for x in ([80, 130] if b == 1 else [80]):
            identity = dict(building_id=f"b{b}", view_id=f"b{b}-front", anchor_sha256=sha256(images[2000]), bbox_xywh=[x, 60, 32, 32])
            track_id = digest(identity)[:24]
            observations = []
            for index, year in enumerate((2000, 2005, 2010, 2015)):
                matrix = np.eye(3) if year == 2000 else np.array([[.28, .012, 2.], [.004, .29, 3.], [.00004, .00003, 1.]])
                image_sha = sha256(images[year])
                oid = digest([track_id, image_sha])[:24]
                crop_path, support_path = histories / (oid + ".png"), histories / (oid + "-support.png")
                shift = np.array([[1., 0, -x], [0, 1., -60], [0, 0, 1.]])
                rgb = np.asarray(Image.open(images[year]))
                crop = cv2.warpPerspective(rgb, shift @ matrix, (32, 32))
                Image.fromarray(crop).save(crop_path)
                Image.fromarray(np.full((32, 32), 255, np.uint8)).save(support_path)
                observations.append(dict(observation_id=oid, year=year, image_id=f"b{b}-{year}",
                    original_image_path=str(images[year]), image_sha256=image_sha,
                    rgb=crop_path.name, rgb_sha256=sha256(crop_path), support=support_path.name,
                    support_sha256=sha256(support_path), alignment=dict(matrix=matrix.tolist())))
            tracks.append(dict(track_id=track_id, **identity, split="val", distinct_years=4, observations=observations))
    manifest = dict(schema_version=1, kind="canonical_crop_histories", synthetic_fixture=True,
                    report="reports/prepare-1", tracks=tracks, views=[])
    write_history(histories, manifest)
    return histories


def write_history(root, manifest):
    report = root / manifest["report"]
    report.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", manifest)
    write_json(report / "manifest.json", manifest)
    write_json(report / "run.json", dict(status="completed", synthetic_fixture=True, manifest_sha256=sha256(root / "manifest.json")))


def curate(root, histories, *, available_at="2026-11-10"):
    qa.generate(histories, root / "candidates", knowledge_cutoff="2026-11-10", task_mode="retrospective",
                available_at=available_at, splits=("val",))
    candidates_path = root / "candidates/candidates.json"
    decisions = []
    for case in read_json(candidates_path)["candidates"]:
        if case["task_dates"] != ["2010", "2015"]:
            continue
        stable = case["building_id"] == "b3" or case["frame"]["anchor_bbox_xyxy"][0] == 130
        uncertain = case["building_id"] == "b4"
        label = dict(assessment="insufficient_evidence" if uncertain else "confirmed_stability" if stable else "confirmed_change",
            observability="occluded" if uncertain else "visible", reference_answer="Explicit synthetic acceptance label",
            limitations=["synthetic persistent occlusion, including permitted history"] if uncertain else [],
            archival_status="unknown", supporting_evidence_ids=[], events=[])
        if not stable and not uncertain:
            box = [20, 20, 29, 29] if case["building_id"] == "b1" else [5, 6, 12, 14]
            label["events"] = [dict(event_id=case["building_id"] + "-synthetic-event", bbox_xyxy=box,
                coordinate_system="region_crop_pixels", region_id=case["region_id"])]
        decisions.append(dict(question_id=case["question_id"], expected_revision=0, status="approved",
            reviewer="synthetic_fixture_not_human", private_notes="Procedural CPU fixture; assessed against all permitted dates; not a real approval", label=label))
    decision_path = root / "synthetic-decisions.private.json"
    write_json(decision_path, dict(access=qa.PRIVATE, candidate_sha256=sha256(candidates_path), decisions=decisions))
    qa.review(candidates_path, decision_path, root / "review")
    qa.export(root / "review/reviewed.json", root / "export")
    return root / "export/questions.json", root / "export/answers.private.json"


class DetectorStub:
    calls = 0
    empty = False
    failed = False

    def __init__(self, **kwargs):
        pass

    def score(self, method, options, reference, source, support):
        type(self).calls += 1
        if self.failed:
            raise RuntimeError("Injected detector failure")
        mask = np.zeros(support.shape, bool)
        if not self.empty:
            mask[20:29, 20:29] = True
            mask[2:4, 20:22] = True
            mask[0, 0] = True
            mask[31, 31] = True  # Cap loss is retained explicitly, never a size filter.
        scores = mask.astype(np.float32)
        scores[~support] = np.nan
        return dict(scores=scores, native_prediction=mask,
                    metadata=dict(output_kind="native_mask", synthetic_fixture=True, method="anychange",
                                  architecture="stub of native binary AnyChange contract; no model loaded"))

    def close(self):
        pass

    release = close


class QwenStub:
    """A deterministic script shared by every policy; no access to private labels."""
    requests = []

    def __init__(self, models, *args, **kwargs):
        self.models = models

    def reply(self, payload):
        messages = payload["messages"]
        context = json.loads(messages[1]["content"][0]["text"])
        system = messages[0]["content"][0]["text"]
        if system.startswith("Select visual evidence"):
            if any(m["role"] == "assistant" for m in messages):
                return {"type": "stop"}
            selector = next(json.loads(b["text"]) for b in messages[1]["content"]
                            if b["type"] == "text" and '"catalogue"' in b["text"])
            return dict(type="tool", name="inspect_region", arguments=dict(
                observation_ids=[selector["catalogue"][-1]["observation_id"]], bbox_xyxy=[5, 6, 12, 14]))
        images = [b for m in messages for b in m["content"] if b["type"] == "image"]
        evidence = list(dict.fromkeys(b["evidence_id"] for b in images))
        packets = []
        for m in messages:
            for b in m["content"]:
                if b["type"] == "text":
                    try:
                        packet = json.loads(b["text"])
                    except ValueError:
                        continue
                    if isinstance(packet, dict) and packet.get("operation") == "inspect_region":
                        packets.append(packet)
        inspected = [r for p in packets for r in p["observation"]["evidence"]]
        focused = any(r["geometry"]["requested_bbox_xyxy"] == [5, 6, 12, 14] for r in inspected)
        heuristic = any(r["geometry"]["requested_bbox_xyxy"] == [20, 20, 29, 29] for r in inspected)
        building = context["building_id"]
        stable = context["geometry"]["bbox_xyxy"][0] == 130
        decision, box = "no_change", None
        if building == "b4":
            decision = "insufficient_data"
        elif building == "b1" and not stable:
            decision, box = "change", [20, 20, 29, 29]
        elif building == "b2" and focused:
            decision, box = "change", [5, 6, 12, 14]
        elif stable and heuristic:
            decision, box = "change", [20, 20, 29, 29]
        return dict(type="final", decision=decision, visible_difference="Synthetic scripted response; not real model evidence",
            semantic_interpretation=None, evidence_ids=evidence, limitations=["synthetic_fixture_not_scientific_result"],
            missing_information=[], localization=dict(evidence_id=evidence[0], coordinate_system="region_crop_pixels", bbox_xyxy=box) if box else None)

    def call(self, role, payload):
        type(self).requests.append(copy.deepcopy(payload))
        value = self.reply(payload)
        images = [b for m in payload["messages"] for b in m["content"] if b["type"] == "image"]
        return dict(status="ok", text=json.dumps(value), cache_hit=False,
            usage=dict(input_tokens=200+100*len(images), output_tokens=50, images=len(images), model_calls=1, model_loads=0),
            processing=dict(synthetic_processor=True, image_grid_thw=[[1, 2, 2] for _ in images]),
            generation=dict(actual_output_tokens=50, max_new_tokens=payload["max_new_tokens"], reached_max_new_tokens=False),
            provenance=dict(synthetic_fixture=True, model="explicit scripted Qwen stub; no learned model"))

    def close(self):
        pass


def config_fixture(root):
    config = defaults()
    config.update(cache_root=str(root / "cache"), methods=["anychange"], worker_timeout=1800)
    config["budgets"].update(data.CAPS)
    config["models"]["qwen"].update(max_input_tokens=16384, max_images=8, max_new_tokens=1024)
    config["method_options"] = {"anychange": dict(source_root=str(root / "synthetic-source-not-loaded"),
                                                 checkpoint_path=str(root / "synthetic-weights-not-loaded"))}
    write_json(root / "synthetic-config.json", config)
    return root / "synthetic-config.json"


def stub_signature(method, options):
    return dict(method=method, options=options, synthetic_fixture=True, detector="explicit procedural stub")


def demo(out):
    journal = new_run(out, dict(synthetic_fixture=True, learned_models_executed=False), "facade_inspection_demo")
    try:
        histories = fixture(journal.out / "fixture")
        questions, answers = curate(journal.out, histories)
        config_path = config_fixture(journal.out)
        protocol = data.protocol_document(sha256(histories / "manifest.json"), sha256(questions), available_at="2026-11-10")
        write_json(journal.out / "protocol.json", protocol)
        with patch("facade_change.agentic.pipeline._model_signature", side_effect=stub_signature):
            summary = inspection.run(questions, histories, config_path, journal.out / "protocol.json", journal.out / "analysis",
                pool_factory=QwenStub, detector_factory=DetectorStub, check_resources=False)
        if summary["status"] != "completed":
            raise AssertionError("Synthetic inspection failed: " + json.dumps(summary))
        scored = inspection.evaluate(questions, answers, journal.out / "analysis", journal.out / "evaluation")
        primary = scored["primary"]
        if abs(primary["heuristic"] - 1/3) > 1e-12 or primary["agent"] != 1.:
            raise AssertionError("Hand-checkable expected building F1 changed: " + str(primary))
        expected = dict(questions=5, buildings=4, positive_event_buildings=2,
            heuristic=dict(eligible_events=2, matched=1, unmatched_predictions=1, missed=1, mean_building_f1=1/3),
            agent=dict(eligible_events=2, matched=2, unmatched_predictions=0, missed=0, mean_building_f1=1.),
            difference=2/3, per_policy_outcomes=dict(completed=4, abstained=1, error=0))
        for name in ("heuristic", "agent"):
            evaluation = read_json(journal.out / "evaluation" / name / "evaluation.json")
            counts = inspection._at(evaluation["pooled"])
            for key in ("eligible_events", "matched", "unmatched_predictions", "missed"):
                if counts[key] != expected[name][key]:
                    raise AssertionError(f"Expected {name}.{key}={expected[name][key]}, got {counts[key]}")
        result = dict(status="completed", synthetic_fixture=True, learned_models_executed=False, expected=expected,
                      counters=summary["counters"], report_path=summary["report_path"], evaluation_report=scored["report_path"])
        write_json(journal.out / "summary.json", result)
        complete_run(journal, "completed", summary=result)
        return result
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
