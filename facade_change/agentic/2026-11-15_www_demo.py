"""Synthetic integration fixture. Scripted responses are NOT model-quality evidence."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

from PIL import Image

from ..io import read_json, sha256, write_json
from .config import defaults
from .runner import complete_run, new_run

qa = importlib.import_module(".2026-11-10_questions", __package__)
archive = importlib.import_module(".2026-11-15_www_archive", __package__)
runner = importlib.import_module(".2026-11-15_www_runner", __package__)


class SyntheticPool:
    """Explicit test injection; production CLI cannot select this worker."""
    def __init__(self, models, *args, **kwargs):
        self.models, self.progress = models, None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def call(self, role, payload):
        usage = {"model_calls": 1, "input_tokens": 32, "output_tokens": 0, "images": 0}
        if role == "jina":
            return {"status": "ok", "vectors": [[1., 0., 0.]], "space": {"model": "synthetic_fixture",
                "revision": "fixture-v1", "representation": "joint_image_text", "preprocessing": {"synthetic": True}}, "usage": usage}
        messages = payload["messages"]
        initial = json.loads(messages[1]["content"][0]["text"])
        allowance = json.loads(messages[-1]["content"][0]["text"])
        observations = [json.loads(b["text"]) for b in messages[1]["content"][1:] if b["type"] == "text"]
        ids = [r["observation_id"] for r in observations]
        used_tool = any('"retrieval_status"' in b.get("text", "") for m in messages for b in m["content"])
        if allowance["allowed_action"] == "retrieve_or_final" and not used_tool:
            value = {"type": "retrieve", "query": initial["question"], "observation_id": ids[-1]}
        else:
            value = {"type": "final", "decision": "supported", "answer_text": "Синтетический ответ для проверки протокола.",
                "citations": ids, "observation_ids": [], "date_interval": None, "boxes": [], "limitations": []}
            if initial["task"] == "earliest_evidence":
                value.update(decision="earliest_presence", observation_ids=[ids[0]], date_interval={
                    "start": observations[0]["date"], "end": observations[0]["date"], "precision": "year", "uncertain": True})
            elif initial["task"] == "temporal_change":
                value.update(decision="change", observation_ids=ids, date_interval={"start": initial["task_dates"][0],
                    "end": initial["task_dates"][-1], "precision": "interval", "uncertain": True}, boxes=[{
                        "evidence_id": ids[-1], "coordinate_system": "region_crop_pixels", "bbox_xyxy": [3, 1, 6, 4]}])
        usage.update(output_tokens=60, images=sum(b["type"] == "image" for m in messages for b in m["content"]))
        return {"status": "ok", "text": json.dumps(value, ensure_ascii=False), "usage": usage,
                "generation": {"actual_output_tokens": 60, "max_new_tokens": payload["max_new_tokens"], "reached_max_new_tokens": False}}


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True)
    histories, annotations, _ = importlib.import_module(".2026-11-10_qa_demo", __package__).fixture(root / "inputs")
    # Distinct synthetic buildings need distinct original-frame bytes for the inventory leakage audit.
    # Change only a pixel outside every supplied crop while constructing this new fixture.
    history, targets = read_json(histories / "manifest.json"), read_json(annotations)
    paths = sorted({r["original_image_path"] for t in history["tracks"] for r in t["observations"]})
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.copy()
        image.putpixel((19, 19), (index, 0, 0))
        image.save(path)
    for track in history["tracks"]:
        track["anchor_sha256"] = sha256(track["observations"][0]["original_image_path"])
        for row in track["observations"]:
            row["image_sha256"] = sha256(row["original_image_path"])
            targets["observations"][row["observation_id"]]["image_sha256"] = row["image_sha256"]
    write_json(histories / "manifest.json", history)
    write_json(histories / history["report"] / "manifest.json", history)
    write_json(histories / history["report"] / "run.json", {"status": "completed", "synthetic_fixture": True,
                                                            "manifest_sha256": sha256(histories / "manifest.json")})
    targets["input_sha256"] = {str(histories / "manifest.json"): sha256(histories / "manifest.json")}
    write_json(annotations, targets)
    write_json(annotations.parent / "run.json", {"status": "completed", "synthetic_fixture": True,
                                                 "artifact_sha256": {"index.json": sha256(annotations)}})
    qa.generate(histories, root / "candidates", annotation_targets=annotations, knowledge_cutoff="2026-11-15",
                task_mode="retrospective", available_at="observation_date")
    rows = read_json(root / "candidates/candidates.json")["candidates"]
    selected = [next(r for r in rows if r["region_id"] == "earliest" and r["task"] == "earliest_evidence"),
                next(r for r in rows if r["region_id"] == "change")]
    questions = [{k: copy.deepcopy(row[k]) for k in qa.PUBLIC_KEYS} for row in selected]
    questions.append({**copy.deepcopy(questions[-1]), "question_id": "fixture-archival-question", "task": "archival_qa",
                      "question": "Что можно установить об истории этого участка по предоставленным источникам?"})
    public = qa.validate_public({"schema_version": 1, "kind": "facade_qa_questions", "protocol": qa.PROTOCOL,
        "questions": questions, "retrieval_contract": {"comparison_variants": list(runner.CONDITIONS)}})
    write_json(root / "questions.json", public)
    # Deliberately poisonous private sibling: the inference loader must never open it.
    write_json(root / "answers.private.json", {"access": "private_evaluation", "reviewed_labels": "DO_NOT_EXPOSE"})
    image = Path(selected[-1]["observations"][0]["rgb_path"])
    import shutil
    shutil.copyfile(image, root / "archive.png")
    record = {"evidence_id": "fixture-archive-b1", "record_type": "image", "building_id": "b1", "view_id": "b1-view",
        "identity_status": "verified", "source_page_url": "https://example.invalid/synthetic", "media_url": None,
        "title": "Explicit synthetic source", "publisher": "CPU fixture", "creator": None, "source_text": "Synthetic source text.",
        "observation_date": {"start": "2009", "end": "2009", "precision": "year", "basis": "synthetic fixture"},
        "publication_date": "2010", "first_known_available_at": None, "collected_at": "2026-11-15T10:00:00+03:00",
        "available_at": "2010", "availability_basis": "documented_publication", "date_source": "fixture metadata",
        "assets": [{"path": "archive.png", "sha256": sha256(root / "archive.png"), "media_type": "image/png",
                    "width": 10, "height": 10, "page_index": None}], "rights": {"statement": "synthetic", "url": None},
        "access_status": "downloaded", "source_warnings": ["Synthetic only"], "parent_evidence_id": None}
    package = {"schema_version": 1, "kind": "www_archival_evidence_package", "corpus_id": "cpu-fixture",
        "created_at": "2026-11-15T10:00:00+03:00", "task_mode": "retrospective", "knowledge_cutoff": "2026-11-15", "records": [record]}
    write_json(root / "package.json", package)
    config = defaults()
    config["cache_root"] = str(root / "cache")
    return root / "questions.json", root / "package.json", config


def demo(out, progress=None):
    journal = new_run(out, {"synthetic_fixture": True, "models_executed": False}, "www_cpu_demo", progress)
    try:
        questions, package, config = fixture(journal.out / "fixture")
        archive.import_archive(package, journal.out / "archive")
        archive.index_archive(journal.out / "archive/corpus.json", config, journal.out / "index", pool_factory=SyntheticPool, synthetic=True)
        result = runner.run(questions, config, journal.out / "inference", archive_index=journal.out / "index/index.json",
                            pool_factory=SyntheticPool, synthetic=True, progress=progress)
        if result["outcome_counts"] != {"completed": 9, "abstained": 0, "error": 0}:
            raise RuntimeError("Synthetic 3-task × 3-condition acceptance failed: " + str(result))
        summary = {**result, "synthetic_fixture": True, "models_executed": False, "quality_claim": None}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
