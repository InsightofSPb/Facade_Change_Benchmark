"""Collect the selected WWW questions and their actual annotation inputs for private review."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def collect(candidates_path, package, out, archive_corpus=None, decisions=()):
    candidates_path, package, out = [Path(p).expanduser().resolve() for p in (candidates_path, package, out)]
    if out.exists():
        raise FileExistsError(f"ZIP уже существует; выберите новый --out: {out}")
    source = read_json(candidates_path)
    if source.get("kind") != "facade_qa_candidates" or source.get("access") != "private_evaluation":
        raise ValueError("Нужен исходный приватный candidates.json")
    candidate_sha = sha256(candidates_path)
    selection_path = package / "batch/selection.private.json"
    selection = read_json(selection_path)
    if selection.get("kind") != "www_question_selection" or selection.get("candidate_sha256") != candidate_sha:
        raise ValueError("Выборка и candidates.json относятся к разным исходным данным")
    ids = selection["question_ids"]
    by_id = {row["question_id"]: row for row in source["candidates"]}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError("Выборка пуста, содержит повторы или неизвестные question_id")
    cases = [by_id[q] for q in ids]
    histories = Path(source["source"]["histories"]).resolve()
    history_path = histories / "manifest.json"
    history_sha = sha256(history_path)
    if any(c["source_provenance"]["history_manifest_sha256"] != history_sha for c in cases):
        raise ValueError("Истории изменились после создания выбранных вопросов")
    history = read_json(history_path)
    tracks = {row["track_id"]: row for row in history["tracks"]}
    annotation_path = source["source"].get("annotation_targets")
    annotations = read_json(annotation_path) if annotation_path else {}
    annotations_by_id = annotations.get("observations", {})
    bindings, files = {}, {}

    def add(filename, role, expected=None):
        original = str(filename)
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Не хватает {role}: {path}")
        if path not in files:
            actual = sha256(path)
            files[path] = {"path": path, "sha256": actual, "size_bytes": path.stat().st_size,
                           "member": f"assets/2026-11-16-{actual}{path.suffix.lower()}", "roles": []}
        entry = files[path]
        if expected and entry["sha256"] != expected:
            raise ValueError(f"SHA-256 не совпадает ({role}): {path}")
        if role not in entry["roles"]:
            entry["roles"].append(role)
        bindings[original] = entry["member"]
        return entry["member"]

    print(f"Вопросов: {len(cases)}. Проверяю RGB, маски, классы и исходные кадры...", flush=True)
    add(candidates_path, "original_candidates", candidate_sha)
    add(selection_path, "original_selection")
    add(history_path, "original_history_manifest", history_sha)
    if annotation_path:
        expected = source.get("input_sha256", {}).get(str(Path(annotation_path).resolve()))
        if not expected:
            raise ValueError("Исходный candidates.json не содержит хеш annotation_targets")
        add(annotation_path, "original_annotation_index_and_ontology", expected)
    drafts = None
    for name in ("run.json", "summary.json", "qa-policy.json", "identity-review.json",
                 "archive-coverage.private.json", "batch/composition.private.json",
                 "inventory/inventory.json", "review/drafts.private.json"):
        path = package / name
        if path.is_file():
            if name == "review/drafts.private.json":
                drafts = read_json(path)
                if drafts.get("candidate_sha256") != candidate_sha or {c["question_id"] for c in drafts["items"]} != set(ids):
                    raise ValueError("Редактор связан с другой выборкой/candidates.json")
            add(path, "review_package_metadata")
    if archive_corpus:
        add(archive_corpus, "public_archive_metadata_only", (drafts or {}).get("archive_corpus_sha256"))
    for path in decisions:
        add(path, "owner_saved_decisions_not_automatically_applied")
    project = Path(__file__).resolve().parents[1]
    for name in ("handoff.private.json", "selection.private.json"):
        path = project / "runs/2026-11-15-www-handoff-001" / name
        if path.is_file():
            add(path, "preserved_original_16_separate_candidate_binding")
    # Include saved decisions from this preparation folder, never model results.
    for pattern in ("*decisions*.private.json", "checked-*/reviewed.json"):
        for path in sorted(package.parent.glob(pattern)):
            add(path, "owner_saved_decisions_not_automatically_applied")

    selected_tracks, missing_annotations = [], []
    region_ids = list(dict.fromkeys(c["region_id"] for c in cases))
    for number, region_id in enumerate(region_ids, 1):
        track = tracks.get(region_id)
        if track is None:
            raise ValueError(f"В истории нет выбранной области: {region_id}")
        matching = [c for c in cases if c["region_id"] == region_id]
        if any((c["building_id"], c["view_id"], c["frame"]["anchor_sha256"]) !=
               (track["building_id"], track["view_id"], track["anchor_sha256"]) for c in matching):
            raise ValueError(f"Изменилась привязка области к зданию/кадру: {region_id}")
        selected_tracks.append(track)
        print(f"  [{number}/{len(region_ids)}] {track['building_id']} / {region_id}", flush=True)
        for obs in track["observations"]:
            add(histories / obs["rgb"], "history_rgb", obs["rgb_sha256"])
            add(histories / obs["support"], "geometric_support_not_visibility", obs["support_sha256"])
            add(obs["original_image_path"], "original_full_frame", obs["image_sha256"])
            mapped = annotations_by_id.get(obs["observation_id"])
            if mapped:
                if mapped["track_id"] != region_id or mapped["image_sha256"] != obs["image_sha256"]:
                    raise ValueError(f"Маска привязана к другой области/дате: {obs['observation_id']}")
                add(Path(annotation_path).resolve().parent / mapped["artifact_path"],
                    "packed_annotation_and_support", mapped["artifact_sha256"])
            else:
                missing_annotations.append(obs["observation_id"])
    for case in cases:
        for obs in case["observations"]:
            add(obs["rgb_path"], "selected_question_rgb", obs["rgb_sha256"])
            add(obs["support_path"], "selected_question_support", obs["support_sha256"])
        for evidence in case.get("annotation_evidence", {}).values():
            add(evidence["path"], "selected_question_annotation", evidence["sha256"])

    members = {}
    for entry in files.values():
        saved = members.setdefault(entry["member"], entry)
        saved["roles"] = sorted(set(saved["roles"]) | set(entry["roles"]))
    total = sum(entry["size_bytes"] for entry in members.values())
    print(f"Уникальных файлов: {len(members)}; до ZIP: {total / 2**20:.1f} MiB. Упаковываю...", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for number, (name, entry) in enumerate(members.items(), 1):
            # Hash the exact bytes being written, so a concurrent edit cannot enter unnoticed.
            check = hashlib.sha256()
            with entry["path"].open("rb") as src, archive.open(name, "w", force_zip64=True) as dst:
                while chunk := src.read(1024 * 1024):
                    check.update(chunk)
                    dst.write(chunk)
            if check.hexdigest() != entry["sha256"]:
                raise ValueError(f"Файл изменился во время упаковки; ZIP неполный: {entry['path']}")
            if number % 10 == 0 or number == len(members):
                print(f"  ZIP: {number}/{len(members)} файлов", flush=True)
        payload = {"schema_version": 1, "kind": "www_private_review_input_bundle", "access": "private_evaluation",
                   "created_at": datetime.now(timezone.utc).isoformat(), "source_candidates": str(candidates_path),
                   "source_candidates_sha256": candidate_sha, "selection_sha256": sha256(selection_path),
                   "cases": cases, "selected_history_tracks": selected_tracks,
                   "annotation_contract": annotations.get("contract"), "archive_paths": bindings,
                   "assets": [{k: v for k, v in entry.items() if k != "path"} for entry in members.values()],
                   "missing_annotation_observations": sorted(set(missing_annotations)),
                   "context_policy": "Additional history/full frames are private review context; question observations are unchanged.",
                   "archive_media": "Not resent: use existing V4.1; corpus metadata is included when supplied.",
                   "models_executed": False, "automatic_approvals": 0}
        archive.writestr("2026-11-16-review-inputs.private.json", json.dumps(payload, ensure_ascii=False, indent=2))
        summary = {"status": "completed", "questions": len(cases), "regions": len(region_ids),
                   "building_ids": len({c["building_id"] for c in cases}), "assets": len(members),
                   "question_types": dict(Counter(c["candidate_type"] for c in cases)),
                   "missing_annotation_observations": len(set(missing_annotations)),
                   "models_executed": False, "automatic_approvals": 0}
        archive.writestr("2026-11-16-run.json", json.dumps(summary, ensure_ascii=False, indent=2))
    summary.update(archive=str(out), size_mib=round(out.stat().st_size / 2**20, 1))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--archive-corpus", type=Path)
    parser.add_argument("--decisions", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        collect(args.candidates, args.review_package, args.out, args.archive_corpus, args.decisions)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Упаковка не завершена: {exc}\n")


if __name__ == "__main__":
    main()
