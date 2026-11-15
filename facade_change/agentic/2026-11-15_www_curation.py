"""CPU inventory and private batch preparation; proposals never become approved gold here."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
import importlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from ..history_features import Fingerprints, _history_document, _inside, digest
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .evaluation import box_iou
from .report import _copy
from .runner import complete_run, new_run
from .store import temporal_value

qa = importlib.import_module(".2026-11-10_questions", __package__)
runner = importlib.import_module(".2026-11-15_www_runner", __package__)


def csv_rows(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def inventory(histories, out, *, candidates_path=None, exposure_path=None, addresses_path=None):
    root, fingerprints = Path(histories).resolve(), Fingerprints()
    document, manifest_sha = _history_document(root, fingerprints)
    candidates = qa._load_candidates(candidates_path, fingerprints) if candidates_path else None
    if candidates and any(r["source_provenance"]["history_manifest_sha256"] != manifest_sha for r in candidates["candidates"]):
        raise ValueError("Inventory and question candidates must use the exact same history revision")
    exposure = qa.read_bound(exposure_path, fingerprints) if exposure_path else {}
    if exposure and exposure.get("kind") != "www_exposure_audit":
        raise ValueError("Expected an explicit www_exposure_audit; unknown exposure is not untouched evaluation")
    audits = {r["building_id"]: r for r in exposure.get("buildings", [])}
    for row in audits.values():
        if row.get("cohort") == "evaluation":
            protocol_source = exposure.get("protocol_source", {})
            if not protocol_source.get("path") or protocol_source.get("sha256") != exposure.get("protocol_sha256"):
                raise ValueError("Evaluation exposure audit needs the actual frozen protocol file/hash")
            fingerprints.add(protocol_source["path"], protocol_source["sha256"])
            for source in row.get("audit_sources", []):
                if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
                    raise ValueError("Exposure audit sources need actual file paths and hashes")
                fingerprints.add(source["path"], source["sha256"])
    addresses = qa.read_bound(addresses_path, fingerprints) if addresses_path else {"buildings": []}
    address_rows = {r["building_id"]: r for r in addresses["buildings"]}
    for row in address_rows.values():
        if not row.get("address") or not row.get("source"):
            raise ValueError("An address needs genuine source metadata; building_id transliteration is not a source")
    for index, view in enumerate(document.get("views", [])):
        anchor = view.get("anchor", {})
        if anchor.get("address") and view["building_id"] not in address_rows:
            address_rows[view["building_id"]] = {"building_id": view["building_id"], "address": anchor["address"],
                "aliases": anchor.get("address_aliases", []), "source": {"path": str(root / "manifest.json"),
                    "sha256": manifest_sha, "field": f"views[{index}].anchor.address"},
                "verification": anchor.get("address_status", "source_metadata_not_independently_verified")}
    journal = new_run(out, {"history_manifest_sha256": manifest_sha, "exposure_audit": exposure,
        "candidate_sha256": sha256(candidates_path) if candidates_path else None}, "www_cpu_inventory")
    try:
        (journal.out / "assets").mkdir()
        groups, preview_cache, originals, owners = defaultdict(list), {}, {}, defaultdict(set)
        for track in document["tracks"]:
            groups[(track["building_id"], track["view_id"], track["split"])].append(track)
            for observation in track["observations"]:
                owners[observation["image_sha256"]].add((track["building_id"], track["split"]))
        rows, sections = [], []
        for (building, view, split), tracks in sorted(groups.items(), key=lambda p: str(p[0])):
            observations = [r for t in tracks for r in t["observations"]]
            years = sorted({r["year"] for r in observations})
            reasons = []
            if not building or not view:
                reasons.append("unresolved_identity")
            if len(years) < 2:
                reasons.append("fewer_than_two_dated_observations")
            if any(len(owners[r["image_sha256"]]) > 1 for r in observations):
                reasons.append("original_image_sha_cross_building_or_split")
            refs = []
            for row in sorted(observations, key=lambda r: (r["year"], r["image_sha256"])):
                key = (row["original_image_path"], row["image_sha256"])
                if key not in originals:
                    path = Path(key[0]).expanduser()
                    if path.is_file():
                        fingerprints.add(path, key[1])
                        originals[key] = "verified_bytes"
                    else:
                        originals[key] = "missing_original_file"
                ref = {"year": row["year"], "path": key[0], "sha256": key[1], "status": originals[key]}
                if ref not in refs:
                    refs.append(ref)
            if any(r["status"] != "verified_bytes" for r in refs):
                reasons.append("some_original_frames_unavailable_for_identity_audit")
            audit = audits.get(building, {})
            if split == "val" or audit.get("cohort") == "development":
                cohort, exposure_status = "development", "development_exposure"
            elif (split == "test" and audit.get("cohort") == "evaluation" and audit.get("prior_exposure") == "none_found"
                  and audit.get("reviewer") and audit.get("audit_sources") and exposure.get("protocol_sha256")):
                cohort, exposure_status = "evaluation", "audited_before_frozen_protocol_run"
            else:
                cohort, exposure_status = "unassigned", "training_exposure" if split == "train" else "earlier_exposure_not_audited"
            eligible = not reasons and cohort in {"development", "evaluation"}
            if cohort == "unassigned":
                reasons.append(exposure_status)
            preview = None
            original = next((r for r in refs if r["status"] == "verified_bytes"), None)
            if original:
                key = original["sha256"]
                if key not in preview_cache:
                    path = journal.out / "assets" / (key + "-preview.jpg")
                    with Image.open(original["path"]) as image:
                        rgb = image.convert("RGB")  # No EXIF transpose; thumbnail is review-only.
                        rgb.thumbnail((960, 720))
                        rgb.save(path)
                    preview_cache[key] = path.relative_to(journal.out).as_posix()
                preview = preview_cache[key]
            questions = [r for r in (candidates or {}).get("candidates", []) if r["building_id"] == building and r["view_id"] == view]
            row = {"building_id": building, "view_id": view, "split": split, "observation_years": years,
                "tracks": len(tracks), "question_proposals": len(questions), "question_counts_complete": bool(candidates and split in candidates["source"]["splits"]),
                "original_frames": refs, "reference_preview": preview, "reference_preview_date": original["year"] if original else None,
                "reference_preview_original_sha256": original["sha256"] if original else None, "address_metadata": address_rows.get(building),
                "address_status": "source_supplied" if building in address_rows else "unknown_not_inferred_from_id",
                "cohort": cohort, "exposure_status": exposure_status, "exposure_audit": audit or None,
                "eligible": eligible, "exclusion_reasons": reasons}
            rows.append(row)
            sections.append({"title": str(building) + " / " + str(view), "caption": json.dumps(row, ensure_ascii=False),
                "panels": [{"src": preview, "label": "Original-frame preview; full bytes referenced in inventory"}] if preview else []})
        result = {"schema_version": 1, "kind": "www_history_inventory", "history_manifest_sha256": manifest_sha,
            "candidate_sha256": sha256(candidates_path) if candidates_path else None, "views": rows,
            "unexported_frames": [{"building_id": v["building_id"], "view_id": v["view_id"],
                "image": f.get("image"), "status": f.get("status"), "reason": f.get("reason")}
                for v in document.get("views", []) for f in v.get("frames", []) if f.get("status") != "exported"],
            "input_sha256": fingerprints.files}
        write_json(journal.out / "inventory.json", result)
        csv_rows(journal.out / "archive-targets.csv", rows)
        write_static_gallery(journal.out, "WWW — all canonical tracks/splits, CPU inventory", sections,
            notes=["No authoritative address is inferred from an ID. Unknown earlier exposure is not untouched evaluation.",
                   "Question counts are proposals, not approved labels. Preview thumbnails are not model inputs."])
        fingerprints.verify()
        counts = Counter(r["cohort"] for r in rows)
        summary = {"status": "completed", "tracks": len(document["tracks"]), "views": len(rows),
            "buildings": len({r["building_id"] for r in rows if r["building_id"]}), "views_by_cohort": dict(counts),
            "eligible_views": sum(r["eligible"] for r in rows), "excluded_views": sum(not r["eligible"] for r in rows),
            "exclusion_reasons": dict(Counter(reason for r in rows for reason in r["exclusion_reasons"])),
            "report_path": str(journal.out / "gallery.html")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def select_batch(cases, inventory_document, *, size=100, per_building=12, seed="2026-11-15-www-v1", max_overlap=.5, cohort="development"):
    if size < 1 or per_building < 1 or not 0 <= max_overlap <= 1 or cohort not in {"development", "evaluation"}:
        raise ValueError("Invalid frozen batch selection limits")
    views = {(r["building_id"], r["view_id"]): r for r in inventory_document["views"]}
    buckets, excluded = defaultdict(list), []
    for case in cases:
        view = views.get((case["building_id"], case["view_id"]), {})
        if not view.get("eligible") or view.get("cohort") != cohort:
            excluded.append({"question_id": case["question_id"], "reason": "view_not_eligible_for_requested_cohort"})
        else:
            buckets[(case["building_id"], case["candidate_type"])].append(case)
    for rows in buckets.values():
        rows.sort(key=lambda r: (-len(set(r["task_dates"])), -(
            temporal_value(max(r["task_dates"])) - temporal_value(min(r["task_dates"]))), digest([seed, r["question_id"]])))
    order = sorted(buckets, key=lambda k: (digest([seed, k[0]]), k[1]))
    selected, counts = [], Counter()
    while len(selected) < size and any(buckets.values()):
        for key in order:
            if not buckets[key] or len(selected) >= size:
                continue
            row = buckets[key].pop(0)
            reason = None
            if counts[row["building_id"]] >= per_building:
                reason = "per_building_cap"
            elif any(row["building_id"] == p["building_id"] and row["view_id"] == p["view_id"] and
                    row["frame"]["anchor_sha256"] == p["frame"]["anchor_sha256"] and
                    box_iou(row["frame"]["anchor_bbox_xyxy"], p["frame"]["anchor_bbox_xyxy"]) > max_overlap for p in selected):
                reason = "overlapping_anchor_region"
            if reason:
                excluded.append({"question_id": row["question_id"], "reason": reason})
            else:
                selected.append(row)
                counts[row["building_id"]] += 1
    excluded += [{"question_id": r["question_id"], "reason": "batch_size_cap"} for rows in buckets.values() for r in rows]
    return selected, excluded


def batch(candidates_path, inventory_path, out, **options):
    fingerprints = Fingerprints()
    candidates = qa._load_candidates(candidates_path, fingerprints)
    inv = qa.read_bound(inventory_path, fingerprints)
    if inv.get("candidate_sha256") != sha256(candidates_path):
        raise ValueError("Batch inventory must count this exact candidate manifest")
    selected, excluded = select_batch(candidates["candidates"], inv, **options)
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "inventory_sha256": sha256(inventory_path), **options}, "www_batch")
    try:
        result = qa._private("www_question_selection", candidate_sha256=sha256(candidates_path),
            question_ids=[r["question_id"] for r in selected], excluded=excluded, policy={**options,
                "order": "round robin building/family; most dates/span first; SHA256(seed,question_id) ties",
                "physical_event_deduplication": "pending human event_id review; geometric overlap cap is not event deduplication"})
        write_json(journal.out / "selection.private.json", result)
        summary = {"status": "completed", "selected": len(selected), "buildings": len({r["building_id"] for r in selected}),
            "candidate_families": dict(Counter(r["candidate_type"] for r in selected)), "approved": 0,
            "excluded": dict(Counter(r["reason"] for r in excluded)), "target_is_not_guaranteed": True}
        write_json(journal.out / "summary.json", summary)
        fingerprints.verify()
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def handoff(bundle_path, out, candidates_path=None):
    """Read the two named PRIVATE JSON members; never index bundled text or images."""
    path, fingerprints = Path(bundle_path).resolve(), Fingerprints()
    fingerprints.add(path)
    with zipfile.ZipFile(path) as bundle:
        def member(suffix):
            names = [n for n in bundle.namelist() if n.endswith(suffix)]
            if len(names) != 1 or bundle.getinfo(names[0]).file_size > 10_000_000:
                raise ValueError("Missing/ambiguous/oversized private coordination JSON member")
            return json.loads(bundle.read(names[0]))
        review = member("2026-11-15-www-human-review-handoff.private.json")
        packet = member("2026-11-15-www-pilot-cases.private.json")
    if review.get("kind") != "www_human_text_review_handoff" or review.get("access") != qa.PRIVATE or packet.get("access") != qa.PRIVATE:
        raise ValueError("Expected the explicit private human text handoff, not public archive evidence")
    if review["source_candidates_sha256"] != packet["source_candidates_sha256"]:
        raise ValueError("Handoff and packet candidate revisions differ")
    cases = {r["question_id"]: r for r in packet["cases"]}
    if candidates_path:
        fingerprints.add(candidates_path, review["source_candidates_sha256"])
        original = qa._load_candidates(candidates_path, fingerprints)
        available = {r["question_id"]: r for r in original["candidates"]}
        for identifier, row in cases.items():
            clean = {**row, "observations": [{k: o[k] for k in qa.OBS_KEYS} for o in row["observations"]]}
            if any(clean[k] != available[identifier][k] for k in qa.PUBLIC_KEYS):
                raise ValueError("Packet question/geometry differs from the exact source candidates")
    if len(review["cases"]) != 16 or len(cases) != 16 or len({r["question_id"] for r in review["cases"]}) != 16:
        raise ValueError("The supplied handoff must retain all 16 distinct judgments")
    counts = Counter(r["group"] for r in review["cases"])
    if counts != {"earliest_presence": 5, "visible_change": 8, "no_substantial_visible_change": 3}:
        raise ValueError("Use the corrected 5/8/3 handoff, including case 03")
    for row in review["cases"]:
        if any(row[k] != cases[row["question_id"]][k] for k in ("building_id", "view_id", "region_id", "task_dates", "split")):
            raise ValueError("Human text judgment belongs to another question/frame")
        if row["number"] == 3 and (row["group"] != "visible_change" or not row.get("clarification_verbatim")):
            raise ValueError("Case 03's later clarification must be preserved")
        if row.get("localization_review_status") != "not_supplied":
            raise ValueError("This handoff does not contain reviewed boxes")
    journal = new_run(out, {"bundle_sha256": sha256(path), "candidate_sha256": review["source_candidates_sha256"]}, "www_private_handoff")
    try:
        write_json(journal.out / "handoff.private.json", {**review, "imported_from_sha256": sha256(path),
            "cli_approval_status": "pending_structured_binding_and_localization", "approved": 0,
            "original_candidates_validated": bool(candidates_path)})
        write_json(journal.out / "selection.private.json", qa._private("www_question_selection",
            candidate_sha256=review["source_candidates_sha256"], question_ids=[r["question_id"] for r in review["cases"]],
            policy={"basis": "all existing human text judgments; no new result-based selection", "physical_event_deduplication": "pending"}))
        summary = {"status": "completed", "text_judgments": 16, "group_counts": dict(counts), "approved": 0,
            "reviewed_boxes": 0, "candidate_validation": "verified" if candidates_path else "pending_owner_files",
            "case_03_clarification_preserved": True}
        write_json(journal.out / "summary.json", summary)
        fingerprints.verify()
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def selected_cases(candidates_path, selection_path, fingerprints):
    candidates = qa._load_candidates(candidates_path, fingerprints)
    selection = qa.read_bound(selection_path, fingerprints)
    if selection.get("kind") != "www_question_selection" or selection["candidate_sha256"] != sha256(candidates_path):
        raise ValueError("Selection must bind the original candidate manifest")
    rows = {r["question_id"]: r for r in candidates["candidates"]}
    ids = selection["question_ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= set(rows):
        raise ValueError("Invalid/duplicate selected question IDs")
    return candidates, [rows[key] for key in ids]


def public_candidates(candidates_path, selection_path, out, *, limit=None, knowledge_cutoff=None):
    fingerprints = Fingerprints()
    _, rows = selected_cases(candidates_path, selection_path, fingerprints)
    if limit is not None and limit < 1:
        raise ValueError("Smoke subset limit must be positive")
    if limit is None:
        chosen = rows
    else:
        # Stress the longest supplied histories while covering tasks, before any predictions.
        chosen = []
        for task in sorted({r["task"] for r in rows}):
            if len(chosen) < limit:
                chosen.append(max((r for r in rows if r["task"] == task), key=lambda r: len(set(r["task_dates"]))))
        for row in rows:
            if len(chosen) >= limit:
                break
            if row not in chosen:
                chosen.append(row)
    public = []
    for row in chosen:
        case = {k: copy.deepcopy(row[k]) for k in qa.PUBLIC_KEYS}
        if knowledge_cutoff:
            if temporal_value(knowledge_cutoff) < temporal_value(case["knowledge_cutoff"]):
                raise ValueError("Do not backdate the frozen candidate cohort")
            case["knowledge_cutoff"] = knowledge_cutoff
        for obs in case["observations"]:
            for key in ("rgb", "support"):
                fingerprints.add(obs[key + "_path"], obs[key + "_sha256"])
        public.append(case)
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "selection_sha256": sha256(selection_path),
        "limit": limit, "knowledge_cutoff": knowledge_cutoff, "scoring": "unscored_candidates"}, "www_public_candidates")
    try:
        payload = qa.validate_public({"schema_version": 1, "kind": "facade_qa_questions", "protocol": qa.PROTOCOL,
            "questions": public, "retrieval_contract": {"encoder": "jina-clip-v2", "comparison_variants": list(runner.CONDITIONS)}})
        write_json(journal.out / "questions.json", payload)
        write_json(journal.out / "selection.json", {"selected_question_ids": [r["question_id"] for r in chosen],
            "omitted_question_ids": [r["question_id"] for r in rows if r not in chosen],
            "policy": "full frozen selection, or maximum date coverage per task then frozen-order fill for a bounded smoke; ties retain selection order"})
        fingerprints.verify()
        summary = {"status": "completed", "questions": len(public), "scoring": "unscored_candidates", "approved": 0,
            "model_manifest": str(journal.out / "questions.json"), "model_manifest_sha256": sha256(journal.out / "questions.json")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def proposals(case, fingerprints, annotation_index=None):
    """Positive single-observation facts and explicitly unreviewed spatial proposals."""
    facts, maps = [], {}
    categories = []
    if annotation_index:
        _, entries, categories = qa._annotation_maps(annotation_index, case["source_provenance"]["history_manifest_sha256"], fingerprints)
    else:
        entries = {}
    for obs in case["observations"]:
        evidence = case.get("annotation_evidence", {}).get(obs["observation_id"])
        if not evidence:
            continue
        entry = entries.get(obs["observation_id"])
        if not entry or entry["track_id"] != case["region_id"] or entry["artifact_sha256"] != evidence["sha256"]:
            raise ValueError("Proposal annotation facts must bind the saved ontology and original observation")
        fingerprints.add(evidence["path"], evidence["sha256"])
        fingerprints.add(obs["support_path"], obs["support_sha256"])
        with np.load(evidence["path"], allow_pickle=False) as loaded:
            packed, support = loaded["packed"].copy(), loaded["support"].copy()
        with Image.open(obs["support_path"]) as image:
            actual = np.asarray(image) > 0
        if (packed.dtype != np.uint16 or packed.shape != tuple(case["frame"]["shape"]) or not np.array_equal(support, actual)
                or np.any(packed[~support]) or np.any(packed.astype(np.uint32) >> len(categories))):
            raise ValueError("Annotation proposal grid/support/ontology mismatch")
        maps[obs["observation_id"]] = packed, support
        for index, category in enumerate(categories):
            pixels = int(((packed & (1 << index)) != 0).sum())
            if pixels:
                facts.append({"observation_id": obs["observation_id"], "date": obs["date"], "category": category["name"],
                    "positive_pixels": pixels, "label_provenance": "existing_human_single_observation_annotation",
                    "annotation_artifact_sha256": evidence["sha256"], "temporal_gold": False})
    boxes = []
    first, last = case["observations"][0]["observation_id"], case["observations"][-1]["observation_id"]
    if case["task"] == "temporal_change" and first in maps and last in maps:
        import cv2
        a, sa = maps[first]
        b, sb = maps[last]
        changed = sa & sb & (a != 0) & (b != 0) & (a != b)
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
        regions = sorted(stats[1:count].tolist(), key=lambda v: (-v[4], v[1], v[0]))
        for left, top, width, height, area in regions[:4]:
            boxes.append({"event_id": "", "bbox_xyxy": [left, top, left+width, top+height],
                "coordinate_system": "region_crop_pixels", "region_id": case["region_id"],
                "proposal_pixels": area, "label_provenance": "automatic_spatial_annotation_difference_pending_review",
                "common_fraction": float((sa & sb)[top:top+height, left:left+width].mean()), "confirmed": False})
    return {"positive_annotation_facts": facts, "proposed_boxes": boxes,
            "caveat": "Unannotated is unknown; equal masks do not prove stability; differing masks do not prove temporal change."}


def review_page(candidates_path, selection_path, out, *, handoff_path=None, previous=None):
    fingerprints = Fingerprints()
    candidates, rows = selected_cases(candidates_path, selection_path, fingerprints)
    handoff_doc = qa.read_bound(handoff_path, fingerprints) if handoff_path else {}
    if handoff_doc and handoff_doc.get("source_candidates_sha256") != sha256(candidates_path):
        raise ValueError("Text review must bind these exact candidates")
    judgments = {r["question_id"]: r for r in handoff_doc.get("cases", [])}
    prior = qa.read_bound(previous, fingerprints) if previous else {}
    if prior and (prior.get("kind") != "facade_qa_review" or prior["candidate_sha256"] != sha256(candidates_path)):
        raise ValueError("Previous review belongs to another cohort")
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "selection_sha256": sha256(selection_path),
        "handoff_sha256": sha256(handoff_path) if handoff_path else None, "previous": str(previous) if previous else None}, "www_private_review_ui")
    try:
        (journal.out / "assets").mkdir()
        sources, items, sections = {}, [], []
        for case in rows:
            draft = proposals(case, fingerprints, candidates["source"].get("annotation_targets"))
            text = judgments.get(case["question_id"])
            state = prior.get("states", {}).get(case["question_id"], {})
            assessment = {"earliest_presence": "earliest_presence", "visible_change": "confirmed_change",
                "no_substantial_visible_change": "confirmed_stability"}.get((text or {}).get("group"),
                "earliest_presence" if case["task"] == "earliest_evidence" else "")
            label = {"assessment": assessment, "observability": "unknown", "reference_answer": (text or {}).get("answer_ru", case["provisional_answer"]),
                "limitations": [(text or {})["limits"]] if (text or {}).get("limits") else [], "archival_status": "unknown",
                "supporting_evidence_ids": [], "events": [], "label_provenance": "human_temporal_review_pending_structured_confirmation" if text else "automatic_proposal_pending_review",
                "archival_stratum": False, "known_support_exhaustive": False}
            if case["task"] == "earliest_evidence":
                label.update(earliest_observation_id=case["observations"][0]["observation_id"], present_by=case["observations"][0]["date"], onset="unknown")
            if state.get("label"):
                label = copy.deepcopy(state["label"])
                draft["proposed_boxes"] = [{**e, "confirmed": False, "label_provenance": "previous_review_revision"} for e in label["events"]]
            label["events"] = []  # Only individually confirmed boxes are included on explicit export.
            panels, images = [], []
            for obs in case["observations"]:
                fingerprints.add(obs["rgb_path"], obs["rgb_sha256"])
                relative = _copy(journal.out, obs["rgb_path"], journal.out, sources)
                images.append({"src": relative, "date": obs["date"], "observation_id": obs["observation_id"]})
                panels.append({"src": relative, "label": obs["date"] + " / " + obs["observation_id"]})
            item = {"question_id": case["question_id"], "question": state.get("reviewed_question", case["question"]),
                "building_id": case["building_id"], "region_id": case["region_id"], "task": case["task"], "frame": case["frame"],
                "revision": state.get("revision", 0), "previous_status": state.get("status", "candidate"), "images": images,
                "label": label, "human_text_judgment": text, **draft}
            items.append(item)
            sections.append({"title": case["question_id"], "caption": json.dumps(item, ensure_ascii=False), "panels": panels})
        payload = qa._private("www_review_page", candidate_sha256=sha256(candidates_path), items=items)
        write_json(journal.out / "drafts.private.json", payload)
        # Existing static gallery remains available, with a small separate editor over the same copied RGB.
        write_static_gallery(journal.out, "PRIVATE — draft questions and RGB", sections,
            notes=["Imported human text is retained. No boxes are imported from chat; automatic proposals are not gold.",
                   "Use review.html to confirm/edit localization and explicitly export CLI review decisions."])
        template = Path(__file__).with_name("2026-11-15_www_review.html").read_text(encoding="utf-8")
        embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        (journal.out / "review.html").write_text(template.replace("__REVIEW_DATA__", embedded), encoding="utf-8")
        fingerprints.verify()
        summary = {"status": "completed", "questions": len(items), "imported_text_judgments": len(judgments),
            "approved": 0, "annotation_facts": sum(len(r["positive_annotation_facts"]) for r in items),
            "automatic_proposals": sum(len(r["proposed_boxes"]) for r in items), "report_path": str(journal.out / "review.html")}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
