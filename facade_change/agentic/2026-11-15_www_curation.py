"""CPU inventory and private batch preparation; proposals never become approved gold here."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import html
import importlib
import json
from pathlib import Path
import shutil
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
identity = runner.archive.identity
REVIEW_TARGETS = {"earliest_evidence": 40, "visible_change": 30, "stability": 20, "insufficient_evidence": 10}
REVIEW_FAMILIES = {"earliest_damage": "earliest_evidence", "possible_change": "visible_change",
                  "stable_control": "stability", "uncertain_control": "insufficient_evidence"}


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


def inventory(histories, out, *, candidates_path=None, exposure_path=None, addresses_path=None,
              identity_map=None, exposed_questions=()):
    root, fingerprints = Path(histories).resolve(), Fingerprints()
    identities = identity.Identities(identity_map, fingerprints)
    exposed = {}
    for path in exposed_questions:
        public = qa.validate_public(qa.read_bound(path, fingerprints))
        for question in public["questions"]:
            exposed.setdefault(question["building_id"], []).append({"path": str(Path(path).resolve()), "sha256": sha256(path)})
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
                owners[observation["image_sha256"]].add((identities.physical(track["building_id"]), track["split"]))
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
            aliases = sorted(name for name in set(audits) | set(exposed) if identities.physical(name) == identities.physical(building))
            alias_exposure = [source for name in aliases for source in exposed.get(name, [])]
            if split == "val" or audit.get("cohort") == "development" or exposed.get(building):
                cohort, exposure_status = "development", "development_exposure"
            elif (split == "test" and audit.get("cohort") == "evaluation" and audit.get("prior_exposure") == "none_found"
                  and audit.get("reviewer") and audit.get("audit_sources") and exposure.get("protocol_sha256")):
                cohort, exposure_status = "evaluation", "audited_before_frozen_protocol_run"
            else:
                cohort, exposure_status = "unassigned", "training_exposure" if split == "train" else "earlier_exposure_not_audited"
            original_cohort, original_exposure_status = cohort, exposure_status
            if any(audits.get(name, {}).get("cohort") == "development" for name in aliases) or alias_exposure:
                cohort, exposure_status = "development", "development_exposure"
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
                "original_cohort": original_cohort, "original_exposure_status": original_exposure_status,
                "known_exposure_manifests": alias_exposure,
                "reviewed_alias_exposure_audits": {name: audits[name] for name in aliases if name in audits},
                "eligible": eligible, "exclusion_reasons": reasons}
            rows.append(row)
            sections.append({"title": str(building) + " / " + str(view), "caption": json.dumps(row, ensure_ascii=False),
                "panels": [{"src": preview, "label": "Original-frame preview; full bytes referenced in inventory"}] if preview else []})
        audited = {name for name, row in audits.items() if row.get("cohort") == "evaluation" and
                   row.get("prior_exposure") == "none_found" and row.get("reviewer") and row.get("audit_sources") and exposure.get("protocol_sha256")}
        identities.annotate_inventory(rows, audited_buildings=audited)
        for section, row in zip(sections, rows):
            section["caption"] = json.dumps(row, ensure_ascii=False)
        result = {"schema_version": 1, "kind": "www_history_inventory", "history_manifest_sha256": manifest_sha,
            "identity_source": identities.source,
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
            "physical_buildings": len({r["physical_building_id"] for r in rows if r["physical_building_id"]}),
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


def same_event_candidate(a, b, max_overlap):
    """Conservative spatiotemporal duplicate proposal, not proof of a physical event."""
    if a["frame"]["anchor_sha256"] != b["frame"]["anchor_sha256"] or a["view_id"] != b["view_id"]:
        return False
    if a["task"] == b["task"] == "earliest_evidence":
        same_dates = min(a["task_dates"]) == min(b["task_dates"])
    else:
        same_dates = sorted(a["task_dates"]) == sorted(b["task_dates"])
    x, y = a["frame"]["anchor_bbox_xyxy"], b["frame"]["anchor_bbox_xyxy"]
    intersection = max(0, min(x[2], y[2])-max(x[0], y[0])) * max(0, min(x[3], y[3])-max(x[1], y[1]))
    smaller = min((x[2]-x[0])*(x[3]-x[1]), (y[2]-y[0])*(y[3]-y[1]))
    return same_dates and smaller > 0 and intersection / smaller > max_overlap


def select_batch(cases, inventory_document, *, size=100, per_building=12, seed="2026-11-15-www-v1", max_overlap=.5,
                 cohort="development", targets=None, excluded_cases=()):
    if size < 1 or per_building < 1 or not 0 <= max_overlap <= 1 or cohort not in {"development", "evaluation"}:
        raise ValueError("Invalid frozen batch selection limits")
    views = {(r["building_id"], r["view_id"]): r for r in inventory_document["views"]}
    if targets is not None and (set(targets) != set(REVIEW_TARGETS) or
            any(type(v) is not int or v < 0 for v in targets.values()) or sum(targets.values()) != size):
        raise ValueError("Review targets must be four nonnegative provisional counts summing to batch size")
    def physical(row):
        return views.get((row["building_id"], row["view_id"]), {}).get("physical_building_id", row["building_id"])
    buckets, excluded = defaultdict(list), []
    for case in cases:
        view = views.get((case["building_id"], case["view_id"]), {})
        if not view.get("eligible") or view.get("cohort") != cohort:
            excluded.append({"question_id": case["question_id"], "reason": "view_not_eligible_for_requested_cohort",
                             "details": view.get("exclusion_reasons", [])})
        elif any(case["question_id"] == old["question_id"] or
                 physical(case) == physical(old) and same_event_candidate(case, old, max_overlap) for old in excluded_cases):
            excluded.append({"question_id": case["question_id"], "reason": "previously_exposed_question_or_overlapping_event"})
        else:
            family = REVIEW_FAMILIES.get(case["candidate_type"]) if targets is not None else case["candidate_type"]
            if family is None:
                excluded.append({"question_id": case["question_id"], "reason": "no_provisional_review_target"})
            else:
                buckets[(physical(case), family)].append(case)
    for rows in buckets.values():
        rows.sort(key=lambda r: (-len(set(r["task_dates"])), -(
            temporal_value(max(r["task_dates"])) - temporal_value(min(r["task_dates"]))), digest([seed, r["question_id"]])))
    order = sorted(buckets, key=lambda k: (digest([seed, k[0]]), k[1]))
    selected, counts, target_counts = [], Counter(), Counter()
    while len(selected) < size and any(buckets.values()):
        for key in order:
            if not buckets[key] or len(selected) >= size:
                continue
            row = buckets[key].pop(0)
            reason = None
            if targets is not None and target_counts[key[1]] >= targets[key[1]]:
                reason = "provisional_target_quota"
            elif counts[physical(row)] >= per_building:
                reason = "per_building_cap"
            elif targets is not None and any(physical(row) == physical(p) and same_event_candidate(row, p, max_overlap) for p in selected):
                reason = "overlapping_same_interval_event_proposal"
            elif targets is None and any(physical(row) == physical(p) and row["view_id"] == p["view_id"] and
                    row["frame"]["anchor_sha256"] == p["frame"]["anchor_sha256"] and
                    box_iou(row["frame"]["anchor_bbox_xyxy"], p["frame"]["anchor_bbox_xyxy"]) > max_overlap for p in selected):
                reason = "overlapping_anchor_region"
            if reason:
                excluded.append({"question_id": row["question_id"], "reason": reason})
            else:
                selected.append(row)
                counts[physical(row)] += 1
                target_counts[key[1]] += 1
    excluded += [{"question_id": r["question_id"], "reason": "batch_size_cap"} for rows in buckets.values() for r in rows]
    return selected, excluded


def batch(candidates_path, inventory_path, out, *, exposed_questions=(), **options):
    fingerprints = Fingerprints()
    candidates = qa._load_candidates(candidates_path, fingerprints)
    inv = qa.read_bound(inventory_path, fingerprints)
    if inv.get("candidate_sha256") != sha256(candidates_path):
        raise ValueError("Batch inventory must count this exact candidate manifest")
    old_cases = []
    for path in exposed_questions:
        old_cases.extend(qa.validate_public(qa.read_bound(path, fingerprints))["questions"])
    selected, excluded = select_batch(candidates["candidates"], inv, excluded_cases=old_cases, **options)
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "inventory_sha256": sha256(inventory_path), **options}, "www_batch")
    try:
        result = qa._private("www_question_selection", candidate_sha256=sha256(candidates_path),
            question_ids=[r["question_id"] for r in selected], excluded=excluded, policy={**options,
                "order": "round robin building/family; most dates/span first; SHA256(seed,question_id) ties",
                "physical_event_deduplication": "same-anchor/view/date overlap suppression; physical event_id still requires human review"},
            identity_source=inv.get("identity_source"), selected_at=datetime.now(timezone.utc).isoformat(),
            provisional_targets={r["question_id"]: REVIEW_FAMILIES.get(r["candidate_type"]) for r in selected}, input_sha256=fingerprints.files)
        write_json(journal.out / "selection.private.json", result)
        summary = {"status": "completed", "selected": len(selected), "buildings": len({r["building_id"] for r in selected}),
            "candidate_families": dict(Counter(r["candidate_type"] for r in selected)), "approved": 0,
            "excluded": dict(Counter(r["reason"] for r in excluded)), "target_is_not_guaranteed": True}
        targets = options.get("targets")
        counts = Counter(REVIEW_FAMILIES.get(r["candidate_type"]) for r in selected)
        summary.update(requested=options.get("size", 100), shortfall=max(0, options.get("size", 100)-len(selected)),
            physical_buildings=len({v.get("physical_building_id", v["building_id"]) for v in inv["views"]
                                    if any(r["building_id"] == v["building_id"] and r["view_id"] == v["view_id"] for r in selected)}),
            provisional_target_counts=dict(counts), target_shortfalls={k: max(0, v-counts[k]) for k, v in (targets or {}).items()},
            target_semantics="review sampling targets only, never imposed answer labels")
        summary["eligibility_reasons"] = dict(Counter(reason for row in excluded for reason in row.get("details", [])))
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


def review_page(candidates_path, selection_path, out, *, handoff_path=None, previous=None,
                archive_corpus=None, identity_map=None, progress=None):
    fingerprints = Fingerprints()
    candidates, rows = selected_cases(candidates_path, selection_path, fingerprints)
    selection = qa.read_bound(selection_path, fingerprints)
    identities = identity.Identities(identity_map, fingerprints)
    if selection.get("identity_source") is not None and selection["identity_source"] != identities.source:
        raise ValueError("Review page and batch must retain the same identity mapping")
    corpus = None
    corpus_sha = None
    if archive_corpus:
        corpus_path = Path(archive_corpus).resolve()
        corpus = runner.archive.validate_package(qa.read_bound(corpus_path, fingerprints), corpus_path.parent, fingerprints)
        corpus_sha = sha256(corpus_path)
    handoff_doc = qa.read_bound(handoff_path, fingerprints) if handoff_path else {}
    if handoff_doc and handoff_doc.get("source_candidates_sha256") != sha256(candidates_path):
        raise ValueError("Text review must bind these exact candidates")
    judgments = {r["question_id"]: r for r in handoff_doc.get("cases", [])}
    prior = qa.read_bound(previous, fingerprints) if previous else {}
    if prior and (prior.get("kind") != "facade_qa_review" or prior["candidate_sha256"] != sha256(candidates_path)):
        raise ValueError("Previous review belongs to another cohort")
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "selection_sha256": sha256(selection_path),
        "handoff_sha256": sha256(handoff_path) if handoff_path else None, "previous": str(previous) if previous else None,
        "archive_corpus_sha256": sha256(archive_corpus) if archive_corpus else None,
        "identity_source": identities.source}, "www_private_review_ui", progress)
    try:
        (journal.out / "assets").mkdir()
        sources, items, sections = {}, [], []
        for number, case in enumerate(rows, 1):
            journal.event("case_started", case_id=case["question_id"], index=number, total=len(rows))
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
            archive_items = []
            for record in (corpus or {}).get("records", []):
                if record["building_id"] not in identities.candidate_members(case["building_id"]):
                    continue
                available = (corpus["knowledge_cutoff"] == case["knowledge_cutoff"] and
                             runner.archive._date(record["available_at"], "available_at") <= temporal_value(case["knowledge_cutoff"]))
                row = {k: copy.deepcopy(record[k]) for k in ("evidence_id", "record_type", "building_id", "view_id", "identity_status",
                    "source_page_url", "media_url", "title", "publisher", "source_text", "observation_date", "publication_date",
                    "available_at", "date_source", "parent_evidence_id", "source_warnings")}
                row.update(corpus_sha256=corpus_sha, record_sha256=digest(record),
                    retrieval_eligible=available and record["identity_status"] == "verified" and identities.compatible(record, case),
                    correspondence_status="unreviewed; identity/date compatibility does not prove this target region is shown",
                    image=None, document=None)
                assets = runner.archive.image_assets(record)
                if assets:
                    asset = assets[0]
                    path = runner.archive._inside(corpus_path.parent, asset["path"])
                    row["image"] = {"src": _copy(journal.out, path, journal.out, sources), "sha256": asset["sha256"],
                                    "shape": [asset["height"], asset["width"]], "page_index": asset["page_index"]}
                pdf = next((a for a in record["assets"] if a["media_type"] == "application/pdf"), None)
                if pdf:
                    row["document"] = _copy(journal.out, runner.archive._inside(corpus_path.parent, pdf["path"]), journal.out, sources)
                archive_items.append(row)
            item = {"question_id": case["question_id"], "question": state.get("reviewed_question", case["question"]),
                "building_id": case["building_id"], "region_id": case["region_id"], "task": case["task"], "frame": case["frame"],
                "physical_building_id": identities.physical(case["building_id"]), "identity_review_pending": identities.pending(case["building_id"]),
                "provisional_review_target": selection.get("provisional_targets", {}).get(case["question_id"]),
                "archive_evidence": archive_items,
                "revision": state.get("revision", 0), "previous_status": state.get("status", "candidate"), "images": images,
                "label": label, "human_text_judgment": text, **draft}
            items.append(item)
            sections.append({"title": case["question_id"], "caption": json.dumps(item, ensure_ascii=False), "panels": panels})
            journal.event("case_prepared", case_id=case["question_id"], status="completed")
        payload = qa._private("www_review_page", candidate_sha256=sha256(candidates_path), items=items,
            identity_source=identities.source, archive_corpus_sha256=sha256(archive_corpus) if archive_corpus else None)
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


def review_package(candidates_path, histories, archive_corpus, out, *, identity_map=None, exposure_path=None,
                   exposed_questions=(), handoff_path=None, preserved_selection=None, previous=None,
                   targets=None, per_building=12, seed="2026-11-16-www-review-v1", progress=None):
    """CPU orchestration of the existing inventory/batch/editor; never reads model outcomes."""
    targets = dict(REVIEW_TARGETS if targets is None else targets)
    if bool(handoff_path) != bool(preserved_selection):
        raise ValueError("Preserve the original human handoff together with its bound selection")
    fingerprints = Fingerprints()
    fingerprints.add(candidates_path)
    fingerprints.add(archive_corpus)
    journal = new_run(out, {"candidate_sha256": sha256(candidates_path), "archive_corpus_sha256": sha256(archive_corpus),
        "targets": targets, "per_building": per_building, "seed": seed,
        "selection_basis": "public inputs and annotation drafting families before new model outcomes"}, "www_review_package", progress)
    try:
        if identity_map:
            identity.Identities(identity_map, fingerprints)
            shutil.copyfile(identity_map, journal.out / "identity-review.json")
        else:
            write_json(journal.out / "identity-review.json", identity.template())
        identity_path = journal.out / "identity-review.json"
        exposed_paths = list(exposed_questions)
        preserved = None
        if handoff_path:
            handoff_doc = qa.read_bound(handoff_path, fingerprints)
            if handoff_doc.get("source_candidates_sha256") != sha256(candidates_path):
                raise ValueError("Preserved human decisions must bind the exact original candidate bytes; do not rebase labels automatically")
            saved_selection = qa.read_bound(preserved_selection, fingerprints)
            handed_ids = [r["question_id"] for r in handoff_doc["cases"]]
            if len(set(handed_ids)) != len(handed_ids) or set(saved_selection["question_ids"]) != set(handed_ids):
                raise ValueError("Preserved selection must retain every original human decision, without adding cases")
            shutil.copyfile(handoff_path, journal.out / "preserved-handoff.private.json")
            # Existing decisions were already seen by people/models: retain them in a
            # separate section, and exclude their questions/overlapping events from new selection.
            public_candidates(candidates_path, preserved_selection, journal.out / "preserved-public")
            exposed_paths.append(journal.out / "preserved-public/questions.json")
        journal.event("review_stage", stage="инвентаризация identity, split и прежней экспозиции")
        inventory(histories, journal.out / "inventory", candidates_path=candidates_path,
            exposure_path=exposure_path, identity_map=identity_path, exposed_questions=exposed_paths)
        journal.event("review_stage", stage="отбор новых вопросов по целевым квотам до просмотра ответов моделей")
        selected = batch(candidates_path, journal.out / "inventory/inventory.json", journal.out / "batch",
            size=sum(targets.values()), targets=targets, per_building=per_building, seed=seed, max_overlap=.5,
            cohort="development", exposed_questions=exposed_paths)
        journal.event("review_stage", stage="экспорт датированных RGB и архивных источников для приватного review")
        reviewed = review_page(candidates_path, journal.out / "batch/selection.private.json", journal.out / "review",
            archive_corpus=archive_corpus, identity_map=identity_path, progress=progress)
        if handoff_path:
            journal.event("review_stage", stage="сохранение прежних человеческих решений отдельным разделом")
            preserved = review_page(candidates_path, preserved_selection, journal.out / "preserved-review",
                handoff_path=handoff_path, previous=previous, archive_corpus=archive_corpus, identity_map=identity_path, progress=progress)
        summary = {**selected, "approved": 0, "models_executed": False,
            "review_path": reviewed["report_path"], "preserved_review": preserved,
            "identity_review": str(identity_path), "report_path": str(journal.out / "index.html"),
            "full_target_cohort_created": selected["selected"] == sum(targets.values()),
            "preserved_cases_scope": "separate original review; not counted toward new provisional targets"}
        write_json(journal.out / "summary.json", summary)
        write_static_gallery(journal.out, "PRIVATE — new review selection and preserved human decisions", [], notes=[
            json.dumps(summary, ensure_ascii=False),
            "Open review/review.html for new questions; preserved-review/review.html retains the earlier judgments when supplied.",
            "identity-review.json is a proposal: review a separate copy, then create a new inventory/selection with that reviewed map.",
            "Archive correspondence proposals are separate from change-event GT. No labels are approved by this exporter."])
        links = '<li><a href="review/review.html">Новые вопросы и архивные источники</a></li>'
        if preserved:
            links += '<li><a href="preserved-review/review.html">Сохранённые человеческие решения, включая уточнения</a></li>'
        links += '<li><a href="inventory/gallery.html">Исходные IDs, split и exposure</a></li><li><a href="identity-review.json" download>Копия identity mapping для отдельного review</a></li>'
        (journal.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>PRIVATE WWW review</title>'
            '<h1>Приватный review, без автоматических одобрений</h1><ul>' + links + '</ul>'
            '<p>Квоты — цели просмотра. Недобор не заполняется повторяющимися событиями или неаудированными зданиями.</p>'
            '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
        fingerprints.verify()
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
