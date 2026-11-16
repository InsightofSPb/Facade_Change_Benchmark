"""Prospective RGB QA preparation; original segmentation partitions stay immutable."""
from collections import Counter, defaultdict
import copy
import importlib

from ..history_features import digest
from ..io import sha256
from .store import temporal_value

runner = importlib.import_module(".2026-11-15_www_runner", __package__)
qa = importlib.import_module(".2026-11-10_questions", __package__)
LEGACY = "segmentation-splits-2026-11-15-v1"
POLICY = "rgb-qa-2026-11-16-v2"
POLICIES = (LEGACY, POLICY)
CROSS_SPLIT_POLICY = "Original segmentation split overlap is recorded, not an RGB QA exclusion; QA development/evaluation overlap is checked separately"


def contract(name):
    if name not in POLICIES:
        raise ValueError("Unknown QA preparation policy")
    if name == LEGACY:
        return {"id": LEGACY}
    return {"id": POLICY, "allowed_original_splits": ["train", "val", "test"],
        "model_training": "Qwen/Jina are frozen; our segmentation predictions are not QA inputs",
        "unit": "reviewed physical building", "unknown_qa_exposure": "review allowed; not audited evaluation",
        "selection": "one question per physical building per deterministic pass; then type and independent event diversity",
        "soft_questions_per_type_per_building": 2, "hard_category_quotas": False,
        "max_local_images": runner.PROTOCOL["max_local_images"],
        "earliest_scope": "never truncate or rewrite; incompatible cases require an explicit reviewed revision"}


def annotate_inventory(rows, identities, exposure, exposed):
    """Keep legacy flags, but derive QA eligibility from its own exposure domain."""
    scoped = exposure.get("exposure_domain") == "rgb_qa" and exposure.get("qa_policy") == POLICY
    audits = {r["building_id"]: r for r in exposure.get("buildings", [])}

    def qa_use(building):
        audit = audits.get(building, {}) if scoped else {}
        known = bool(exposed.get(building) or audit.get("prior_exposure") == "used_for_qa_development")
        checked = bool(audit.get("prior_exposure") == "none_found" and audit.get("reviewer")
                       and audit.get("audit_sources") and exposure.get("protocol_sha256"))
        return {"building_id": building, "declared_cohort": audit.get("cohort", "unassigned"),
            "status": "known_qa_use" if known else "audited_none_found" if checked else "unknown",
            "known_exposure_manifests": exposed.get(building, []), "audit": audit or None}

    groups = defaultdict(list)
    for row in rows:
        building = row["building_id"]
        row.update(legacy_cohort=row["cohort"], legacy_exposure_status=row["exposure_status"],
            legacy_exclusion_reasons=list(row["exclusion_reasons"]), original_split=row["split"],
            physical_building_id=identities.physical(building),
            segmentation_usage={"original_split": row["split"], "recorded_use": audits.get(building, {}).get("segmentation_usage", "not_audited"),
                                "split_assignment_does_not_prove_actual_use": True},
            qa_exposure=qa_use(building))
        groups[row["physical_building_id"]].append(row)
    for physical, members in groups.items():
        names = {r["building_id"] for r in members}
        names.update(n for n in set(audits) | set(exposed) if identities.physical(n) == physical)
        for r in members:
            if not identities.pending(r["building_id"]):
                names.update(identities.candidate_members(r["building_id"]))
        uses = [qa_use(n) for n in sorted(names, key=str)]
        declared = {r["declared_cohort"] for r in uses}
        known = any(r["status"] == "known_qa_use" for r in uses)
        all_checked = all(r["status"] == "audited_none_found" for r in uses)
        overlap = "evaluation" in declared and ("development" in declared or known)
        cohort = ("development" if known or "development" in declared else
                  "evaluation" if all_checked and declared == {"evaluation"} else "unassigned")
        splits = sorted({r["split"] for r in members})
        for row in members:
            reasons = list(row["technical_exclusion_reasons"])
            if "original_image_sha_cross_building_or_split" in reasons:
                reasons.remove("original_image_sha_cross_building_or_split")
                if row["original_sha_cross_physical_building"]:
                    reasons.append("original_image_sha_cross_physical_building")
            if row["split"] not in {"train", "val", "test"}:
                reasons.append("original_split_not_in_rgb_qa_policy")
            if identities.pending(row["building_id"]):
                reasons.append("physical_identity_review_pending")
            status = "known_qa_use" if known else "audited_none_found" if all_checked else "unknown"
            row.update(cohort=cohort, exposure_status=status, qa_exposure_status=status,
                physical_group_splits=splits, segmentation_split_overlap=len(splits) > 1,
                qa_group_exposure=uses, qa_development_evaluation_overlap=overlap,
                qa_partition_issues=["physical_building_crosses_qa_development_evaluation"] if overlap else [],
                eligible=not reasons, review_eligible=not reasons,
                evaluation_eligible=not reasons and cohort == "evaluation" and not overlap,
                exclusion_reasons=reasons)
    return rows


def image_budget(case):
    """Mirror the existing runner's date/ID selection without reading images or changing scope."""
    limit = runner.PROTOCOL["max_local_images"]
    observations = sorted(case["observations"], key=lambda r: (temporal_value(r["date"]), r["observation_id"]))
    required = sorted(set(case["task_dates"]), key=temporal_value)
    dates = {r["date"] for r in observations}
    reasons = []
    if not required or not set(required) <= dates:
        reasons.append("required_dates_missing_from_observations")
    if len(required) > limit:
        reasons.append("required_dates_exceed_six_local_images")
    if case["task"] == "earliest_evidence" and len(observations) > limit:
        reasons.append("earliest_observation_set_exceeds_image_cap_scope_revision_required")
    if case["task"] == "temporal_change" and len(required) < 2:
        reasons.append("temporal_question_needs_two_dates")
    chosen = [next(r for r in observations if r["date"] == date) for date in required if date in dates]
    if not reasons:
        for row in observations:
            if row not in chosen and len(chosen) < limit:
                chosen.append(row)
    chosen.sort(key=lambda r: (temporal_value(r["date"]), r["observation_id"]))
    return {"question_id": case["question_id"], "compatible": not reasons, "reasons": reasons,
        "max_local_images": limit, "observation_count": len(observations), "required_dates": required,
        "selected_observation_ids": [r["observation_id"] for r in chosen] if not reasons else [],
        "optional_omitted_observation_ids": [r["observation_id"] for r in observations if r not in chosen] if not reasons else [],
        "scope_changed": False, "requires_reviewed_scope_revision": case["task"] == "earliest_evidence" and bool(reasons)}


def select(cases, inv, *, size, per_building, seed, max_overlap, cohort, excluded_cases, families, same_event):
    if size < 1 or per_building < 1 or not 0 <= max_overlap <= 1 or cohort not in {"all", "development", "evaluation"}:
        raise ValueError("Invalid RGB QA review selection limits")
    views = {(v["building_id"], v["view_id"]): v for v in inv["views"]}
    def physical(case):
        return inv.get("physical_building_ids", {}).get(case["building_id"],
            views.get((case["building_id"], case["view_id"]), {}).get("physical_building_id", case["building_id"]))
    buckets, excluded = defaultdict(lambda: defaultdict(list)), []
    for case in cases:
        view = views.get((case["building_id"], case["view_id"]), {})
        budget = image_budget(case)
        if not view.get("eligible") or cohort != "all" and view.get("cohort") != cohort:
            excluded.append({"question_id": case["question_id"], "reason": "view_not_eligible_for_requested_cohort", "details": view.get("exclusion_reasons", [])})
        elif not budget["compatible"]:
            excluded.append({"question_id": case["question_id"], "reason": "local_image_budget_incompatible", "details": budget["reasons"]})
        elif any(case["question_id"] == old["question_id"] or physical(case) == physical(old) and same_event(case, old, max_overlap)
                 for old in excluded_cases):
            excluded.append({"question_id": case["question_id"], "reason": "previously_exposed_question_or_overlapping_event"})
        else:
            buckets[physical(case)][families.get(case["candidate_type"], case["candidate_type"])].append(case)
    for by_type in buckets.values():
        for rows in by_type.values():
            rows.sort(key=lambda r: (-len(set(r["task_dates"])),
                -(temporal_value(max(r["task_dates"])) - temporal_value(min(r["task_dates"]))), digest([seed, r["question_id"]])))
    buildings = sorted(buckets, key=lambda b: digest([seed, b]))
    selected, building_counts, type_counts, total_types = [], Counter(), Counter(), Counter()
    while len(selected) < size:
        added = False
        for building in buildings:
            if len(selected) >= size:
                break
            if building_counts[building] >= per_building:
                continue
            # Exactly one selection per building per pass. Missing types never block another type.
            by_type = buckets[building]
            types = sorted(by_type, key=lambda f: (type_counts[building, f] >= 2, type_counts[building, f],
                                                   total_types[f], digest([seed, building, f])))
            found = False
            for family in types:
                while by_type[family]:
                    row = by_type[family].pop(0)
                    if any(physical(p) == building and same_event(row, p, max_overlap) for p in selected):
                        excluded.append({"question_id": row["question_id"], "reason": "overlapping_same_interval_event_proposal"})
                        continue
                    selected.append(row)
                    building_counts[building] += 1
                    type_counts[building, family] += 1
                    total_types[family] += 1
                    added = found = True
                    break
                if found:
                    break
        if not added:
            break
    for building, by_type in buckets.items():
        excluded.extend({"question_id": r["question_id"], "reason": "per_building_cap" if building_counts[building] >= per_building else "batch_size_cap"}
                        for rows in by_type.values() for r in rows)
    return selected, excluded


def composition(selected, inventory, families):
    views = {(v["building_id"], v["view_id"]): v for v in inventory["views"]}
    rows = [{"question_id": c["question_id"], "building_id": c["building_id"], "view_id": c["view_id"],
        "physical_building_id": views[c["building_id"], c["view_id"]]["physical_building_id"],
        "question_type": families.get(c["candidate_type"], c["candidate_type"]), "task": c["task"],
        "original_split": c["split"], "qa_exposure": views[c["building_id"], c["view_id"]]["qa_exposure_status"],
        "qa_cohort": views[c["building_id"], c["view_id"]]["cohort"], "image_budget": image_budget(c)} for c in selected]
    return {"cases": rows, "distributions": {key: dict(Counter(r[key] for r in rows)) for key in
        ("physical_building_id", "question_type", "original_split", "qa_exposure", "qa_cohort")},
        "building_types": {b: dict(Counter(r["question_type"] for r in rows if r["physical_building_id"] == b))
                           for b in sorted({r["physical_building_id"] for r in rows})}}


def archive_coverage(cases, inventory, corpus_path, identities, fingerprints, review_path=None, selected_ids=None):
    """Private descriptive coverage, never a selection filter or model input."""
    from pathlib import Path
    corpus = runner.archive.validate_package(qa.read_bound(corpus_path, fingerprints), Path(corpus_path).resolve().parent,
        fingerprints) if corpus_path else {"records": []}
    corpus_sha = sha256(corpus_path) if corpus_path else None
    records = {r["evidence_id"]: r for r in corpus["records"]}
    selected_ids = {c["question_id"] for c in cases} if selected_ids is None else set(selected_ids)
    def compatible(record, case):
        return (record["identity_status"] == "verified" and identities.compatible(record, case)
                and corpus.get("knowledge_cutoff") == case["knowledge_cutoff"]
                and temporal_value(record["available_at"]) <= temporal_value(case["knowledge_cutoff"]))
    review = qa.read_bound(review_path, fingerprints) if review_path else {}
    if review and (review.get("kind") != "www_archive_coverage_review" or review.get("access") != qa.PRIVATE
                   or review.get("corpus_sha256") != corpus_sha):
        raise ValueError("Private archive coverage review must bind this exact corpus SHA")
    searches = {}
    for row in review.get("buildings", []):
        if row["building_id"] in searches or row.get("search_status") not in {"not_documented", "not_searched", "searched_no_sources", "searched_with_sources"}:
            raise ValueError("Duplicate/invalid archive search review")
        if row["search_status"] != "not_documented" and (not row.get("reviewer") or not row.get("notes")):
            raise ValueError("Archive search assertions need an explicit reviewer and notes")
        if row["search_status"].startswith("searched_") and not row.get("audit_sources"):
            raise ValueError("A recorded archive search needs source paths/hashes")
        for source in row.get("audit_sources", []):
            fingerprints.add(source["path"], source["sha256"])
        searches[row["building_id"]] = row
    available = {c["question_id"]: c for c in cases}
    supports, seen = defaultdict(list), set()
    for row in review.get("support", []):
        pair = row["question_id"], row["evidence_id"]
        if pair in seen or pair[0] not in available or pair[1] not in records:
            raise ValueError("Unknown/duplicate private archive support judgment")
        if row.get("verdict") not in {"useful", "not_useful", "unresolved"} or not row.get("reviewer") or not row.get("notes"):
            raise ValueError("Archive support requires an explicit human verdict, reviewer and notes")
        if row["verdict"] == "useful" and not compatible(records[pair[1]], available[pair[0]]):
            raise ValueError("Useful archive support must retain reviewed identity, view and frozen-date restrictions")
        seen.add(pair)
        supports[pair[0]].append(copy.deepcopy(row))
    groups = defaultdict(list)
    for view in inventory["views"]:
        groups[view["physical_building_id"]].append(view)
    buildings = []
    for physical, views in sorted(groups.items(), key=lambda item: str(item[0])):
        names = sorted({v["building_id"] for v in views})
        evidence = [r["evidence_id"] for r in records.values() if identities.physical(r["building_id"]) == physical]
        logs = [r for b, r in searches.items() if identities.physical(b) == physical]
        buildings.append({"physical_building_id": physical, "original_building_ids": names,
            "source_evidence_ids": evidence, "source_presence": "present_in_corpus" if evidence else "no_source_in_supplied_corpus",
            "search_status": sorted({r["search_status"] for r in logs}) or ["not_documented"], "search_reviews": logs,
            "search_corpus_disagreement": bool(evidence and any(r["search_status"] == "searched_no_sources" for r in logs)),
            "human_confirmed_useful": sum(r["verdict"] == "useful" for c in cases if c["question_id"] in selected_ids and identities.physical(c["building_id"]) == physical
                                          for r in supports[c["question_id"]])})
    questions = [{"question_id": c["question_id"], "selected": c["question_id"] in selected_ids,
        "compatible_source_ids": [identifier for identifier, record in records.items() if compatible(record, c)],
        "human_support_review": supports[c["question_id"]],
        "human_usefulness_status": "reviewed" if supports[c["question_id"]] else "unreviewed",
        "human_confirmed_useful_ids": [r["evidence_id"] for r in supports[c["question_id"]] if r["verdict"] == "useful"]} for c in cases]
    return qa._private("www_rgb_archive_coverage", corpus_sha256=corpus_sha, identity_source=identities.source,
        selection_filter=False, buildings=buildings, questions=questions,
        note="Source presence is not human-confirmed usefulness. No source in corpus does not imply a search was conducted.")
