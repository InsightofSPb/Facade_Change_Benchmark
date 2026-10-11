"""Add explicitly unverified observations to a semantic-only training manifest."""
from __future__ import annotations

import copy
from collections import Counter


def include_auxiliary_training(prepared: dict, split: dict, enabled: bool):
    """Preserve reviewed ownership while including usable unresolved TRAIN images.

    This does not establish physical-building disjointness for auxiliary images.
    Known holdout identities are blocked, including absent historical owners.
    No identities, temporal pairs, or metadata confirmations are manufactured.
    """
    manifest = copy.deepcopy(prepared)
    report = {"enabled": bool(enabled), "included_images": [], "blocked_images": [],
              "duplicate_images": [], "physical_disjointness_verified": True}
    holdout = {key: set() for key in ("sha256", "view_id", "building_id")}
    owners = split.get("building_assignments", {})
    holdout["building_id"].update(building for building, owner in owners.items()
                                  if owner in {"val", "test"})
    history = split.get("historical_cohort", []) + split.get("gold_cohort", [])
    for row in prepared["images"] + history:
        if row.get("split") in {"val", "test"} or owners.get(row.get("building_id")) in {"val", "test"}:
            for key in holdout:
                if row.get(key):
                    holdout[key].add(row[key])
    train_hashes = {row["sha256"] for row in prepared["images"]
                    if row.get("split") == "train" and row.get("sha256")}
    allowed_reasons = {"metadata_not_reviewed", "unknown_building"}

    for row in manifest["images"]:
        if row.get("split") in {"train", "val", "test"}:
            row["training_role"] = "reviewed"
            continue
        reasons = set(row.get("preparation_exclusion_reasons", []))
        if (not enabled or row.get("split") != "excluded" or row.get("image_status") != "ready"
                or not row.get("sha256") or not reasons or not reasons <= allowed_reasons):
            continue
        identity = {key: row.get(key) for key in ("image_id", "file_name", "sha256")}
        overlaps = [key for key, values in holdout.items() if row.get(key) and row[key] in values]
        if overlaps:
            reason = "known_holdout_overlap:" + ",".join(overlaps)
            row["auxiliary_training_exclusion_reason"] = reason
            report["blocked_images"].append({**identity, "reason": reason})
            continue
        if row["sha256"] in train_hashes:
            reason = "duplicate_train_sha256"
            row["auxiliary_training_exclusion_reason"] = reason
            report["duplicate_images"].append({**identity, "reason": reason})
            continue
        train_hashes.add(row["sha256"])
        row.update(split="train", semantic_eligible=True, temporal_eligible=False,
                   training_role="auxiliary_semantic_train")
        row["reviewed_preparation_exclusion_reasons"] = row.pop("preparation_exclusion_reasons", [])
        row["preparation_exclusion_reasons"] = []
        temporal_reasons = list(row.get("temporal_exclusion_reasons", []))
        if "unverified_physical_building" not in temporal_reasons:
            temporal_reasons.append("unverified_physical_building")
        row["temporal_exclusion_reasons"] = temporal_reasons
        report["included_images"].append(identity)

    report["physical_disjointness_verified"] = not report["included_images"]
    manifest["auxiliary_training"] = report
    manifest["physical_disjointness_verified"] = report["physical_disjointness_verified"]
    if "summary" in manifest:
        manifest["summary"]["reviewed_eligible_image_count"] = manifest["summary"].get("eligible_image_count")
        manifest["summary"]["eligible_image_count"] = sum(
            row.get("split") in {"train", "val", "test"} for row in manifest["images"])
        manifest["summary"]["image_splits"] = dict(Counter(row.get("split") for row in manifest["images"]))
        manifest["summary"]["semantic_eligible_image_count"] = sum(
            bool(row.get("semantic_eligible")) for row in manifest["images"])
    return manifest, report
