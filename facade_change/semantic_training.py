"""Semantic TRAIN inclusion and final checks for split leakage."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict


def audit_semantic_leakage(manifest: dict, split: dict | None = None) -> dict:
    """Audit final semantic rows and temporal pairs without inventing identities.

    Case-only identity variants are treated as equal; no fuzzy address matching is
    performed. Unknown TRAIN identities are allowed but prevent a claim of fully
    verified physical disjointness. Callers persist this report before rejecting
    a ``failed`` status, so failures remain inspectable.
    """
    split = split or {}
    partitions = {"train", "val", "test"}
    keys = ("building_id", "view_id", "sha256")
    ownership = {key: defaultdict(lambda: defaultdict(list)) for key in keys}
    view_buildings = defaultdict(set)
    violations, unverified_train = [], []
    images_by_id = {}
    counts = Counter()
    current_hashes = defaultdict(list)

    def normalized(value):
        return str(value).strip().casefold() if value is not None else ""

    def identity(row):
        return {key: row.get(key) for key in ("image_id", "file_name", "building_id")}

    def add_owner(row, origin):
        owner = row.get("split")
        if owner not in partitions:
            return
        for key in keys:
            value = normalized(row.get(key))
            if value:
                ownership[key][value][owner].append({"image_id": row.get("image_id"), "origin": origin})
        view, building = normalized(row.get("view_id")), normalized(row.get("building_id"))
        if view and building:
            view_buildings[view].add(building)

    for row in manifest.get("images", []):
        image_id = str(row.get("image_id"))
        if image_id in images_by_id:
            violations.append({"type": "duplicate_image_id", "image_id": row.get("image_id")})
        images_by_id[image_id] = row
        owner = row.get("split")
        if owner not in partitions:
            continue
        counts[owner] += 1
        add_owner(row, "current")
        if not normalized(row.get("sha256")):
            violations.append({"type": "missing_sha256", **identity(row), "split": owner})
        else:
            current_hashes[(owner, normalized(row["sha256"]))].append(row.get("image_id"))
        reviewed = bool(normalized(row.get("building_id"))) and row.get("metadata_status") == "reviewed"
        if not reviewed:
            if owner == "train":
                unverified_train.append(identity(row))
            else:
                violations.append({"type": "unreviewed_holdout_image", **identity(row), "split": owner})

    for field in ("historical_cohort", "gold_cohort"):
        for row in split.get(field, []):
            add_owner(row, field)
    for building, owner in split.get("building_assignments", {}).items():
        if owner not in partitions or not normalized(building):
            violations.append({"type": "invalid_building_assignment", "building_id": building, "split": owner})
        else:
            add_owner({"building_id": building, "split": owner}, "building_assignments")

    for key, values in ownership.items():
        for value, owners in sorted(values.items()):
            if len(owners) > 1:
                violations.append({"type": "cross_split_identity", "key": key, "value": value,
                                   "splits": sorted(owners), "observations": dict(owners)})
    for view, buildings in sorted(view_buildings.items()):
        if len(buildings) > 1:
            violations.append({"type": "view_has_multiple_buildings", "view_id": view,
                               "building_ids": sorted(buildings)})

    pairs = manifest.get("pairs", [])
    for pair in pairs:
        pair_id = pair.get("pair_id")
        endpoints = [images_by_id.get(str(pair.get(key))) for key in ("reference_id", "source_id")]
        if any(row is None for row in endpoints):
            violations.append({"type": "pair_missing_endpoint", "pair_id": pair_id})
            continue
        reference, source = endpoints
        if (pair.get("split") not in partitions
                or any(row.get("split") != pair.get("split") for row in endpoints)):
            violations.append({"type": "pair_split_mismatch", "pair_id": pair_id,
                               "pair_split": pair.get("split"),
                               "endpoint_splits": [row.get("split") for row in endpoints]})
        for key in ("building_id", "view_id"):
            values = [normalized(row.get(key)) for row in [pair, *endpoints]]
            if not all(values) or len(set(values)) != 1:
                violations.append({"type": "pair_identity_mismatch", "pair_id": pair_id, "key": key})
        if any(row.get("metadata_status") != "reviewed" or row.get("temporal_eligible") is False
               or row.get("training_role") == "auxiliary_semantic_train" for row in endpoints):
            violations.append({"type": "pair_ineligible_endpoint", "pair_id": pair_id})
        if (str(reference.get("image_id")) == str(source.get("image_id"))
                or normalized(reference.get("sha256")) == normalized(source.get("sha256"))):
            violations.append({"type": "pair_identical_observations", "pair_id": pair_id})

    return {"schema_version": 1, "status": "failed" if violations else "passed",
            "known_identity_disjointness_verified": not violations,
            "physical_disjointness_verified": not violations and not unverified_train,
            "image_counts": {name: counts[name] for name in ("train", "val", "test")},
            "pair_count": len(pairs), "unverified_train_images": unverified_train,
            "violations": violations,
            "same_split_duplicate_hashes": [
                {"split": owner, "sha256": digest, "image_ids": image_ids}
                for (owner, digest), image_ids in sorted(current_hashes.items()) if len(image_ids) > 1],
            "identity_comparison": "trimmed case-insensitive exact identifiers; no fuzzy address inference"}


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
