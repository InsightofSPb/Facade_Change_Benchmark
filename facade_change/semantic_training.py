"""Semantic split checks and review candidates from aligned COCO observations."""
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


def _transition_windows(reference, source, common, class_pairs, window_sizes,
                        min_support_fraction, min_transition_fraction,
                        min_component_pixels, *, event_types=("replacement",),
                        boundary_tolerance_pixels=2, categories=None):
    """Component-supported multilabel changes with same-class contour tolerance.

    Dilating the OLD same-class mask suppresses contour jitter without eroding a
    wholly new thin defect. Unknown pixels remain unknown at both dates. Raw
    context belongs to the ancestor raw component of the selected robust one;
    its other surviving subcomponents are counted separately, never pooled to
    make a candidate pass the area thresholds.
    """
    import cv2
    import numpy as np

    h, w = common.shape
    if categories is None:
        categories = list({row["channel"]: row for pair in class_pairs for row in pair}.values())
    raw_gain = source & ~reference
    raw_loss = reference & ~source
    raw_gain[~common] = 0
    raw_loss[~common] = 0
    gain, loss = np.zeros_like(reference), np.zeros_like(reference)
    gain_channels, loss_channels = set(), set()
    kernel = np.ones((2 * boundary_tolerance_pixels + 1,) * 2, np.uint8)
    for category in categories:
        bit = np.uint16(1 << category["channel"])
        old_mask, new_mask = (reference & bit) != 0, (source & bit) != 0
        old_near = cv2.dilate(old_mask.astype(np.uint8), kernel).astype(bool)
        new_near = cv2.dilate(new_mask.astype(np.uint8), kernel).astype(bool)
        gained, lost = common & new_mask & ~old_near, common & old_mask & ~new_near
        gain[gained] |= bit
        loss[lost] |= bit
        if gained.any():
            gain_channels.add(category["channel"])
        if lost.any():
            loss_channels.add(category["channel"])
    specs = []
    if "replacement" in event_types:
        specs.extend(("replacement", old, new) for old, new in class_pairs)
    if "appearance" in event_types:
        specs.extend(("appearance", None, new) for new in categories)
    if "disappearance" in event_types:
        specs.extend(("disappearance", old, None) for old in categories)
    windows = []
    for event_type, old, new in specs:
        if ((old is not None and old["channel"] not in loss_channels)
                or (new is not None and new["channel"] not in gain_channels)):
            continue
        if old is None:
            bit = np.uint16(1 << new["channel"])
            changed, raw_changed = (gain & bit) != 0, (raw_gain & bit) != 0
        elif new is None:
            bit = np.uint16(1 << old["channel"])
            changed, raw_changed = (loss & bit) != 0, (raw_loss & bit) != 0
        else:
            old_bit, new_bit = np.uint16(1 << old["channel"]), np.uint16(1 << new["channel"])
            changed = ((loss & old_bit) != 0) & ((gain & new_bit) != 0)
            raw_changed = ((raw_loss & old_bit) != 0) & ((raw_gain & new_bit) != 0)
        if not changed.any():
            continue
        number, components, stats, centers = cv2.connectedComponentsWithStats(
            changed.astype(np.uint8), connectivity=8)
        _, raw_components = cv2.connectedComponents(raw_changed.astype(np.uint8), connectivity=8)
        seen = set()
        for component in range(1, number):
            if int(stats[component, cv2.CC_STAT_AREA]) < min_component_pixels:
                continue
            cx, cy = centers[component]
            for size in window_sizes:
                if size > min(h, w):
                    continue
                x = min(max(0, int(round(cx - (size - 1) / 2))), w - size)
                y = min(max(0, int(round(cy - (size - 1) / 2))), h - size)
                if (x, y, size) in seen:
                    continue
                support = int(common[y:y+size, x:x+size].sum())
                # Count only the contiguous component that proposed this window.
                event = components[y:y+size, x:x+size] == component
                count = int(event.sum())
                if (count < min_component_pixels or support / size**2 < min_support_fraction
                        or count / size**2 < min_transition_fraction):
                    continue
                seen.add((x, y, size))
                raw_local = raw_components[y:y+size, x:x+size]
                ancestor = int(raw_local[event][0])
                raw_event = raw_local == ancestor
                ignored = raw_event & ~changed[y:y+size, x:x+size]
                windows.append({"event_type": event_type,
                                "from_class": old["name"] if old else None,
                                "to_class": new["name"] if new else None,
                                "from_category_id": old["id"] if old else None,
                                "to_category_id": new["id"] if new else None,
                                "bbox_canvas_xyxy": [x, y, x+size, y+size], "window_size": size,
                                "common_annotated_pixels": support, "candidate_pixels": count,
                                "raw_candidate_pixels": int(raw_event.sum()),
                                "ignored_boundary_pixels": int(ignored.sum()),
                                "other_robust_component_pixels": int((raw_event & ~ignored & ~event).sum()),
                                "component_pixels": int(stats[component, cv2.CC_STAT_AREA]),
                                "boundary_tolerance_pixels": boundary_tolerance_pixels,
                                "support_fraction": support / size**2,
                                "transition_fraction": count / size**2,
                                "candidate_mask": event.copy(), "raw_candidate_mask": raw_event.copy(),
                                "ignored_boundary_mask": ignored.copy()})
    return sorted(windows, key=lambda row: (-row["transition_fraction"],
                                           row["event_type"], row["from_class"] or "", row["to_class"] or "",
                                           row["window_size"], row["bbox_canvas_xyxy"]))


def _select_transition_windows(windows, limit):
    """Round-robin event/class groups, retaining strongest windows within groups."""
    groups = defaultdict(list)
    for row in windows:
        groups[(row["event_type"], row["from_class"] or "", row["to_class"] or "")].append(row)
    by_type = {kind: sorted(key for key in groups if key[0] == kind)
               for kind in ("replacement", "appearance", "disappearance")}
    # Interleave event types before class groups: numerous appearances must not
    # consume the entire review cap before any replacement is shown.
    keys = [by_type[kind][index]
            for index in range(max((len(rows) for rows in by_type.values()), default=0))
            for kind in by_type if index < len(by_type[kind])]
    selected, counts = [], Counter()
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if depth < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][depth])
                counts[key] += 1
                added = True
        if not added:
            break
        depth += 1
    report = [{"event_type": key[0], "from_class": key[1] or None, "to_class": key[2] or None,
               "qualifying_windows": len(groups[key]), "selected_windows": counts[key],
               "omitted_windows": len(groups[key]) - counts[key]} for key in keys]
    return selected, report


def _transition_alignments(path, method):
    """Read completed pair/batch provenance without touching image pixels."""
    from pathlib import Path
    from .io import read_json, sha256

    path = Path(path).expanduser().resolve()
    record = read_json(path / "run.json")
    if not str(record.get("status", "")).startswith("completed"):
        raise ValueError("Transition mining requires a completed alignment run")
    if record.get("kind") == "pair_alignment":
        inputs = [(path, record.get("config", {}).get("method", "unknown"))]
    elif record.get("kind") == "alignment_and_data_batch":
        results_path = path / "results.json"
        if record.get("artifact_sha256", {}).get("results.json") != sha256(results_path):
            raise ValueError("Alignment results hash mismatch or absent provenance")
        inputs = []
        for row in read_json(results_path):
            if method is not None and row["method"] != method:
                continue
            child = (path / row["path"]).resolve()
            if not child.is_relative_to(path):
                raise ValueError("Alignment child path leaves its batch")
            inputs.append((child, row["method"]))
    else:
        raise ValueError("Expected pair_alignment or alignment_and_data_batch")
    found = defaultdict(list)
    unreadable = []
    for child, child_method in inputs:
        if method is not None and child_method != method:
            continue
        if not (child / "run.json").is_file():
            unreadable.append({"path": str(child), "reason": "missing_pair_run"})
            continue
        child_hash = sha256(child / "run.json")
        item = read_json(child / "run.json")
        if record.get("kind") == "alignment_and_data_batch":
            relative = (child / "run.json").relative_to(path).as_posix()
            if record.get("artifact_sha256", {}).get(relative) != child_hash:
                raise ValueError(f"Alignment child provenance hash mismatch: {relative}")
        observations = item.get("observations", {})
        if not all(observations.get(role, {}).get("sha256") for role in ("reference", "source")):
            unreadable.append({"path": str(child), "reason": "missing_observation_provenance"})
            continue
        key = tuple(observations[role]["sha256"] for role in ("reference", "source"))
        found[key].append({"path": child, "record": item, "method": child_method,
                           "record_sha256": child_hash})
    return path, found, unreadable


def mine_transition_candidates(dataset, alignment_run, out, window_sizes=(50, 100),
                               splits=("train", "val"), min_support_fraction=.5,
                               min_transition_fraction=.02, min_component_pixels=25,
                               max_windows_per_pair=20, alignment_method=None,
                               all_class_pairs=True,
                               event_types=("replacement", "appearance", "disappearance"),
                               boundary_tolerance_pixels=2):
    """CPU-only review pack from existing first-anchor raw-RGB alignments.

    Absence is meaningful only within the annotated union at BOTH dates. Even
    there annotation completeness, occlusion and dense alignment remain unknown.
    No segmentation model, matching, TEST pixels, temporal metrics or inferred GT.
    """
    import csv
    import html
    from pathlib import Path
    import cv2
    import numpy as np
    from PIL import Image
    from .geometry import Canvas, expanded_canvas, warp_pair
    from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
    from .segmentation.data import PreparedData

    if (not window_sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 8
                               for size in window_sizes) or len(set(window_sizes)) != len(window_sizes)):
        raise ValueError("window_sizes must be distinct integers >=8")
    if (not 0 < min_support_fraction <= 1 or not 0 < min_transition_fraction <= 1
            or isinstance(min_component_pixels, bool) or not isinstance(min_component_pixels, int)
            or min_component_pixels < 1 or isinstance(max_windows_per_pair, bool)
            or not isinstance(max_windows_per_pair, int) or max_windows_per_pair < 1):
        raise ValueError("Positive pixel/window limits and fractions in (0,1] are required")
    if (isinstance(boundary_tolerance_pixels, bool) or not isinstance(boundary_tolerance_pixels, int)
            or boundary_tolerance_pixels < 0):
        raise ValueError("boundary_tolerance_pixels must be an integer >=0")
    if (not event_types or len(set(event_types)) != len(event_types)
            or not set(event_types) <= {"replacement", "appearance", "disappearance"}):
        raise ValueError("event_types must contain distinct replacement/appearance/disappearance choices")
    data = dataset if isinstance(dataset, PreparedData) else PreparedData(dataset, splits=splits)
    if tuple(data.splits) != tuple(splits):
        raise ValueError("Requested splits differ from the prepared reader")
    categories = [{**row, "channel": i} for i, row in enumerate(data.categories)]
    by_name = {row["name"]: row for row in categories}
    requested = ([(a["name"], b["name"]) for a in categories for b in categories if a != b]
                 if all_class_pairs else [("DELAMINATION", "SPALLING"), ("SPALLING", "REPAIRS")])
    class_pairs = [(by_name[a], by_name[b]) for a, b in requested if a in by_name and b in by_name]
    if not class_pairs and set(event_types) == {"replacement"}:
        raise ValueError("No requested directed class pair exists in the COCO ontology")
    alignment_path, alignments, unreadable = _transition_alignments(alignment_run, alignment_method)
    pairs = [row for row in data.manifest.get("pairs", []) if row.get("split") in splits]
    config = {"candidate_version": 2, "dataset": data.metadata, "alignment_run": str(alignment_path),
              "alignment_run_sha256": sha256(alignment_path / "run.json"),
              "splits": list(splits), "window_sizes": list(window_sizes),
              "min_support_fraction": min_support_fraction,
              "min_transition_fraction": min_transition_fraction,
              "min_component_pixels": min_component_pixels,
              "boundary_tolerance_pixels": boundary_tolerance_pixels,
              "boundary_distance": "Chebyshev radius in native reference-grid pixels; square same-class dilation",
              "event_types": list(event_types), "all_class_pairs": all_class_pairs,
              "max_windows_per_pair": max_windows_per_pair, "alignment_method": alignment_method,
              "class_pairs": [[a["name"], b["name"]] for a, b in class_pairs],
              "unavailable_class_pairs": [list(pair) for pair in requested
                                           if any(name not in by_name for name in pair)],
              "candidate_rule": "appearance/loss outside dilated same-class mask at the other date; replacement intersects robust loss(A) and gain(B)",
              "raw_mask_scope": "ancestor raw component of the selected robust component; any other robust subcomponents counted separately",
              "window_selection": "component-centered; strongest fraction within event/class groups; deterministic round-robin capped per pair",
              "window_counting": "same robust connected component must meet absolute and full-window area thresholds; window is not labelled entirely positive",
              "support": "common geometric opaque support AND annotated non-crowd union at both dates",
              "pair_policy": "prepared earliest reference per view; never adjacent or future matching",
              "interpretation": "semantic annotation transition candidates; human review required; not temporal ground truth"}
    out = new_directory(out)
    record = run_record("real_semantic_transition_candidates", config)
    write_json(out / "run.json", record)
    candidates, pair_rows, gallery, views = [], [], [], {}
    try:
        for image in data.images.values():
            if image.get("metadata_status") == "reviewed" and image.get("view_id") and image.get("year") is not None:
                key = (image["split"], image["building_id"], image["view_id"])
                entry = views.setdefault(key, {"split": key[0], "building_id": key[1], "view_id": key[2],
                                              "available_years": set(), "prepared_pairs": 0,
                                              "usable_pairs": 0, "usable_source_years": set()})
                entry["available_years"].add(int(image["year"]))
        for pair_index, pair in enumerate(pairs, 1):
            reference, source = [data.images[pair[key]] for key in ("reference_id", "source_id")]
            key = (pair["split"], pair["building_id"], pair["view_id"])
            view = views[key]
            view["prepared_pairs"] += 1
            pair_row = {**pair, "candidate_count": 0}
            pair_rows.append(pair_row)
            if (int(reference["year"]) >= int(source["year"])
                    or int(reference["year"]) != min(view["available_years"])):
                raise ValueError(f"Prepared pair violates first-anchor chronology: {pair.get('pair_id')}")
            matches = alignments.get((reference["sha256"], source["sha256"]), [])
            if not matches:
                pair_row.update(status="missing_alignment", reason="No matching endpoint SHA-256 pair")
                continue
            # Reject old split assignments even when image IDs have been renumbered.
            for match in matches:
                for role, current in (("reference", reference), ("source", source)):
                    previous = match["record"]["observations"][role]
                    for field in ("split", "building_id", "view_id", "year", "width", "height"):
                        if previous.get(field) != current.get(field):
                            raise ValueError(f"Alignment/current dataset {field} mismatch for {role}: "
                                             f"{previous.get(field)!r} != {current.get(field)!r}; "
                                             "rerun alignment on the current prepared split")
            accepted = [item for item in matches if
                        str(item["record"].get("status", "")).startswith("completed")
                        and item["record"].get("quality_gate", {}).get("passed") is True]
            if not accepted:
                pair_row.update(status="rejected_alignment", reason="No completed alignment passed its recorded gate",
                                attempts=[{"path": str(item["path"]),
                                           "status": item["record"].get("status"),
                                           "quality_gate": item["record"].get("quality_gate")} for item in matches])
                continue
            accepted.sort(key=lambda item: ({"cascade": 0, "sift": 1, "loftr": 2}.get(item["method"], 3),
                                             str(item["path"])))
            selected = accepted[0]
            child, child_record = selected["path"], selected["record"]
            if sha256(child / "run.json") != selected["record_sha256"]:
                raise ValueError("Alignment child provenance changed during mining")
            geometry_path = child / "geometry.json"
            if child_record.get("artifact_sha256", {}).get("geometry.json") != sha256(geometry_path):
                raise ValueError(f"Alignment geometry hash mismatch: {child}")
            geometry = read_json(geometry_path)
            canvas = expanded_canvas((reference["height"], reference["width"]),
                                     (source["height"], source["width"]), geometry["source_to_reference"])
            if ((canvas.width, canvas.height) != (geometry["width"], geometry["height"])
                    or not np.allclose(canvas.reference_to_canvas, geometry["reference_to_canvas"], atol=1e-10)
                    or not np.allclose(canvas.reference_to_canvas @ canvas.source_to_reference,
                                       geometry["source_to_canvas"], atol=1e-10)):
                raise ValueError("Stored alignment canvas/direction differs from native-scale geometry")
            print(f"Transition candidates {pair_index}/{len(pairs)}: {pair['view_id']} "
                  f"{reference['year']} -> {source['year']}", flush=True)
            arrays = [data.load_image(row["image_id"]) for row in (reference, source)]
            matrices = [canvas.reference_to_canvas, canvas.reference_to_canvas @ canvas.source_to_reference]
            packed, supports = [], []
            for (_, labels, opaque), matrix in zip(arrays, matrices):
                packed.append(cv2.warpPerspective(labels, matrix, (canvas.width, canvas.height),
                                                   flags=cv2.INTER_NEAREST, borderValue=0))
                supports.append(cv2.warpPerspective(opaque.astype(np.float32), matrix,
                                                      (canvas.width, canvas.height), flags=cv2.INTER_LINEAR,
                                                      borderValue=0) >= 1 - 1e-6)
            common = supports[0] & supports[1] & (packed[0] != 0) & (packed[1] != 0)
            windows = _transition_windows(*packed, common, class_pairs, window_sizes,
                                          min_support_fraction, min_transition_fraction, min_component_pixels,
                                          event_types=event_types, boundary_tolerance_pixels=boundary_tolerance_pixels,
                                          categories=categories)
            selected_windows, group_counts = _select_transition_windows(windows, max_windows_per_pair)
            pair_row.update(status="mined_needs_review", alignment_run=str(child),
                            alignment_run_sha256=selected["record_sha256"],
                            geometry_sha256=sha256(geometry_path), alignment_method=selected["method"],
                            alignment_quality_gate=child_record["quality_gate"],
                            common_annotated_pixels=int(common.sum()), qualifying_windows=len(windows),
                            truncated_windows=max(0, len(windows)-max_windows_per_pair),
                            candidate_count=len(selected_windows), window_counts_by_event_and_class=group_counts)
            view["usable_pairs"] += 1
            view["usable_source_years"].add(int(source["year"]))
            for window in selected_windows:
                event = window.pop("candidate_mask")
                raw_event = window.pop("raw_candidate_mask")
                ignored = window.pop("ignored_boundary_mask")
                x, y, x1, y1 = window["bbox_canvas_xyxy"]
                offset = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
                crop_canvas = Canvas(x1-x, y1-y, offset @ canvas.reference_to_canvas, canvas.source_to_reference)
                warped = warp_pair(arrays[0][0], arrays[0][2], arrays[1][0], arrays[1][2], crop_canvas)
                candidate_id = f"candidate-{len(candidates):05d}"
                directory = new_directory(out / candidate_id)
                for role in ("reference", "source"):
                    Image.fromarray(warped[role]).save(directory / f"{role}.png")
                    overlay = warped[role].copy()
                    overlay[event] = (overlay[event].astype(np.float32)*.4 + np.array([255, 45, 10])*.6).astype(np.uint8)
                    Image.fromarray(overlay).save(directory / f"{role}_candidate.png")
                Image.fromarray(event.astype(np.uint8)*255).save(directory / "candidate_mask.png")
                Image.fromarray(raw_event.astype(np.uint8)*255).save(directory / "raw_candidate_mask.png")
                Image.fromarray(ignored.astype(np.uint8)*255).save(directory / "ignored_boundary_mask.png")
                Image.fromarray(common[y:y1, x:x1].astype(np.uint8)*255).save(directory / "annotated_support.png")
                # Packed labels retain all overlap channels and can be re-inspected.
                np.savez_compressed(directory / "semantic_masks.npz",
                                    reference=packed[0][y:y1, x:x1], source=packed[1][y:y1, x:x1])
                result = {**window, "candidate_version": 2, "candidate_id": candidate_id, "path": candidate_id,
                          "pair_id": pair.get("pair_id"), "split": pair["split"],
                          "building_id": pair["building_id"], "view_id": pair["view_id"],
                          "reference_id": reference["image_id"], "source_id": source["image_id"],
                          "reference_year": reference["year"], "source_year": source["year"],
                          "image_sha256": {"reference": reference["sha256"], "source": source["sha256"]},
                          "reference_to_crop": crop_canvas.reference_to_canvas.tolist(),
                          "source_to_crop": (crop_canvas.reference_to_canvas @ crop_canvas.source_to_reference).tolist(),
                          "review_status": "needs_review", "is_temporal_ground_truth": False,
                          "physical_visibility": "unknown; opaque geometric support does not remove occluders",
                          "alignment_run": str(child), "alignment_quality_gate": child_record["quality_gate"],
                          "limitations": ["dense/local alignment not certified", "annotation completeness not certified",
                                          "category differences may be annotation disagreement, not physical change"]}
                candidates.append(result)
                title = html.escape(f"{pair['view_id']} {reference['year']} → {source['year']}: "
                                    f"{window['event_type']}: {window['from_class'] or '∅'} → {window['to_class'] or '∅'}")
                gallery.append(f'<h2>{title}</h2><p>{candidate_id}; {window["window_size"]} px window; '
                               f'candidate pixels: {window["candidate_pixels"]}/{window["window_size"]**2} '
                               f'({window["transition_fraction"]:.2%} of window area); '
                               f'raw component pixels: {window["raw_candidate_pixels"]}; '
                               f'boundary-excluded pixels: {window["ignored_boundary_pixels"]}; '
                               f'other retained subcomponent pixels: {window["other_robust_component_pixels"]}. '
                               'Area fractions are not confidence scores. Needs review.</p>'
                               + "".join(f'<img src="{candidate_id}/{name}.png" alt="{name}">'
                                         for name in ("reference", "source", "reference_candidate", "source_candidate",
                                                      "raw_candidate_mask", "candidate_mask", "ignored_boundary_mask")))
            write_json(out / "pairs.json", pair_rows)
            del arrays, packed, supports, common, windows, selected_windows
        view_rows = []
        for entry in views.values():
            entry["available_years"] = sorted(entry["available_years"])
            entry["usable_source_years"] = sorted(entry["usable_source_years"])
            entry["usable_observations"] = len(entry["usable_source_years"]) + int(entry["usable_pairs"] > 0)
            entry["history_of_at_least_3_observations"] = entry["usable_observations"] >= 3
            view_rows.append(entry)
        all_group_counts = {}
        for pair_row in pair_rows:
            for group in pair_row.get("window_counts_by_event_and_class", []):
                key = (group["event_type"], group["from_class"] or "", group["to_class"] or "")
                aggregate = all_group_counts.setdefault(key, {**group, "qualifying_windows": 0,
                                                              "selected_windows": 0, "omitted_windows": 0})
                for field in ("qualifying_windows", "selected_windows", "omitted_windows"):
                    aggregate[field] += group[field]
        summary = {"status": "completed_needs_review", "candidate_version": 2,
                   "out": str(out), "candidate_count": len(candidates),
                   "qualifying_windows": sum(row.get("qualifying_windows", 0) for row in pair_rows),
                   "omitted_windows": sum(row.get("truncated_windows", 0) for row in pair_rows),
                   "window_counts_by_event_and_class": [all_group_counts[key] for key in sorted(all_group_counts)],
                   "prepared_pairs": len(pairs), "pair_status_counts": dict(Counter(row["status"] for row in pair_rows)),
                   "alignment_inputs_without_observations": unreadable, "views": view_rows,
                   "test_rgb_loaded": False, "test_annotations_rasterized": False,
                   "quantitative_temporal_evaluation_allowed": False,
                   "note": "A missing candidate does not establish an unchanged region; candidates are a biased review shortlist. Window counts are not event counts: sizes, classes and event types can overlap."}
        write_json(out / "candidates.json", {"candidate_version": 2, "categories": categories, "candidates": candidates})
        write_json(out / "pairs.json", pair_rows)
        write_json(out / "summary.json", summary)
        fields = ["candidate_id", "split", "building_id", "view_id", "reference_year", "source_year",
                  "event_type", "from_class", "to_class", "window_size", "candidate_pixels",
                  "raw_candidate_pixels", "ignored_boundary_pixels", "other_robust_component_pixels",
                  "common_annotated_pixels",
                  "transition_fraction", "support_fraction", "review_status", "path"]
        with (out / "candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidates)
        (out / "gallery.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Real semantic transition candidates</title>'
            '<style>body{font:16px system-ui;margin:24px}img{width:200px;image-rendering:pixelated;margin:4px}</style>'
            '<h1>Real semantic transition candidates — review required</h1>'
            '<p>Original earlier/later RGB, then annotation candidate overlays. Red is an annotation-derived '
            'candidate, not confirmed change. Review alignment, occlusion, class consistency and missing labels. '
            'Then raw component, selected robust component and boundary-excluded pixels. '
            '∅ means this category is absent within common annotated support, not an unlabelled or healthy state. '
            'Candidate windows are not distinct physical events. No temporal metrics or automatically accepted ground truth.</p>'
            + "".join(gallery), encoding="utf-8")
        if sha256(alignment_path / "run.json") != config["alignment_run_sha256"]:
            raise ValueError("Alignment run changed during mining")
        for row in pair_rows:
            if row["status"] == "mined_needs_review" and (
                    sha256(Path(row["alignment_run"]) / "run.json") != row["alignment_run_sha256"]
                    or sha256(Path(row["alignment_run"]) / "geometry.json") != row["geometry_sha256"]):
                raise ValueError("Used alignment provenance or geometry changed during mining")
        finish_record(out, record, "completed_needs_review")
        return summary
    except Exception as exc:
        write_json(out / "pairs.json", pair_rows)
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
