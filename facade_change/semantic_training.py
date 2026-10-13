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
                        boundary_tolerance_pixels=2, categories=None,
                        crop_mode="windows", context_padding=100):
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
            boxes = []
            if crop_mode == "component":
                x, y, width, height = map(int, stats[component, :4])
                boxes.append((x, y, x + width, y + height, None))
            else:
                for size in window_sizes:
                    if size <= min(h, w):
                        x = min(max(0, int(round(cx - (size - 1) / 2))), w - size)
                        y = min(max(0, int(round(cy - (size - 1) / 2))), h - size)
                        boxes.append((x, y, x + size, y + size, size))
            for x, y, x1, y1, size in boxes:
                if (x, y, x1, y1) in seen:
                    continue
                core_box = [x, y, x1, y1]
                core_area = (x1 - x) * (y1 - y)
                support = int(common[y:y1, x:x1].sum())
                event = components[y:y1, x:x1] == component
                count = int(event.sum())
                # Select on the unpadded region. Context must not dilute the
                # selection score or discard a thin, wholly new crack.
                if (count < min_component_pixels or support / core_area < min_support_fraction
                        or count / core_area < min_transition_fraction):
                    continue
                seen.add((x, y, x1, y1))
                if crop_mode == "component":
                    x, y = max(0, x-context_padding), max(0, y-context_padding)
                    x1, y1 = min(w, x1+context_padding), min(h, y1+context_padding)
                event = components[y:y1, x:x1] == component
                area = (x1-x) * (y1-y)
                raw_local = raw_components[y:y1, x:x1]
                ancestor = int(raw_local[event][0])
                raw_event = raw_local == ancestor
                ignored = raw_event & ~changed[y:y1, x:x1]
                windows.append({"event_type": event_type,
                                "from_class": old["name"] if old else None,
                                "to_class": new["name"] if new else None,
                                "from_category_id": old["id"] if old else None,
                                "to_category_id": new["id"] if new else None,
                                "bbox_canvas_xyxy": [x, y, x1, y1], "window_size": size,
                                "selection_bbox_canvas_xyxy": core_box,
                                "crop_mode": crop_mode, "crop_width": x1-x, "crop_height": y1-y,
                                "crop_area": area, "context_padding": context_padding if crop_mode == "component" else 0,
                                "selection_support_fraction": support / core_area,
                                "selection_transition_fraction": count / core_area,
                                "common_annotated_pixels": int(common[y:y1, x:x1].sum()),
                                "candidate_pixels": int(event.sum()),
                                "raw_candidate_pixels": int(raw_event.sum()),
                                "ignored_boundary_pixels": int(ignored.sum()),
                                "other_robust_component_pixels": int((raw_event & ~ignored & ~event).sum()),
                                "component_pixels": int(stats[component, cv2.CC_STAT_AREA]),
                                "boundary_tolerance_pixels": boundary_tolerance_pixels,
                                "support_fraction": float(common[y:y1, x:x1].mean()),
                                "transition_fraction": int(event.sum()) / area,
                                "candidate_mask": event.copy(), "raw_candidate_mask": raw_event.copy(),
                                "ignored_boundary_mask": ignored.copy()})
    return sorted(windows, key=lambda row: (-row["selection_transition_fraction"],
                                           row["event_type"], row["from_class"] or "", row["to_class"] or "",
                                           row["window_size"] or 0, row["bbox_canvas_xyxy"]))


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
                               boundary_tolerance_pixels=2, crop_mode="component", context_padding=100):
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
    from .segmentation.metrics import _gallery_colors, _write_full_semantic_view

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
    if crop_mode not in {"component", "windows"} or type(context_padding) is not int or context_padding < 0:
        raise ValueError("crop_mode must be component/windows; context_padding must be an integer >=0")
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
    version = 3 if crop_mode == "component" else 2
    colors = _gallery_colors(categories)
    config = {"candidate_version": version, "dataset": data.metadata, "alignment_run": str(alignment_path),
              "alignment_run_sha256": sha256(alignment_path / "run.json"),
              "splits": list(splits), "window_sizes": list(window_sizes),
              "crop_mode": crop_mode, "context_padding": context_padding if crop_mode == "component" else 0,
              "selection_uses_annotations": True,
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
              "window_selection": ("complete connected-component bounding box plus context, clipped to alignment canvas; gates on unpadded box"
                                   if crop_mode == "component" else "component-centered fixed windows") +
                                  "; strongest selection fraction within event/class groups; deterministic round-robin capped per pair",
              "window_counting": "same robust connected component must meet absolute and selection-box area thresholds before padding; crop is not labelled entirely positive",
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
            pair_row.update(reference_year=reference["year"], source_year=source["year"])
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
                                          categories=categories, crop_mode=crop_mode, context_padding=context_padding)
            selected_windows, group_counts = _select_transition_windows(windows, max_windows_per_pair)
            pair_row.update(status="mined_needs_review", alignment_run=str(child),
                            alignment_run_sha256=selected["record_sha256"],
                            geometry_sha256=sha256(geometry_path), alignment_method=selected["method"],
                            reference_to_canvas=canvas.reference_to_canvas.tolist(),
                            canvas_size=[canvas.width, canvas.height],
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
                    index = 0 if role == "reference" else 1
                    bits = packed[index][y:y1, x:x1]
                    valid = supports[index][y:y1, x:x1] & (bits != 0)
                    _write_full_semantic_view(warped[role], bits, valid, colors,
                                              directory / f"{role}_classes.png",
                                              directory / f"{role}_semantic.png")
                Image.fromarray(event.astype(np.uint8)*255).save(directory / "candidate_mask.png")
                Image.fromarray(raw_event.astype(np.uint8)*255).save(directory / "raw_candidate_mask.png")
                Image.fromarray(ignored.astype(np.uint8)*255).save(directory / "ignored_boundary_mask.png")
                Image.fromarray(common[y:y1, x:x1].astype(np.uint8)*255).save(directory / "annotated_support.png")
                Image.fromarray(warped["overlap"].astype(np.uint8)*255).save(directory / "geometric_support.png")
                shift_x, shift_y = canvas.reference_to_canvas[:2, 2]
                # Packed labels retain all overlap channels and can be re-inspected.
                np.savez_compressed(directory / "semantic_masks.npz",
                                    reference=packed[0][y:y1, x:x1], source=packed[1][y:y1, x:x1])
                result = {**window, "candidate_version": version, "candidate_id": candidate_id, "path": candidate_id,
                          "bbox_reference_xyxy": [int(x-shift_x), int(y-shift_y), int(x1-shift_x), int(y1-shift_y)],
                          "selection_uses_annotations": True,
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
                gallery.append(f'<h2>{title}</h2><p>{candidate_id}; {x1-x} × {y1-y} px crop; '
                               f'candidate pixels: {window["candidate_pixels"]}/{window["crop_area"]} '
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
        summary = {"status": "completed_needs_review", "candidate_version": version,
                   "crop_mode": crop_mode, "context_padding": config["context_padding"],
                   "selection_uses_annotations": True,
                   "out": str(out), "candidate_count": len(candidates),
                   "qualifying_windows": sum(row.get("qualifying_windows", 0) for row in pair_rows),
                   "omitted_windows": sum(row.get("truncated_windows", 0) for row in pair_rows),
                   "window_counts_by_event_and_class": [all_group_counts[key] for key in sorted(all_group_counts)],
                   "prepared_pairs": len(pairs), "pair_status_counts": dict(Counter(row["status"] for row in pair_rows)),
                   "alignment_inputs_without_observations": unreadable, "views": view_rows,
                   "test_rgb_loaded": False, "test_annotations_rasterized": False,
                   "quantitative_temporal_evaluation_allowed": False,
                   "note": "A missing candidate does not establish an unchanged region; candidates are a biased review shortlist. Window counts are not event counts: sizes, classes and event types can overlap."}
        write_json(out / "candidates.json", {"candidate_version": version, "categories": categories, "candidates": candidates})
        write_json(out / "pairs.json", pair_rows)
        write_json(out / "summary.json", summary)
        fields = ["candidate_id", "split", "building_id", "view_id", "reference_year", "source_year",
                  "event_type", "from_class", "to_class", "window_size", "crop_width", "crop_height",
                  "context_padding", "candidate_pixels",
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


def prepare_transition_review(candidate_runs, out, synthetic_run=None, preview_max_side=0):
    """Static overview of all cached TRAIN/VAL histories; no inference or event GT."""
    import hashlib
    import json
    from pathlib import Path
    from PIL import Image, ImageChops
    import numpy as np
    from .segmentation.metrics import semantic_legend
    from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
    from .visualization import write_static_gallery

    if not candidate_runs or type(preview_max_side) is not int or preview_max_side < 0:
        raise ValueError("Provide candidate runs and preview_max_side >=0 (0 preserves native dimensions)")

    def plain_path(value):
        path = Path(value).expanduser().absolute()
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"Symlink paths are not accepted: {path}")
        return path.resolve()

    def checked(root, relative, hashes=None):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact path leaves its source run")
        path = plain_path(root / relative)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing source artifact: {path}")
        if hashes is not None and hashes.get(relative.as_posix()) != sha256(path):
            raise ValueError(f"Artifact hash mismatch: {relative}")
        return path

    ownership, provenance, snapshots = {}, [], {}

    def own(row, image_hashes=()):
        split = row.get("split")
        if split not in {"train", "val"}:
            raise ValueError("Review accepts TRAIN/VAL only; TEST is excluded")
        for field, value in (("building", row.get("building_id")), ("view", row.get("view_id")),
                             *(("image", value) for value in image_hashes)):
            if field == "view" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {field} identity")
            key = field, value.strip().casefold()
            if ownership.setdefault(key, split) != split:
                raise ValueError("Building, view or image overlaps TRAIN and VAL")

    def source_record(root, kind, expected=None):
        path = checked(root, "run.json")
        digest, record = sha256(path), read_json(path)
        config = record.get("config", {})
        encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if (record.get("kind") != kind or not str(record.get("status", "")).startswith("completed")
                or hashlib.sha256(encoded).hexdigest() != record.get("config_sha256")
                or (expected is not None and expected != digest)):
            raise ValueError(f"Requires unchanged completed {kind}: {root}")
        snapshots[str(path)] = digest
        return record

    roots = sorted({plain_path(path) for path in candidate_runs})
    if synthetic_run is not None:
        synthetic_run = plain_path(synthetic_run)
    destination = plain_path(out)
    if any(destination.is_relative_to(root) for root in roots + ([synthetic_run] if synthetic_run else [])):
        raise ValueError("Review output must be outside immutable source runs")
    dataset_identity, categories, pairs, all_candidates = None, None, [], []
    for root in roots:
        record = source_record(root, "real_semantic_transition_candidates")
        config, hashes = record["config"], record.get("artifact_sha256", {})
        if config.get("candidate_version") not in {2, 3}:
            raise ValueError("Review requires v2/v3 candidate runs")
        if not config.get("splits") or not set(config["splits"]) <= {"train", "val"}:
            raise ValueError("Review accepts TRAIN/VAL only; TEST is excluded")
        identity = {key: config.get("dataset", {}).get(key) for key in ("paths", "sha256")}
        if not all(identity.values()) or (dataset_identity is not None and identity != dataset_identity):
            raise ValueError("Candidate runs must share dataset paths and hashes")
        dataset_identity = identity
        document = read_json(checked(root, "candidates.json", hashes))
        pair_document = read_json(checked(root, "pairs.json", hashes))
        if document.get("candidate_version") != config["candidate_version"]:
            raise ValueError("Unsupported or inconsistent candidate document version")
        if categories is not None and categories != document.get("categories"):
            raise ValueError("Candidate category ontologies differ")
        categories = document.get("categories", [])
        candidate_groups, identifiers = defaultdict(list), set()
        pair_ids = {str(row["pair_id"]) for row in pair_document}
        if len(pair_ids) != len(pair_document):
            raise ValueError("Pair IDs must be unique within each candidate run")
        for candidate in document["candidates"]:
            own(candidate, candidate.get("image_sha256", {}).values())
            if candidate["split"] not in config["splits"]:
                raise ValueError("Candidate split differs from declared TRAIN/VAL input")
            cid = candidate.get("candidate_id")
            if not cid or cid in identifiers or str(candidate.get("pair_id")) not in pair_ids:
                raise ValueError("Candidate IDs must be unique and belong to a recorded pair")
            identifiers.add(cid)
            bbox = candidate.get("bbox_canvas_xyxy", [])
            if (len(bbox) != 4 or any(type(v) is not int for v in bbox) or min(bbox) < 0
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                raise ValueError("Invalid candidate bounding box")
            candidate_groups[str(candidate["pair_id"])].append(candidate)
        for pair in pair_document:
            own(pair)
            if pair["split"] not in config["splits"]:
                raise ValueError("Pair split differs from declared TRAIN/VAL input")
            local = candidate_groups[str(pair["pair_id"])]
            if len(local) != pair.get("candidate_count", 0):
                raise ValueError("Pair candidate count does not match saved candidates")
            for item in local:
                if any(item.get(key) != pair.get(key) for key in
                       ("split", "building_id", "view_id", "reference_id", "source_id")):
                    raise ValueError("Candidate endpoints differ from its recorded pair")
            pairs.append({**pair, "candidate_run": str(root), "candidate_version": config["candidate_version"],
                          "candidates": local, "hashes": hashes})
        all_candidates.extend(document["candidates"])
        provenance.append({"run_path": str(root), "run_sha256": sha256(root / "run.json"),
                           "candidates_sha256": hashes["candidates.json"], "pairs_sha256": hashes["pairs.json"]})
        for name in ("candidates.json", "pairs.json"):
            snapshots[str(root / name)] = hashes[name]

    manifest_path = plain_path(dataset_identity["paths"].get("manifest", ""))
    manifest_sha = dataset_identity["sha256"].get("manifest")
    if not manifest_path.is_file() or not manifest_sha or sha256(manifest_path) != manifest_sha:
        raise ValueError("Prepared image manifest is missing or its hash differs from candidate provenance")
    manifest = read_json(manifest_path)
    images = {str(row["image_id"]): row for row in manifest.get("images", [])}
    if len(images) != len(manifest.get("images", [])):
        raise ValueError("Prepared manifest contains duplicate image IDs")
    snapshots[str(manifest_path)] = manifest_sha
    required_images = {}
    for pair in pairs:
        for role in ("reference", "source"):
            image_id = str(pair[role + "_id"])
            row = images.get(image_id)
            if (row is None or any(row.get(key) != pair.get(key) for key in ("split", "building_id", "view_id"))
                    or row.get("year") != pair.get(role + "_year")):
                raise ValueError("Prepared image identity/year differs from a recorded real pair")
            own(row, [row.get("sha256")])
            image_path = Path(row["image_path"]).expanduser()
            image_path = plain_path(image_path if image_path.is_absolute() else manifest_path.parent / image_path)
            if not image_path.is_file():
                raise ValueError(f"Required original image is unavailable: {image_path}; restore it before creating the complete history gallery")
            required_images[image_id] = {**row, "image_path": str(image_path)}

    synthetic, synthetic_record = None, None
    if synthetic_run:
        synthetic_record = source_record(synthetic_run, "history_dataset")
        config = synthetic_record["config"]
        if (not config.get("splits") or not set(config["splits"]) <= {"train", "val"}
                or config.get("input_sha256") != dataset_identity["sha256"]):
            raise ValueError("Synthetic history must use the same TRAIN/VAL dataset hashes")
        hashes = synthetic_record.get("artifact_sha256", {})
        synthetic = read_json(checked(synthetic_run, "manifest.json", hashes))
        if synthetic.get("kind") != "persistent_procedural_histories" or synthetic.get("config") != config:
            raise ValueError("Synthetic manifest differs from its run configuration")
        bases = {row["base_id"]: row for row in synthetic["bases"]}
        if len(bases) != len(synthetic["bases"]):
            raise ValueError("Duplicate synthetic base IDs")
        for base in bases.values():
            own(base, [base["image_sha256"]])
            if base["split"] not in config["splits"]:
                raise ValueError("Synthetic base split differs from declared TRAIN/VAL input")
        seen = set()
        for sequence in synthetic["sequences"]:
            base = bases.get(sequence["base_id"])
            key = sequence["base_id"], sequence["arm"]
            if (base is None or sequence["split"] != base["split"]
                    or sequence["building_id"] != base["building_id"]
                    or key in seen or sequence["arm"] not in {"clean", "event"}
                    or [f["frame"] for f in sequence["frames"]] != [1, 2, 3, 4, 5]):
                raise ValueError("Synthetic histories require matched five-frame clean/event arms")
            seen.add(key)
        if seen != {(key, arm) for key in bases for arm in ("clean", "event")}:
            raise ValueError("Synthetic history has missing paired arms")
        provenance.append({"run_path": str(synthetic_run), "run_sha256": sha256(synthetic_run / "run.json"),
                           "manifest_sha256": hashes["manifest.json"]})
        snapshots[str(synthetic_run / "manifest.json")] = hashes["manifest.json"]

    notes = ["Все подготовленные TRAIN/VAL истории: синтетика и реальные наблюдения. TEST не используется.",
             "Синтетика: H0 — без новых процедурных событий; исходный фасад не объявляется здоровым. H1 содержит известные добавленные события.",
             "Реальные данные показаны кропами после выравнивания: вся связная область перехода классов плюс контекст. Полные фасады не выводятся. Старые v2-запуски сохраняют размер своих окон; для контекста их нужно переэкспортировать.",
             "Фиолетовая область — объединение кандидатов разметки в одном кропе, не подтверждённое изменение. Отсутствие кандидатов не означает H0. Отклонённое выравнивание отмечено как пробел; выровненный кроп для него не придумывается.",
             "Классы реальных кандидатов перечислены под историей. Нужна проверка видимого изменения, совмещения и полноты аннотаций; метрики temporal GT не вычисляются.",
             "Области выбраны по аннотациям: это направленный просмотр переходов, а не независимая проверка обнаружения на всём фасаде. Истории одного участка используют общий прямоугольник в координатах ранней опоры; они могут содержать больше или меньше пяти пригодных дат.",
             "Классовые маски COCO имеют общую легенду; штриховка сохраняет пересечение классов. По клику открывается панель в полном размере.",
             f"Изображения — копии только для просмотра (JPEG); {'исходные размеры сохранены' if not preview_max_side else 'длинная сторона ограничена ' + str(preview_max_side) + ' px'}. Исходные RGB, маски и запуски не изменены."]
    config = {"sources": provenance, "dataset": dataset_identity, "preview_max_side": preview_max_side,
              "real_display": "aligned annotation-selected crops and same-region timelines",
              "timeline_min_geometric_support_fraction": .5,
              "selection_uses_annotations": True,
              "scope": "cached TRAIN/VAL crop review; no scoring, re-alignment or temporal ground truth"}
    out = new_directory(destination)
    record = run_record("static_history_review", config)
    write_json(out / "run.json", record)
    try:
        new_directory(out / "images")
        sections, exported, image_number, checked_originals = [], [], 0, set()
        crop_exports, crop_histories = [], []

        def rgb(path):
            with Image.open(path) as image:
                result = image.convert("RGB")
            if preview_max_side:
                result.thumbnail((preview_max_side, preview_max_side), Image.Resampling.LANCZOS)
            return result

        def panel(image, label):
            nonlocal image_number
            image_number += 1
            relative = f"images/{image_number:05d}.jpg"
            image.save(out / relative, quality=95, subsampling=0)
            if max(image.size) > 1600:
                preview = image.copy()
                preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                small = f"images/{image_number:05d}-preview.jpg"
                preview.save(out / small, quality=90, subsampling=0)
                return {"label": label, "src": small, "href": relative}
            return {"label": label, "src": relative}

        def tint(image, mask, color):
            mask = mask.resize(image.size, Image.Resampling.NEAREST).point(lambda x: 115 if x else 0)
            return Image.composite(Image.new("RGB", image.size, color), image, mask)

        if synthetic:
            by_base = defaultdict(dict)
            for sequence in synthetic["sequences"]:
                by_base[sequence["base_id"]][sequence["arm"]] = sequence
            for base in sorted(synthetic["bases"], key=lambda b: (b["split"], b["building_id"], b["base_id"])):
                title = f"СИНТЕТИКА | {base['split'].upper()} | {base['building_id']} | {base['base_id']}"
                for arm in ("clean", "event"):
                    raw_panels, overlay_panels = [], []
                    for frame in by_base[base["base_id"]][arm]["frames"]:
                        paths = frame["paths"]
                        image_path = checked(synthetic_run, paths["rgb"], synthetic_record["artifact_sha256"])
                        image = rgb(image_path)
                        label = f"Кадр {frame['frame']} · {frame['event']} · {frame['nuisance']['kind']}"
                        raw_panels.append(panel(image, label))
                        if arm == "event":
                            for name, color in (("damage_present", "#e53935"), ("new_change", "#ffb300"), ("repaired", "#00a878")):
                                mask_path = checked(synthetic_run, paths[name], synthetic_record["artifact_sha256"])
                                with Image.open(mask_path) as mask:
                                    image = tint(image, mask.convert("L"), color)
                            overlay_panels.append(panel(image, label))
                    sections.append({"title": title + (" · H0" if arm == "clean" else " · H1"),
                                     "caption": "Пять кадров, условия съёмки совпадают между H0/H1. " +
                                     ("Новых процедурных событий нет." if arm == "clean" else "Кадр 4 сохраняет повреждение без нового события."),
                                     "panels": raw_panels})
                    if arm == "event":
                        sections.append({"title": title + " · известные процедурные состояния/события",
                                         "caption": "Наложены скрытые процедурные маски: повреждение, новое изменение, восстановление. Под окклюзией они могут быть невидимы; это не видимая GT-маска.",
                                         "panels": overlay_panels})

        histories = defaultdict(list)
        for pair in pairs:
            histories[(pair["split"], pair["building_id"], pair["view_id"], pair["reference_id"])].append(pair)
        for key, chain in sorted(histories.items(), key=lambda item: tuple(map(str, item[0]))):
            title = f"РЕАЛЬНЫЕ | {key[0].upper()} | {key[1]} | {key[2]} | опора ID {key[3]}"
            captions, chain_sections, aligned, history_seeds = [], [], [], {}
            for pair in sorted(chain, key=lambda p: (p.get("source_year", 0), str(p["source_id"]))):
                for role in ("reference", "source"):
                    row = required_images[str(pair[role + "_id"])]
                    path = Path(row["image_path"])
                    if str(path) not in checked_originals:
                        if sha256(path) != row["sha256"]:
                            raise ValueError(f"Original image hash mismatch: {path}")
                        with Image.open(path) as native:
                            if native.size != (row["width"], row["height"]):
                                raise ValueError(f"Original image dimensions differ from manifest: {path}")
                        snapshots[str(path)] = row["sha256"]
                        checked_originals.add(str(path))
                dates = f"{pair.get('reference_year', '?')} → {pair.get('source_year', '?')}"
                exported.append({k: v for k, v in pair.items() if k != "hashes"})
                if pair["status"] != "mined_needs_review":
                    captions.append(dates + ": ПРОБЕЛ — " + pair["status"] + "; " + pair.get("reason", "выравнивание недоступно"))
                    continue
                child = plain_path(pair["alignment_run"])
                if destination.is_relative_to(child):
                    raise ValueError("Review output must be outside immutable alignment inputs")
                alignment = source_record(child, "pair_alignment", pair["alignment_run_sha256"])
                if alignment.get("quality_gate", {}).get("passed") is not True:
                    raise ValueError("A mined pair no longer passes its recorded alignment gate")
                observations = alignment.get("observations", {})
                for role in ("reference", "source"):
                    observation = observations.get(role, {})
                    own(observation, [observation.get("sha256")])
                    if any(observation.get(k) != pair.get(k) for k in ("split", "building_id", "view_id")):
                        raise ValueError("Alignment observation differs from candidate pair metadata")
                    if observation.get("year") != pair.get(role + "_year"):
                        raise ValueError("Alignment observation chronology differs from candidate pair")
                    if observation.get("sha256") != required_images[str(pair[role + "_id"])]["sha256"]:
                        raise ValueError("Alignment observation fingerprint differs from the prepared original")
                artifact_hashes = alignment.get("artifact_sha256", {})
                reference_path = checked(child, "reference_rgb.png", artifact_hashes)
                source_path = checked(child, "source_rgb.png", artifact_hashes)
                with Image.open(source_path) as native, Image.open(reference_path) as reference_native:
                    native_size = native.size
                    if reference_native.size != native_size:
                        raise ValueError("Saved aligned RGB pair has inconsistent canvas sizes")
                shift = None
                if pair["candidate_version"] == 3:
                    geometry_path = checked(child, "geometry.json", artifact_hashes)
                    geometry = read_json(geometry_path)
                    matrix = np.asarray(geometry["reference_to_canvas"], dtype=float)
                    saved_matrix = np.asarray(pair.get("reference_to_canvas"), dtype=float)
                    if (matrix.shape != (3, 3) or saved_matrix.shape != (3, 3)
                            or not np.isfinite(matrix).all()):
                        raise ValueError("Crop history requires finite 3x3 reference geometry")
                    shift = matrix[:2, 2].round().astype(int)
                    expected = np.array([[1., 0., shift[0]], [0., 1., shift[1]], [0., 0., 1.]])
                    if (not np.allclose(matrix, expected, atol=1e-10, rtol=0)
                            or not np.allclose(matrix, saved_matrix, atol=1e-10, rtol=0)
                            or [geometry["width"], geometry["height"]] != list(native_size)
                            or pair.get("geometry_sha256") != sha256(geometry_path)):
                        raise ValueError("Crop history reference coordinates differ from alignment geometry")
                    overlap_path = checked(child, "overlap.png", artifact_hashes)
                    with Image.open(overlap_path) as overlap:
                        if overlap.size != native_size:
                            raise ValueError("Alignment overlap has inconsistent canvas size")
                    aligned.append({"pair": pair, "source": source_path, "overlap": overlap_path,
                                    "shift": shift.tolist()})
                groups = defaultdict(list)
                for candidate in pair["candidates"]:
                    if (candidate.get("alignment_run") != str(child)
                            or candidate.get("image_sha256") != {role: observations[role]["sha256"] for role in ("reference", "source")}):
                        raise ValueError("Candidate alignment/endpoints differ from pair provenance")
                    box = candidate["bbox_canvas_xyxy"]
                    if box[2] > native_size[0] or box[3] > native_size[1]:
                        raise ValueError("Candidate bounding box exceeds saved alignment canvas")
                    groups[tuple(box)].append(candidate)
                captions.append(dates + (f": {len(groups)} кропов" if groups else ": кандидатов нет — отсутствие изменений не подтверждено"))
                for box, local in sorted(groups.items()):
                    size = (box[2]-box[0], box[3]-box[1])
                    union, labels = Image.new("L", size), set()
                    for candidate in local:
                        path = checked(Path(pair["candidate_run"]), Path(candidate["path"]) / "candidate_mask.png", pair["hashes"])
                        with Image.open(path) as mask:
                            if mask.size != size:
                                raise ValueError("Candidate mask dimensions differ from bounding box")
                            union = ImageChops.lighter(union, mask.convert("L"))
                        labels.add(f"{candidate['event_type']}: {candidate.get('from_class') or '∅'} → {candidate.get('to_class') or '∅'}")
                    crop_panels, crop_images = [], []
                    for role, path in (("reference", reference_path), ("source", source_path)):
                        with Image.open(path) as native:
                            image = native.convert("RGB").crop(box)
                        if preview_max_side:
                            image.thumbnail((preview_max_side, preview_max_side), Image.Resampling.LANCZOS)
                        crop_images.append(image)
                        crop_panels.append(panel(image, f"{pair[role + '_year']} · RGB · совмещённый кроп"))
                    for role in ("reference", "source"):
                        relative = Path(local[0]["path"]) / f"{role}_semantic.png"
                        if pair["candidate_version"] == 3 or relative.as_posix() in pair["hashes"]:
                            path = checked(Path(pair["candidate_run"]), relative, pair["hashes"])
                            with Image.open(path) as native:
                                if native.size != size:
                                    raise ValueError("Semantic crop overlay has inconsistent dimensions")
                            crop_panels.append(panel(rgb(path), f"{pair[role + '_year']} · GT классов COCO"))
                    for role, image in zip(("reference", "source"), crop_images):
                        crop_panels.append(panel(tint(image, union, "#a855f7"), f"{pair[role + '_year']} · кандидат изменения"))
                    crop_id = f"real-crop-{len(crop_exports):05d}"
                    crop_record = {"crop_id": crop_id, "candidate_run": pair["candidate_run"],
                                   "candidate_ids": [row["candidate_id"] for row in local],
                                   "pair_id": pair["pair_id"], "split": pair["split"],
                                   "building_id": pair["building_id"], "view_id": pair["view_id"],
                                   "bbox_canvas_xyxy": list(box), "native_size": list(size),
                                   "selection_uses_annotations": True, "panels": crop_panels}
                    crop_exports.append(crop_record)
                    padding = local[0].get("context_padding", 0)
                    chain_sections.append({"title": title + f" | {dates} | {crop_id}",
                                           "caption": f"{size[0]} × {size[1]} px; контекст до {padding} px с каждой стороны. " +
                                                      "; ".join(sorted(labels)) + ". Совпадающие прямоугольники показаны один раз; это кандидаты из разметки.",
                                           "panels": crop_panels})
                    if shift is not None:
                        anchor_box = tuple(int(v-shift[i % 2]) for i, v in enumerate(box))
                        if any(row.get("bbox_reference_xyxy") != list(anchor_box) for row in local):
                            raise ValueError("Candidate reference crop box differs from alignment coordinates")
                        history_seeds.setdefault(anchor_box, {"crop_id": crop_id, "reference_panel": crop_panels[0],
                                                             "reference_year": pair["reference_year"]})
            sections.append({"title": title, "caption": " | ".join(captions), "panels": []})
            sections.extend(chain_sections)
            for anchor_box, seed in history_seeds.items():
                timeline = [{**seed["reference_panel"], "label": f"{seed['reference_year']} · ранняя опора · тот же участок"}]
                frames, gaps = [], []
                for entry in aligned:
                    pair, shift = entry["pair"], entry["shift"]
                    box = tuple(int(v+shift[i % 2]) for i, v in enumerate(anchor_box))
                    with Image.open(entry["overlap"]) as overlap:
                        support = np.asarray(overlap.convert("L").crop(box)) > 0
                    fraction = float(support.mean())
                    frame = {"pair_id": pair["pair_id"], "year": pair["source_year"],
                             "bbox_canvas_xyxy": list(box), "geometric_support_fraction": fraction,
                             "alignment_run": pair["alignment_run"]}
                    if fraction < config["timeline_min_geometric_support_fraction"]:
                        frame["status"] = "insufficient_geometric_support"
                        gaps.append(f"{pair['source_year']}: участок виден геометрически менее чем на 50%")
                    else:
                        with Image.open(entry["source"]) as native:
                            image = np.array(native.convert("RGB").crop(box))
                        image[~support] = 70
                        preview = Image.fromarray(image)
                        if preview_max_side:
                            preview.thumbnail((preview_max_side, preview_max_side), Image.Resampling.LANCZOS)
                        frame["status"] = "aligned_crop_preview"
                        frame["panel"] = panel(preview, f"{pair['source_year']} · тот же участок · поддержка {fraction:.0%}")
                        timeline.append(frame["panel"])
                    frames.append(frame)
                crop_histories.append({"crop_id": seed["crop_id"], "split": key[0], "building_id": key[1],
                                       "view_id": key[2], "reference_id": key[3],
                                       "bbox_reference_xyxy": list(anchor_box), "frames": frames,
                                       "selection_uses_annotations": True, "preview_only": True})
                sections.append({"title": title + f" | {seed['crop_id']} | история одного участка",
                                 "caption": "Одинаковый прямоугольник ранней опоры во всех пригодных датах. Серый — нет общей геометрической поддержки; это не проверка окклюзий. " + " | ".join(gaps),
                                 "panels": timeline})
        legend = [{"name": "Текущее процедурное повреждение", "color": "#e53935"},
                  {"name": "Новое процедурное изменение", "color": "#ffb300"},
                  {"name": "Процедурное восстановление", "color": "#00a878"},
                  {"name": "Реальный кандидат из аннотаций — не GT", "color": "#a855f7"}] + semantic_legend(categories)
        write_static_gallery(out, "Истории фасадов: синтетика и реальные наблюдения", sections, legend=legend, notes=notes)
        write_json(out / "histories.json", {"sources": provenance, "real_pairs": exported, "real_images": list(required_images.values()),
                                             "synthetic_run": str(synthetic_run) if synthetic_run else None,
                                             "categories": categories, "real_crops": crop_exports,
                                             "real_crop_histories": crop_histories})
        for path, expected in snapshots.items():
            if sha256(path) != expected:
                raise ValueError("Source provenance changed during gallery preparation")
        summary = {"status": "completed_needs_review", "out": str(out), "gallery": "gallery.html",
                   "synthetic_bases": len(synthetic["bases"]) if synthetic else 0,
                   "synthetic_sequences": len(synthetic["sequences"]) if synthetic else 0,
                   "real_histories": len(histories), "real_pairs": len(pairs), "real_original_frames": len(required_images),
                   "pair_status_counts": dict(Counter(pair["status"] for pair in pairs)),
                   "input_candidates": len(all_candidates), "rendered_images": image_number,
                   "real_crops": len(crop_exports), "real_crop_histories": len(crop_histories),
                   "real_display": "aligned_crops", "selection_uses_annotations": True,
                   "quantitative_temporal_evaluation_allowed": False, "test_rgb_loaded": False,
                   "annotation_files_opened": False, "preview_max_side": preview_max_side}
        write_json(out / "summary.json", summary)
        finish_record(out, record, "completed_needs_review")
        return summary
    except Exception as exc:
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
