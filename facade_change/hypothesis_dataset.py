"""Deterministic, building-disjoint procedural H0/H1 cases from indexed crops."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json


LABEL_SCOPE = "known procedural state edits only; not real facade damage or real temporal change ground truth"
IDENTITY = np.eye(3).tolist()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be relative to their dataset root")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Artifact path escapes its dataset root")
    return path


def _check_parent_file(root, record, relative):
    path = _inside(root, relative)
    digest = sha256(path)
    if record.get("artifact_sha256", {}).get(relative) != digest:
        raise ValueError(f"Parent artifact changed or its hash is absent: {relative}")
    return path, digest


def _selection(rows, split, selection):
    requested = selection.get("splits", ["train", "val", "test"])
    per_building = selection.get("crops_per_building", 2)
    limit = selection.get("max_buildings_per_split", 0)
    minimum = selection.get("min_valid_fraction", .95)
    if not isinstance(requested, list) or not requested or len(set(requested)) != len(requested) or any(part not in {"train", "val", "test"} for part in requested):
        raise ValueError("selection.splits must contain unique reviewed partitions")
    if type(per_building) is not int or per_building < 1 or type(limit) is not int or limit < 0:
        raise ValueError("Selection budgets must be integers: crops >=1, buildings >=0")
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not 0 < minimum <= 1:
        raise ValueError("selection.min_valid_fraction must be in (0, 1]")
    groups, reasons, seen_ids, ownership = {}, {}, set(), {}
    for row in rows:
        identifier = row["dataset_crop_id"]
        if identifier in seen_ids:
            raise ValueError("Duplicate indexed dataset_crop_id")
        seen_ids.add(identifier)
        building, part = row.get("building_id"), row.get("split")
        if not building or part not in {"train", "val", "test"} or split["building_assignments"].get(building) != part:
            raise ValueError("Indexed crop disagrees with its reviewed building split")
        if building in ownership and ownership[building] != part:
            raise ValueError("One building appears in multiple partitions")
        ownership[building] = part
        fraction = row.get("valid_fraction")
        if not isinstance(fraction, (float, int)) or isinstance(fraction, bool) or not 0 <= fraction <= 1:
            raise ValueError("Indexed crop has invalid valid_fraction")
        if part not in requested:
            reasons[identifier] = "outside_requested_split"
        elif fraction < minimum:
            reasons[identifier] = "below_min_valid_fraction"
        else:
            groups.setdefault((part, building), []).append(row)
    selected = []
    for part in sorted(requested):
        buildings = sorted((building for candidate_part, building in groups if candidate_part == part), key=_digest)
        for building_index, building in enumerate(buildings):
            candidates = groups[part, building]
            if limit and building_index >= limit:
                reasons.update((row["dataset_crop_id"], "building_budget") for row in candidates)
                continue
            unique = {}
            for row in sorted(candidates, key=lambda row: row["dataset_crop_id"]):
                reference_hash = row["artifact_sha256"]["reference_rgb.png"]
                if reference_hash in unique:
                    reasons[row["dataset_crop_id"]] = "duplicate_reference_rgb"
                else:
                    unique[reference_hash] = row
            ranked = list(unique.values())
            used_views, used_references = set(), set()
            for _ in range(min(per_building, len(ranked))):
                def rank(row):
                    reference = row["source_image_sha256"]["reference"]
                    identity = [building, row["view_id"], reference,
                                row["artifact_sha256"]["reference_rgb.png"], row["bbox_canvas_xyxy"]]
                    return (row["view_id"] in used_views, reference in used_references, _digest(identity))
                choice = min(ranked, key=rank)
                ranked.remove(choice)
                selected.append(choice)
                used_views.add(choice["view_id"])
                used_references.add(choice["source_image_sha256"]["reference"])
            reasons.update((row["dataset_crop_id"], "crop_budget") for row in ranked)
    excluded = [{"dataset_crop_id": row["dataset_crop_id"], "building_id": row["building_id"],
                 "split": row["split"], "reason": reasons[row["dataset_crop_id"]]}
                for row in sorted(rows, key=lambda row: row["dataset_crop_id"]) if row["dataset_crop_id"] in reasons]
    return selected, excluded, {"splits": requested, "crops_per_building": per_building,
                                "max_buildings_per_split": limit, "min_valid_fraction": minimum}


def _checked_crop(root, record, row):
    directory = _inside(root, row["path"])
    metadata_relative = (directory / "crop.json").relative_to(root).as_posix()
    metadata_path, metadata_hash = _check_parent_file(root, record, metadata_relative)
    metadata = read_json(metadata_path)
    for key in ("crop_id", "building_id", "view_id", "split", "reference_id", "source_id", "valid_fraction",
                "bbox_canvas_xyxy", "tile_shape", "reference_to_crop", "source_to_crop", "source_image_sha256", "artifact_sha256"):
        if metadata.get(key) != row.get(key):
            raise ValueError(f"Indexed crop metadata disagrees with crop.json: {key}")
    checked = {metadata_relative: metadata_hash}
    for name, expected in row["artifact_sha256"].items():
        relative = (directory / name).relative_to(root).as_posix()
        path, digest = _check_parent_file(root, record, relative)
        if digest != expected:
            raise ValueError(f"Indexed crop image hash disagrees: {relative}")
        checked[relative] = digest
    for key in ("reference_rgb", "source_rgb", "geometric_overlap", "not_comparable"):
        if _inside(root, row[key]) != directory / (key + ".png"):
            raise ValueError(f"Indexed crop artifact path disagrees: {key}")
    rgb, opaque = load_rgb(directory / "reference_rgb.png")
    with Image.open(directory / "reference_support.png") as image:
        support = np.asarray(image) == 255
    with Image.open(directory / "geometric_overlap.png") as image:
        overlap = np.asarray(image) == 255
    if support.shape != rgb.shape[:2] or overlap.shape != support.shape or list(support.shape) != row["tile_shape"]:
        raise ValueError("Selected crop support/dimensions disagree")
    if not np.isclose(overlap.mean(), row["valid_fraction"]) or np.any(overlap & ~support) or np.any(support & ~opaque):
        raise ValueError("Selected crop geometric overlap/support disagrees")
    return rgb, support, checked


def _mask(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype != bool:
        raise ValueError(f"Renderer {name} must be a boolean mask on the native crop grid")
    return result


def _save(path, array):
    Image.fromarray(array).save(path)


def prepare_hypothesis_dataset(crop_run, config_path, out):
    """Build same-reference H0/H1 controls, never labels from the real later photo."""
    from .augmentations import apply_nuisance, render_state, validate_scenarios

    crop_run, config_path = Path(crop_run).expanduser().resolve(), Path(config_path).expanduser().resolve()
    parent_hash, config_hash = sha256(crop_run / "run.json"), sha256(config_path)
    parent, config = read_json(crop_run / "run.json"), read_json(config_path)
    if parent.get("kind") != "crop_dataset" or parent.get("status") not in {"completed", "completed_needs_review", "completed_with_issues"}:
        raise ValueError("Input must be a completed crop dataset")
    summary_path, summary_hash = _check_parent_file(crop_run, parent, "summary.json")
    parent_summary = read_json(summary_path)
    index_path, index_hash = _check_parent_file(crop_run, parent, parent_summary["index_path"])
    split_path, split_hash = _check_parent_file(crop_run, parent, "split.json")
    index, split = read_json(index_path), read_json(split_path)
    if split.get("mode") != "reviewed" or split.get("development_only") is not False or index.get("split") != split:
        raise ValueError("H0/H1 controls require the existing reviewed split snapshot")
    states = config.get("states", ["unchanged", "crack", "paint_patch", "self_paste"])
    if not isinstance(states, list) or not states or len(set(states)) != len(states) or any(state not in {"unchanged", "crack", "paint_patch", "self_paste"} for state in states):
        raise ValueError("states must contain unique supported procedural states")
    scenarios = validate_scenarios(config["scenarios"])
    selected, excluded, selection = _selection(index["crops"], split, config.get("selection", {}))
    if not selected:
        raise ValueError("No indexed crops meet the requested partitions and valid-fraction threshold")
    inputs = {"parent_run": parent_hash, "parent_summary": summary_hash, "parent_index": index_hash,
              "split": split_hash, "config": config_hash}
    out = new_directory(out)
    record = run_record("hypothesis_dataset", {"crop_run": str(crop_run), "config_path": str(config_path),
                        "input_sha256": inputs, "selection": selection, "states": states, "scenarios": scenarios,
                        "label_scope": LABEL_SCOPE, "selection_algorithm": "stable SHA ranking with view/reference diversity; no random split"})
    write_json(out / "run.json", record)
    try:
        shutil.copyfile(split_path, out / "split.json")
        shutil.copyfile(config_path, out / "config.json")
        write_json(out / "scenarios.json", scenarios)
        selection_document = {"schema_version": 1, "selection": selection, "parent_index_sha256": index_hash,
                              "selected": selected, "excluded_crops": excluded,
                              "parent_summary": parent_summary,
                              "parent_excluded_pairs": index.get("excluded_pairs", []),
                              "policy": "Frozen selection for this run; newly appended candidate crops may change a future SHA-ranked selection."}
        write_json(out / "selection.json", selection_document)
        cases, bases, selected_hashes = [], [], {}
        families = {state: [] for state in states}
        for row in selected:
            rgb, support, checked = _checked_crop(crop_run, parent, row)
            selected_hashes.update(checked)
            base_id = _digest([row["building_id"], row["view_id"], row["source_image_sha256"]["reference"],
                               row["artifact_sha256"]["reference_rgb.png"], row["bbox_canvas_xyxy"]])[:24]
            base_relative = Path("bases") / base_id
            base = new_directory(out / base_relative)
            original = _inside(crop_run, row["path"])
            for filename in ("reference_rgb.png", "reference_support.png"):
                shutil.copyfile(original / filename, base / filename)
            write_json(base / "geometry.json", {"reference_to_source": IDENTITY, "source_to_reference": IDENTITY,
                                               "scope": "same native crop grid; no geometric augmentation"})
            write_json(base / "metadata.json", {"parent_crop": row, "parent_artifact_sha256": checked,
                                               "reference_is_real_late_source": False, "label_scope": LABEL_SCOPE})
            bases.append({"base_id": base_id, "dataset_crop_id": row["dataset_crop_id"], "building_id": row["building_id"],
                          "split": row["split"], "view_id": row["view_id"], "path": base_relative.as_posix()})
            for state in states:
                rendered = render_state(rgb.copy(), support.copy(), state)
                state_rgb = np.asarray(rendered["rgb"])
                edit = _mask(rendered["edit_mask"], support.shape, "edit_mask")
                boundary = _mask(rendered["insertion_boundary"], support.shape, "insertion_boundary")
                if state_rgb.shape != rgb.shape or state_rgb.dtype != np.uint8 or np.any(edit & ~support):
                    raise ValueError("State renderer changed the native grid or edited unsupported pixels")
                if not np.array_equal(edit, np.any(state_rgb != rgb, axis=2) & support):
                    raise ValueError("State edit mask must label actual RGB edits before nuisance")
                if state in {"unchanged", "self_paste"} and (edit.any() or not np.array_equal(state_rgb, rgb)):
                    raise ValueError("H0/sham states must preserve reference RGB exactly")
                if state in {"crack", "paint_patch"} and not edit.any():
                    raise ValueError("An H1 state must plant at least one actual supported RGB edit")
                state_relative = base_relative / "states" / state
                state_dir = new_directory(out / state_relative)
                _save(state_dir / "true_edit_mask_reference.png", edit.astype(np.uint8) * 255)
                _save(state_dir / "insertion_boundary.png", boundary.astype(np.uint8) * 255)
                for scenario in scenarios:
                    nuisance = apply_nuisance(state_rgb.copy(), support.copy(), scenario)
                    source_rgb = np.asarray(nuisance["rgb"])
                    source_support = _mask(nuisance["support"], support.shape, "support")
                    visibility = _mask(nuisance["true_visibility"], support.shape, "true_visibility")
                    nuisance_mask = _mask(nuisance["nuisance_mask"], support.shape, "nuisance_mask")
                    if source_rgb.shape != rgb.shape or source_rgb.dtype != np.uint8 or np.any(source_support & ~support) or np.any(visibility & ~support):
                        raise ValueError("Nuisance renderer changed the native grid or invented support")
                    comparable = support & source_support & visibility
                    occluded = support & ~visibility
                    labels = np.full(support.shape, 255, np.uint8)
                    labels[comparable] = edit[comparable].astype(np.uint8)
                    case_id = base_id + "-" + state + "-" + scenario["id"]
                    case_relative = Path("cases") / case_id
                    directory = new_directory(out / case_relative)
                    arrays = {"source_rgb": source_rgb, "source_support": source_support,
                              "true_visibility": visibility, "comparable": comparable, "nuisance_mask": nuisance_mask,
                              "source_occlusion": occluded, "visible_edit_mask_reference": edit & comparable,
                              "labels_reference": labels}
                    if "actual_change_mask" in nuisance:
                        arrays["nuisance_actual_change"] = _mask(nuisance["actual_change_mask"], support.shape, "actual_change_mask")
                    paths = {}
                    for name, array in arrays.items():
                        _save(directory / (name + ".png"), array.astype(np.uint8) * 255 if array.dtype == bool else array)
                        paths[name] = (case_relative / (name + ".png")).as_posix()
                    if "intensity" in nuisance or (scenario["kind"] in {"shadow", "occlusion"} and "alpha" in nuisance):
                        intensity = np.asarray(nuisance.get("intensity", nuisance.get("alpha")))
                        if intensity.shape != support.shape or not np.isfinite(intensity).all():
                            raise ValueError("Nuisance intensity must be finite on the native crop grid")
                        np.save(directory / "nuisance_intensity.npy", intensity, allow_pickle=False)
                        paths["nuisance_intensity"] = (case_relative / "nuisance_intensity.npy").as_posix()
                    paths.update(reference_rgb=(base_relative / "reference_rgb.png").as_posix(),
                                 reference_support=(base_relative / "reference_support.png").as_posix(),
                                 true_edit_mask_reference=(state_relative / "true_edit_mask_reference.png").as_posix(),
                                 insertion_boundary=(state_relative / "insertion_boundary.png").as_posix(),
                                 geometry=(base_relative / "geometry.json").as_posix())
                    case = {"case_id": case_id, "base_id": base_id, "parent_dataset_crop_id": row["dataset_crop_id"],
                            **{key: row.get(key) for key in ("pair_id", "building_id", "view_id", "split", "reference_id", "source_id",
                                                            "reference_year", "source_year", "reference_file_name", "source_file_name",
                                                            "reference_gold_member", "source_gold_member", "source_image_sha256")},
                            "hypothesis": "H1" if edit.any() else "H0", "state": state, "scenario_id": scenario["id"],
                            "sham_self_paste": state == "self_paste", "independent_photo_sample": False,
                            "nuisance_kind": scenario["kind"], "scenario": scenario, "state_parameters": rendered["parameters"],
                            "nuisance_parameters": nuisance["parameters"], "synthetic_source_origin": "reference crop; real later RGB is unused",
                            "reference_to_source": IDENTITY, "source_to_reference": IDENTITY, "label_scope": LABEL_SCOPE,
                            "edit_pixel_count_before_nuisance": int(edit.sum()), "full_edit_pixel_count": int(edit.sum()),
                            "visible_edit_pixel_count": int((edit & comparable).sum()),
                            "retained_visible_edit_fraction": float((edit & comparable).sum() / edit.sum()) if edit.any() else None,
                            "exclude_from_visible_recall": bool(edit.any() and not (edit & comparable).any()),
                            "config_sha256": config_hash, "parent_index_sha256": index_hash,
                            "nuisance_mask_scope": "known applied footprint, independent of actual RGB quantization",
                            "comparable_fraction": float(comparable.mean()), "source_occlusion_fraction": float(occluded.mean()),
                            "artifact_sha256": {name: sha256(out / path) for name, path in paths.items()}, **paths}
                    cases.append(case)
                    families[state].append(case)
        for relative, digest in selected_hashes.items():
            if sha256(crop_run / relative) != digest:
                raise ValueError(f"Selected parent crop changed during export: {relative}")
        for path, digest in ((crop_run / "run.json", parent_hash), (summary_path, summary_hash), (index_path, index_hash),
                             (split_path, split_hash), (config_path, config_hash)):
            if sha256(path) != digest:
                raise ValueError(f"Input provenance changed during export: {path.name}")
        write_json(out / "index.json", {"schema_version": 1, "label_scope": LABEL_SCOPE, "input_sha256": inputs,
                                       "selection_path": "selection.json", "split": split, "bases": bases, "cases": cases,
                                       "parent_summary": parent_summary,
                                       "known_visibility_scope": "procedural source occlusion only; real occluders in the base are unknown"})
        fields = ["case_id", "base_id", "parent_dataset_crop_id", "pair_id", "building_id", "view_id", "split", "hypothesis",
                  "state", "scenario_id", "nuisance_kind", "reference_year", "source_year", "reference_file_name", "source_file_name",
                  "full_edit_pixel_count", "visible_edit_pixel_count", "retained_visible_edit_fraction", "exclude_from_visible_recall",
                  "comparable_fraction", "source_occlusion_fraction",
                  "reference_rgb", "source_rgb", "reference_support", "source_support", "true_edit_mask_reference", "true_visibility",
                  "comparable", "nuisance_mask", "source_occlusion", "visible_edit_mask_reference", "insertion_boundary", "labels_reference", "geometry"]
        with (out / "index.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cases)
        _galleries(out, families)
        summary = {"indexed_crop_count": len(index["crops"]), "selected_crop_count": len(selected),
                   "parent_summary": parent_summary,
                   "excluded_crop_count": len(excluded), "parent_excluded_pair_count": len(index.get("excluded_pairs", [])),
                   "selected_building_count": len({row["building_id"] for row in selected}), "case_count": len(cases),
                   "selected_crops_by_building": dict(Counter(row["building_id"] for row in selected)),
                   "selected_crops_by_split": dict(Counter(row["split"] for row in selected)),
                   "cases_by_split": dict(Counter(row["split"] for row in cases)),
                   "cases_by_building": dict(Counter(row["building_id"] for row in cases)),
                   "cases_by_hypothesis": dict(Counter(row["hypothesis"] for row in cases)),
                   "cases_by_state": dict(Counter(row["state"] for row in cases)),
                   "cases_by_nuisance": dict(Counter(row["nuisance_kind"] for row in cases)),
                   "cases_by_scenario": dict(Counter(row["scenario_id"] for row in cases)),
                   "index_path": "index.json", "csv_index_path": "index.csv", "selection_path": "selection.json",
                   "gallery_path": "gallery.html", "label_scope": LABEL_SCOPE,
                   "status": "completed_needs_review"}
        write_json(out / "summary.json", summary)
        (out / "summary.txt").write_text(
            f"Selected {len(selected)} of {len(index['crops'])} indexed crops from {summary['selected_building_count']} buildings.\n"
            f"Cases: {len(cases)}; hypotheses: {summary['cases_by_hypothesis']}; splits: {summary['cases_by_split']}.\n"
            f"Parent excluded pairs retained: {summary['parent_excluded_pair_count']}.\n"
            f"{LABEL_SCOPE}. H0 compares the reference crop with a copy plus one nuisance.\n"
            "Existing base damage remains unchanged. Known source occlusion is explicitly recorded; labels there are 255.\n"
            "Edit masks precede nuisance; shadows and exposure never become state-edit ground truth.\n", encoding="utf-8")
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise


def _galleries(out, families):
    pages = new_directory(out / "families")
    style = '<style>body{font:16px system-ui;margin:24px}td,th{padding:8px;vertical-align:top}img{max-width:160px}</style>'
    root = ['<!doctype html><meta charset="utf-8"><title>Procedural H0/H1 controls</title>', style,
            '<h1>Procedural H0/H1 controls</h1><p>' + html.escape(LABEL_SCOPE) + '</p>',
            '<p>Reference / synthetic source / true pre-nuisance edit. Known occlusion is ignored (255) in labels and separately recorded.</p>',
            '<table><tr><th>State family</th><th>Cases</th></tr>']
    for state, cases in families.items():
        root.append(f'<tr><td><a href="families/{state}.html">{state}</a></td><td>{len(cases)}</td></tr>')
        page = ['<!doctype html><meta charset="utf-8"><title>' + state + '</title>', style,
                '<a href="../gallery.html">All families</a><h1>' + state + '</h1><table>']
        for case in cases:
            label = html.escape(f"{case['building_id']} / {case['view_id']} / {case['split']} / {case['scenario_id']} / {case['hypothesis']}")
            page.append('<tr><td>' + label + '</td>' + ''.join(
                f'<td><img loading="lazy" src="../{html.escape(case[key], quote=True)}" alt="{key}"></td>'
                for key in ("reference_rgb", "source_rgb", "true_edit_mask_reference")) + '</tr>')
        page.append('</table>')
        (pages / (state + ".html")).write_text('\n'.join(page), encoding="utf-8")
    root.append('</table>')
    (out / "gallery.html").write_text('\n'.join(root), encoding="utf-8")
