"""Export one crop dataset from existing, accepted homography pair runs."""
from __future__ import annotations

import csv
import html
import shutil
from collections import Counter
from pathlib import Path

from .derived import build_crops, controlled_examples
from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .pipeline import select_pair
from .preparation import validate_partitions


def _checked_pair_run(batch_run, batch_record, row, reference, source, manifest_hash):
    path = (batch_run / row["path"]).resolve()
    if batch_run not in path.parents:
        raise ValueError("Pair output path escapes the input batch")
    relative = (path / "run.json").relative_to(batch_run).as_posix()
    if batch_record.get("artifact_sha256", {}).get(relative) != sha256(path / "run.json"):
        raise ValueError("Pair run changed or its batch provenance hash is absent")
    record = read_json(path / "run.json")
    config = record.get("config", {})
    if config.get("manifest_sha256") != manifest_hash or config.get("method") != row["method"]:
        raise ValueError("Pair run manifest/method provenance disagrees with the batch")
    for role, expected in (("reference", reference), ("source", source)):
        actual = record.get("observations", {}).get(role, {})
        if str(actual.get("image_id")) != str(expected["image_id"]) or str(config.get(role + "_id")) != str(expected["image_id"]):
            raise ValueError("Pair run observation IDs disagree with the selected pair")
        for key in ("image_path", "sha256", "width", "height", "view_id", "building_id", "year", "split", "metadata_status"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"Pair run observation {role}.{key} disagrees with the prepared manifest")
    return path


def prepare_crop_dataset(batch_run, out, methods=("loftr", "sift"), tile_size=256,
                         stride=128, min_valid_fraction=.8, controls=0, seed=42):
    """Reuse accepted geometry and original RGB; preserve the existing gold split.

    ``controls`` is a total source-crop budget across this dataset. Procedural
    state/nuisance and sham examples are separate from real temporal labels.
    """
    methods = list(methods)
    if not methods or len(set(methods)) != len(methods) or any(name not in {"sift", "loftr", "cascade"} for name in methods):
        raise ValueError("Crop method priority must contain unique sift/loftr/cascade methods")
    if isinstance(controls, bool) or not isinstance(controls, int) or controls < 0:
        raise ValueError("controls must be a nonnegative total source-crop budget")
    if type(tile_size) is not int or tile_size < 8 or type(stride) is not int or not 0 < stride <= tile_size:
        raise ValueError("Crops require tile_size >=8 and 0 < stride <= tile_size")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    batch_run = Path(batch_run).expanduser().resolve()
    batch_hash = sha256(batch_run / "run.json")
    batch_record = read_json(batch_run / "run.json")
    if batch_record.get("status") not in {"completed_needs_review", "completed_with_issues", "interrupted"}:
        raise ValueError("Input batch must be completed or interrupted before exporting crops")
    manifest_path = Path(batch_record["config"]["manifest_path"]).expanduser()
    manifest_path = (manifest_path if manifest_path.is_absolute() else batch_run / manifest_path).resolve()
    split_path = manifest_path.parent / "split.json"
    input_paths = {"batch_run": batch_run / "run.json", "results": batch_run / "results.json",
                   "selected_pairs": batch_run / "selected_pairs.json", "manifest": manifest_path,
                   "split": split_path}
    hashes = {key: sha256(path) for key, path in input_paths.items()}
    hashes["batch_run"] = batch_hash
    if batch_record["config"].get("manifest_sha256") != hashes["manifest"]:
        raise ValueError("Prepared manifest changed since the alignment batch")
    for name in ("results", "selected_pairs"):
        if batch_record.get("artifact_sha256", {}).get(input_paths[name].name) != hashes[name]:
            raise ValueError(f"Batch {name} changed or its provenance hash is absent")
    manifest, split = read_json(manifest_path), read_json(split_path)
    validate_partitions(manifest["images"])
    if split.get("mode") != "reviewed" or split.get("development_only") is not False:
        raise ValueError("Crop datasets require an existing reviewed train/val/test split")
    for image in manifest["images"]:
        if image.get("split") in {"train", "val", "test"} and split["building_assignments"].get(image["building_id"]) != image["split"]:
            raise ValueError("Prepared manifest and split building assignments disagree")
    pairs, results = read_json(input_paths["selected_pairs"]), read_json(input_paths["results"])
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate selected pair IDs")
    lookup = {}
    for row in results:
        key = (row["pair_id"], row["method"])
        if key in lookup or row["pair_id"] not in pair_ids:
            raise ValueError("Duplicate or unselected batch result")
        lookup[key] = row
    out = new_directory(out)
    record = run_record("crop_dataset", {"batch_run": str(batch_run), "manifest_path": str(manifest_path),
                        "input_sha256": hashes, "method_priority": methods, "tile_size": tile_size,
                        "stride": stride, "min_valid_fraction": min_valid_fraction,
                        "controls_total_source_crop_budget": controls, "seed": seed,
                        "geometry_scope": "existing accepted homographies only; no rematching"})
    write_json(out / "run.json", record)
    try:
        shutil.copyfile(split_path, out / "split.json")
        crops, exported, excluded, control_rows, control_failures = [], [], [], [], []
        control_sources = 0
        derivative_failures = 0
        for pair in pairs:
            pair_id = pair["pair_id"]
            chosen = next((lookup[(pair_id, method)] for method in methods
                           if lookup.get((pair_id, method), {}).get("status") == "passed"), None)
            if chosen is None:
                excluded.append({"pair_id": pair_id, "reason": "no_passed_requested_method",
                                 "method_statuses": ";".join(f"{name}:{lookup.get((pair_id, name), {}).get('status', 'not_attempted')}"
                                                           for name in methods)})
                continue
            try:
                reference, source = select_pair(manifest, pair["reference_id"], pair["source_id"])
                if reference.get("split") not in {"train", "val", "test"} or source.get("split") != reference["split"]:
                    raise ValueError("Selected pair is outside the reviewed gold partitions")
                for key in ("view_id", "building_id", "split"):
                    if pair.get(key) != reference.get(key):
                        raise ValueError(f"Selected pair {key} disagrees with the prepared manifest")
                if pair.get("reference_year") != reference["year"] or pair.get("source_year") != source["year"]:
                    raise ValueError("Selected pair years disagree with the prepared manifest")
                if reference["sha256"] == source["sha256"]:
                    raise ValueError("Selected pair contains identical original image bytes")
                if pair_id != f"{reference['image_id']}-{source['image_id']}":
                    raise ValueError("Selected pair ID disagrees with its observations")
                pair_run = _checked_pair_run(batch_run, batch_record, chosen, reference, source, hashes["manifest"])
                crop_root = Path("pairs") / f"pair-{pair_id}" / "crops"
                crop_summary = build_crops(pair_run, out / crop_root, tile_size=tile_size, stride=stride,
                                           min_valid_fraction=min_valid_fraction, split=reference["split"],
                                           group_id=reference["building_id"])
                document = read_json(out / crop_root / "crops.json")
                if not document["crops"]:
                    raise ValueError("No crops met the requested geometric overlap fraction")
                for crop in document["crops"]:
                    path = crop_root / crop["path"]
                    crops.append({**crop, "pair_id": pair_id, "dataset_crop_id": pair_id + "/" + crop["crop_id"],
                                  "alignment_method": chosen["method"], "source_pair_path": chosen["path"],
                                  "path": path.as_posix(),
                                  **{name: (path / filename).as_posix() for name, filename in (
                                      ("reference_rgb", "reference_rgb.png"), ("source_rgb", "source_rgb.png"),
                                      ("geometric_overlap", "geometric_overlap.png"), ("not_comparable", "not_comparable.png"))},
                                  "reference_gold_member": reference.get("gold_member", False),
                                  "source_gold_member": source.get("gold_member", False),
                                  "reference_year": reference["year"], "source_year": source["year"],
                                  "reference_file_name": reference["file_name"], "source_file_name": source["file_name"],
                                  "original_reference_path": reference["image_path"], "original_source_path": source["image_path"]})
                exported.append({**pair, "method": chosen["method"],
                                 "crop_gallery": (crop_root / "gallery.html").as_posix(),
                                 "crop_count": len(document["crops"]), "rejected_low_overlap": crop_summary["rejected_low_overlap"]})
            except Exception as exc:
                derivative_failures += 1
                excluded.append({"pair_id": pair_id, "reason": f"{type(exc).__name__}: {exc}",
                                 "method_statuses": chosen["method"] + ":passed"})
                continue
            remaining = controls - control_sources
            if remaining:
                control_root = Path("controls") / f"pair-{pair_id}"
                try:
                    control_summary = controlled_examples(out / crop_root, out / control_root, seed=seed,
                                                          max_crops=min(remaining, len(document["crops"])))
                    examples = read_json(out / control_root / "examples.json")["examples"]
                    control_rows.extend({**example, "pair_id": pair_id,
                                         "dataset_example_id": pair_id + "/" + example["example_id"],
                                         "path": (control_root / example["path"]).as_posix()} for example in examples)
                    control_sources += control_summary["source_crop_count"]
                    exported[-1]["controls_gallery"] = (control_root / "gallery.html").as_posix()
                except Exception as exc:
                    control_failures.append({"pair_id": pair_id, "reason": f"{type(exc).__name__}: {exc}"})
        index_name = record["started_utc"][:10] + "_dataset_index.json"
        csv_name = record["started_utc"][:10] + "_dataset_index.csv"
        index = {"schema_version": 1, "image_selection": "existing_batch_selected_pairs_only",
                 "batch_run": str(batch_run), "input_sha256": hashes, "split": split,
                 "source_annotations": manifest.get("source", {}), "categories": manifest.get("categories", []),
                 "semantic_masks": {"rasterized": False}, "temporal_ground_truth": "not_created",
                 "pairs": exported, "excluded_pairs": excluded, "crops": crops,
                 "controls": {"label_scope": "procedural edit only; no real damage ground truth",
                              "source_crop_count": control_sources, "examples": control_rows, "failures": control_failures}}
        write_json(out / index_name, index)
        crop_fields = ["dataset_crop_id", "pair_id", "alignment_method", "split", "building_id", "view_id",
                       "reference_id", "source_id", "reference_year", "source_year", "reference_file_name", "source_file_name",
                       "valid_fraction", "reference_rgb", "source_rgb",
                       "geometric_overlap", "not_comparable"]
        for filename, rows, fields in ((csv_name, crops, crop_fields),
                                        ("excluded_pairs.csv", excluded, ["pair_id", "reason", "method_statuses"])):
            with (out / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        exported_by_id = {row["pair_id"]: row for row in exported}
        excluded_by_id = {row["pair_id"]: row for row in excluded}
        gallery = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Prepared RGB crops</title>',
                   '<style>body{font:16px system-ui;margin:24px}td,th{padding:10px;border:1px solid #ccc}</style>',
                   '<h1>Prepared RGB crops</h1><p>Existing reviewed partitions; geometric overlap only. '
                   'Controls label procedural edits, not real façade damage.</p>',
                   '<table><tr><th>Pair</th><th>Method</th><th>Split</th><th>Crops</th><th>Controls</th></tr>']
        for pair in pairs:
            label = html.escape(f"{pair['view_id']}: {pair['reference_year']} → {pair['source_year']} ({pair['pair_id']})")
            row = exported_by_id.get(pair["pair_id"])
            if row:
                controls_link = (f'<a href="{html.escape(row["controls_gallery"], quote=True)}">controls</a>'
                                 if row.get("controls_gallery") else "—")
                gallery.append(f'<tr><td>{label}</td><td>{html.escape(row["method"])}</td>'
                               f'<td>{html.escape(row["split"])}</td>'
                               f'<td><a href="{html.escape(row["crop_gallery"], quote=True)}">{row["crop_count"]} crops</a></td>'
                               f'<td>{controls_link}</td></tr>')
            else:
                gallery.append(f'<tr><td>{label}</td><td colspan="4">Excluded: '
                               + html.escape(excluded_by_id[pair["pair_id"]]["reason"]) + '</td></tr>')
        gallery.append('</table></html>')
        (out / "gallery.html").write_text('\n'.join(gallery), encoding="utf-8")
        for key, path in input_paths.items():
            if sha256(path) != hashes[key]:
                raise ValueError(f"Input {key} changed during crop dataset preparation")
        summary = {"selected_pair_count": len(pairs), "exported_pair_count": len(exported),
                   "excluded_pair_count": len(excluded), "crop_count": len(crops),
                   "pair_method_counts": dict(Counter(row["method"] for row in exported)),
                   "crop_splits": dict(Counter(row["split"] for row in crops)),
                   "control_source_crop_count": control_sources, "control_example_count": len(control_rows),
                   "control_failures": len(control_failures), "index_path": index_name, "csv_index_path": csv_name,
                   "derivative_failures": derivative_failures + len(control_failures), "gallery_path": "gallery.html",
                   "excluded_pairs_path": "excluded_pairs.csv", "split_snapshot_path": "split.json",
                   "status": "completed_with_issues" if excluded or control_failures else "completed_needs_review"}
        write_json(out / "summary.json", summary)
        (out / "summary.txt").write_text(
            f"Pairs selected: {len(pairs)}; exported: {len(exported)}; excluded: {len(excluded)}\n"
            f"Crops: {len(crops)}; split counts: {summary['crop_splits']}\n"
            f"Controls: {control_sources} source crops, {len(control_rows)} procedural examples\n"
            f"Index: {index_name}; exclusions: excluded_pairs.csv\n"
            "Existing reviewed split and original RGB were reused without rematching.\n"
            "Source-only tiles fail the overlap threshold; semantic masks and real temporal labels were not created.\n",
            encoding="utf-8")
        record["summary"] = summary
        finish_record(out, record, summary["status"])
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        finish_record(out, record, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
        raise
