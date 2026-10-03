"""Small sequential comparisons and shared native-data preparation."""
from __future__ import annotations

import csv
import html
import shutil
import subprocess
from pathlib import Path

from .io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .pipeline import run_pair, select_pair
from .preparation import validate_partitions


def batch_pairs(manifest, pairs=None, limit=3, split="all", allow_inferred_metadata=False):
    if "preparation" not in manifest:
        raise ValueError("Run 'facade_change prepare' on the existing manifest before a batch")
    validate_partitions(manifest["images"])
    if limit < 0:
        raise ValueError("limit must be >=0 (0 means all pairs)")
    if pairs:
        candidates = []
        for value in pairs:
            parts = value.split(":")
            if len(parts) != 2:
                raise ValueError(f"Expected --pair REFERENCE_ID:SOURCE_ID, got {value}")
            candidates.append({"reference_id": parts[0], "source_id": parts[1]})
    else:
        candidates = manifest["pairs"]
    selected, seen = [], set()
    for pair in candidates:
        reference, source = select_pair(manifest, pair["reference_id"], pair["source_id"], allow_inferred_metadata)
        partition = reference.get("split")
        if partition not in {"dev", "train", "val", "test"} or source.get("split") != partition:
            raise ValueError(f"Pair {pair} is excluded or crosses prepared partitions")
        if reference["sha256"] == source["sha256"]:
            raise ValueError(f"Pair {pair} contains identical image bytes")
        if split != "all" and partition != split:
            continue
        key = (str(reference["image_id"]), str(source["image_id"]))
        if key in seen:
            raise ValueError(f"Duplicate selected pair: {key}")
        seen.add(key)
        selected.append({"pair_id": "-".join(key), "reference_id": reference["image_id"],
                         "source_id": source["image_id"], "view_id": reference["view_id"],
                         "reference_year": reference["year"], "source_year": source["year"],
                         "building_id": reference.get("building_id"), "split": partition})
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No eligible temporal pairs for this selection")
    return selected


def comparison_counts(pairs, methods, rows):
    """Keep the same eligible-pair denominator, including failures and interruptions."""
    lookup = {(row['pair_id'], row['method']): row['status'] for row in rows}
    counts = {}
    for method in methods:
        statuses = [lookup.get((pair['pair_id'], method)) for pair in pairs]
        counts[method] = {status: statuses.count(status) for status in ('passed', 'rejected', 'failed')}
        counts[method].update(attempted=sum(status is not None for status in statuses),
                              not_attempted=statuses.count(None),
                              passed_fraction_of_eligible=statuses.count('passed') / len(pairs) if pairs else 0.)
    result = {'method_counts': counts}
    if 'sift' in methods and 'loftr' in methods:
        comparison = dict.fromkeys(('passed_both', 'sift_only', 'loftr_only', 'passed_neither',
                                    'not_fully_attempted'), 0)
        for pair in pairs:
            sift, loftr = (lookup.get((pair['pair_id'], method)) for method in ('sift', 'loftr'))
            if sift is None or loftr is None:
                comparison['not_fully_attempted'] += 1
            else:
                key = ('passed_both' if sift == loftr == 'passed' else 'sift_only' if sift == 'passed'
                       else 'loftr_only' if loftr == 'passed' else 'passed_neither')
                comparison[key] += 1
        result['sift_loftr_comparison'] = comparison
    return result


def write_batch_summary(out, pairs, methods, rows, derivatives):
    summary = {'eligible_pairs': len(pairs), 'requested_methods': methods,
               'attempted_runs': len(rows), 'failed_runs': sum(r['status'] == 'failed' for r in rows),
               'rejected_runs': sum(r['status'] == 'rejected' for r in rows),
               'passed_routing_gate': sum(r['status'] == 'passed' for r in rows),
               'derivative_failures': sum('error' in d for d in derivatives.values()),
               'pair_count_by_split': {name: sum(p['split'] == name for p in pairs)
                                       for name in sorted({p['split'] for p in pairs})},
               **comparison_counts(pairs, methods, rows),
               'interpretation': 'Gate success is not dense alignment or damage accuracy'}
    write_json(out / 'summary.json', summary)
    lines = [f"Pairs: {len(pairs)}; methods: {', '.join(methods)}; runs: {len(rows)}",
             f"Passed routing gate: {summary['passed_routing_gate']}; rejected: {summary['rejected_runs']}; "
             f"failed: {summary['failed_runs']}; derivative failures: {summary['derivative_failures']}"]
    for method, counts in summary['method_counts'].items():
        lines.append(f"{method}: passed {counts['passed']}/{len(pairs)}; rejected {counts['rejected']}; "
                     f"failed {counts['failed']}; not attempted {counts['not_attempted']}")
    if 'sift_loftr_comparison' in summary:
        counts = summary['sift_loftr_comparison']
        lines.append(f"SIFT/LoFTR: both passed {counts['passed_both']}; SIFT only {counts['sift_only']}; "
                     f"LoFTR only {counts['loftr_only']}; neither {counts['passed_neither']}; "
                     f"not fully attempted {counts['not_fully_attempted']}")
    lines.append('Inspect comparison.html; gate success requires visual review.')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def comparison_page(out, pairs, methods, rows, derivatives):
    lookup = {(row["pair_id"], row["method"]): row for row in rows}
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Facade comparison</title>',
             '<style>body{font:16px system-ui;margin:24px}td,th{padding:10px;vertical-align:top}'
             'td{border:1px solid #ccc}img{display:block;width:320px;max-width:100%}'
             '.bad{color:#a20}small{display:block;max-width:320px}a{display:block;margin:6px 0}</style>',
             '<h1>Facade alignment comparison</h1><p>Raw photographs, native reference scale. '
             'Overlay above; absolute RGB residual below. Gate results are routing heuristics; '
             'physical visibility and dense registration need visual review.</p>',
             '<p>' + '; '.join(f"{html.escape(method)}: {counts['passed']}/{len(pairs)} passed, "
                              f"{counts['rejected']} rejected, {counts['failed']} failed, "
                              f"{counts['not_attempted']} not attempted"
                              for method, counts in comparison_counts(pairs, methods, rows)['method_counts'].items())
             + '</p>',
             '<table><tr><th>Pair</th>' + ''.join(f'<th>{html.escape(method)}</th>' for method in methods) + '</tr>']
    for pair in pairs:
        label = f"{pair['view_id']}: {pair['reference_year']} → {pair['source_year']} ({pair['pair_id']})"
        parts.append('<tr><th>' + html.escape(label) + '<br>' + html.escape(pair['split']))
        derivative = derivatives.get(pair['pair_id'])
        if derivative:
            for kind in ('crops', 'controls'):
                if derivative.get(kind):
                    parts.append(f'<a href="{html.escape(derivative[kind], quote=True)}/gallery.html">{kind}</a>')
            if derivative.get('error'):
                parts.append('<small class="bad">' + html.escape(derivative['error']) + '</small>')
        parts.append('</th>')
        for method in methods:
            row = lookup.get((pair['pair_id'], method))
            if not row:
                parts.append('<td>Not run</td>')
                continue
            if row['status'] == 'failed':
                parts.append('<td class="bad">' + html.escape(row['error']) + '</td>')
                continue
            path = html.escape(row['path'], quote=True)
            metrics = row['diagnostics']
            parts.append(f'<td><a href="{path}/gallery.html">Open details</a>'
                         f'<small>Selected: {html.escape(metrics["selected_method"])}; '
                         f'inliers: {metrics["inliers"]}; gate: {row["status"]}</small>'
                         f'<img loading="lazy" src="{path}/overlay_preview.jpg">'
                         f'<img loading="lazy" src="{path}/residual_preview.jpg"></td>')
        parts.append('</tr>')
    parts.append('</table></html>')
    (out / 'comparison.html').write_text('\n'.join(parts), encoding='utf-8')


def run_batch(manifest_path, out, methods=None, pairs=None, limit=3, split="all",
              checkpoint="auto", device="cpu", max_side=1024, ransac_threshold=3., confidence=.4,
              seed=42, max_canvas_pixels=50_000_000, max_canvas_side=16000,
              allow_inferred_metadata=False, trust_checkpoint=False, download_weights=False,
              min_inliers=30, min_inlier_ratio=.2, min_hull_fraction=.1, min_overlap_fraction=.2,
              crops=False, tile_size=256, stride=128, min_valid_fraction=.8,
              controls=0, crop_method=None):
    methods = list(methods or ["sift", "loftr", "cascade"])
    if not methods or len(set(methods)) != len(methods) or any(m not in {"sift", "loftr", "cascade"} for m in methods):
        raise ValueError("Methods must be a nonempty unique selection of sift, loftr, cascade")
    if controls < 0 or (controls and not crops):
        raise ValueError("controls must be >=0 and requires --crops")
    if crops and (tile_size < 1 or stride < 1 or not 0 < min_valid_fraction <= 1):
        raise ValueError("Crops require positive tile size/stride and valid fraction in (0,1]")
    crop_method = crop_method or ("cascade" if "cascade" in methods else methods[0])
    if crop_method not in methods:
        raise ValueError("crop_method must be included in methods")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_path)
    selected = batch_pairs(manifest, pairs, limit, split, allow_inferred_metadata)
    options = {"checkpoint": checkpoint, "device": device, "max_side": max_side,
               "ransac_threshold": ransac_threshold, "confidence": confidence, "seed": seed,
               "max_canvas_pixels": max_canvas_pixels, "max_canvas_side": max_canvas_side,
               "allow_inferred_metadata": allow_inferred_metadata,
               "trust_checkpoint": trust_checkpoint, "download_weights": download_weights,
               "min_inliers": min_inliers, "min_inlier_ratio": min_inlier_ratio,
               "min_hull_fraction": min_hull_fraction, "min_overlap_fraction": min_overlap_fraction}
    config = {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "methods": methods, "pairs": selected, "alignment": options, "crops": crops,
              "tile_size": tile_size, "stride": stride, "min_valid_fraction": min_valid_fraction,
              "controls_per_pair": controls, "crop_method": crop_method}
    out = new_directory(out)
    record = run_record("alignment_and_data_batch", config)
    write_json(out / "run.json", record)
    rows, derivatives, matchers = [], {}, {}
    try:
        if device.startswith("cuda") and any(method != "sift" for method in methods):
            if shutil.which("nvidia-smi"):
                state = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
                record["gpu_state_before_models"] = state.stdout or state.stderr
            else:
                record["gpu_state_before_models"] = "nvidia-smi unavailable; matcher will check CUDA"
            write_json(out / "run.json", record)
        write_json(out / "selected_pairs.json", selected)
        for index, pair in enumerate(selected, 1):
            print(f"Pair {index}/{len(selected)}: {pair['view_id']} {pair['pair_id']}", flush=True)
            pair_results = {}
            for method in methods:
                relative = f"{method}/pair-{pair['pair_id']}"
                row = {"pair_id": pair["pair_id"], "method": method, "path": relative}
                try:
                    metrics = run_pair(manifest_path, pair["reference_id"], pair["source_id"], out / relative,
                                       method=method, matchers=matchers, **options)
                    row.update(status="passed" if metrics["quality_gate"]["passed"] else "rejected", diagnostics=metrics)
                    pair_results[method] = row
                    print(f"  {method}: {row['status']}; {metrics['inliers']} inliers", flush=True)
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"  {method}: {row['error']}", flush=True)
                rows.append(row)
                write_json(out / "results.json", rows)
                write_batch_summary(out, selected, methods, rows, derivatives)
                comparison_page(out, selected, methods, rows, derivatives)
            if crops:
                result = pair_results.get(crop_method)
                derivative = {"alignment_method": crop_method}
                derivatives[pair['pair_id']] = derivative
                if not result or result['status'] != 'passed':
                    derivative['error'] = f"No accepted {crop_method} alignment; crops not generated"
                else:
                    try:
                        from .derived import build_crops, controlled_examples
                        crop_path = f"pair-{pair['pair_id']}/crops"
                        summary = build_crops(out / result['path'], out / crop_path, tile_size=tile_size,
                                              stride=stride, min_valid_fraction=min_valid_fraction,
                                              split=pair['split'], group_id=pair.get('building_id') or 'unreviewed:' + pair['view_id'])
                        derivative.update(crops=crop_path, crop_summary=summary)
                        if summary['crop_count'] == 0:
                            raise ValueError("No crop met the requested valid-support fraction; inspect alignment/support")
                        if controls:
                            control_path = f"pair-{pair['pair_id']}/controls"
                            control_summary = controlled_examples(out / crop_path, out / control_path,
                                                                  seed=seed, max_crops=controls)
                            derivative.update(controls=control_path, control_summary=control_summary)
                    except Exception as exc:
                        derivative['error'] = f"{type(exc).__name__}: {exc}"
                        print(f"  derivatives: {derivative['error']}", flush=True)
                write_json(out / "derivatives.json", derivatives)
                comparison_page(out, selected, methods, rows, derivatives)
        summary = write_batch_summary(out, selected, methods, rows, derivatives)
        with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = ['pair_id', 'method', 'path', 'status', 'selected_method', 'matches', 'inliers', 'overlap_pixels', 'error']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                values = row.get('diagnostics', {})
                writer.writerow({key: row.get(key, values.get(key, '')) for key in fields})
        record['summary'] = summary
        issues = summary['failed_runs'] + summary['rejected_runs'] + summary['derivative_failures']
        finish_record(out, record, 'completed_with_issues' if issues else 'completed_needs_review')
        return summary
    except KeyboardInterrupt:
        write_batch_summary(out, selected, methods, rows, derivatives)
        comparison_page(out, selected, methods, rows, derivatives)
        finish_record(out, record, 'interrupted', 'Interrupted by user; completed child runs preserved')
        raise
    except Exception as exc:
        finish_record(out, record, 'failed', f'{type(exc).__name__}: {exc}')
        raise
