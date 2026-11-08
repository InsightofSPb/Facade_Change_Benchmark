"""Local, inspectable reports from saved artifacts; no model calls (2026-11-08)."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from ..io import new_directory, read_json, sha256, write_json
from ..visualization import write_static_gallery


MODES = ("pair", "fixed_context", "agent")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _copy(root, value, out, sources):
    source = _path(root, value)
    digest = sha256(source)
    sources[str(source)] = digest
    relative = Path("assets") / (digest + source.suffix.lower())
    target = out / relative
    if not target.exists():
        shutil.copyfile(source, target)
    return relative.as_posix()


def _array(root, value):
    path = _path(root, value)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image)


def _overlay(rgb, values, support, binary=False):
    if values.shape != rgb.shape[:2] or support.shape != values.shape:
        raise ValueError("Overlay RGB, scores and support must share a grid")
    valid = support.astype(bool) & np.isfinite(values)
    result = rgb.copy()
    result[~valid] = (result[~valid].astype(float) * .25 + 96 * .75).astype(np.uint8)
    if binary:
        if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
            raise ValueError("A prediction overlay must contain a binary mask")
        active = valid & (values != 0)
        result[active] = (.5 * result[active] + .5 * np.array([255, 72, 45])).astype(np.uint8)
        return result, "binary prediction; gray = unsupported"
    finite = values[valid]
    if not finite.size:
        return result, "no supported finite scores"
    low, high = float(finite.min()), float(finite.max())
    display = np.zeros(values.shape, float)
    # Display scaling only. The original score map and its units remain saved.
    display[valid] = (values[valid] - low) / (high - low) if high > low else 0.
    color = np.stack((255 * display, 100 * (1 - display), 255 * (1 - display)), axis=-1)
    result[valid] = (.55 * result[valid] + .45 * color[valid]).astype(np.uint8)
    return result, f"display range [{low:.6g}, {high:.6g}]; per-map scale, not probability"


def _saved_json(path, sources, default=None):
    if not path.is_file():
        return default
    sources[str(path)] = sha256(path)
    return read_json(path)


def render_report(run_dir, out=None):
    """Create a new report folder with native RGB, saved maps and evidence links.

    The caller finishes its run record after rendering. A later rerender must
    use a new output directory, so the original report remains reproducible.
    """
    run = Path(run_dir).expanduser().resolve()
    output = new_directory(out if out is not None else run / "report")
    (output / "assets").mkdir()
    sources, sections, attachments, warnings = {}, [], [], []
    manifest = _saved_json(run / "input_manifest.json", sources, {})
    config = _saved_json(run / "config.json", sources, {})
    run_summary = _saved_json(run / "summary.json", sources, {})
    expected = manifest.get("cases", [])
    if not expected:
        expected = [{"case_id": path.parent.name} for path in sorted((run / "cases").glob("*/case.json"))]
    for selected in expected:
        identifier = selected["case_id"]
        if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("Unsafe case ID in report manifest")
        directory = run / "cases" / identifier
        case = _saved_json(directory / "case.json", sources)
        if case is None:
            sections.append({"title": identifier, "caption": "Case missing: preserved in cohort; no prediction is implied.", "panels": []})
            continue
        panels, source_rgb, observation_images = [], None, {}
        for image in case.get("images", []):
            try:
                relative = _copy(directory, image["path"], output, sources)
                role = image.get("role", "observation")
                panels.append({"label": role, "src": relative})
                with Image.open(output / relative) as opened:
                    rgb = np.asarray(opened.convert("RGB"))
                observation_images[image.get("observation_id", role)] = rgb
                if role in {"source", "current", "later"}:
                    source_rgb = rgb
            except (OSError, KeyError, ValueError) as error:
                warnings.append(f"{identifier}: image unavailable: {error}")
        sections.append({"title": f"{identifier} · {case.get('building_id', 'unknown building')} / {case.get('region_id', 'unknown region')}",
                         "caption": _json({key: case.get(key) for key in (
                             "reference_date", "source_date", "comparability", "limitations", "baseline",
                             "preparation_costs", "baseline_costs")}), "panels": panels})
        for index, method in enumerate(case.get("maps", [])):
            method_panels, caption = [], {key: method.get(key) for key in (
                "method", "status", "metadata", "threshold", "evidence_id", "error") if key in method}
            try:
                if source_rgb is not None and method.get("support_path"):
                    support = _array(directory, method["support_path"])
                    sources[str(_path(directory, method["support_path"]))] = sha256(_path(directory, method["support_path"]))
                    for field in ("scores_path", "prediction_path", "native_prediction_path"):
                        if not method.get(field):
                            continue
                        values = _array(directory, method[field])
                        raw = _copy(directory, method[field], output, sources)
                        overlay, scale = _overlay(source_rgb, values, support, binary=field != "scores_path")
                        filename = f"{sha256(directory / 'case.json')[:16]}-{index}-{field}.png"
                        Image.fromarray(overlay).save(output / "assets" / filename)
                        method_panels.append({"label": f"{field}: {scale}", "src": "assets/" + filename})
                        attachments.append((f"{identifier} / {method.get('method')} / {field}", raw))
            except (OSError, ValueError, KeyError) as error:
                caption["overlay_error"] = str(error)
                warnings.append(f"{identifier}: overlay {method.get('method')}: {error}")
            sections.append({"title": f"{identifier} · method {method.get('method', index)}", "caption": _json(caption), "panels": method_panels})
        for mode in MODES:
            mode_root = directory / "modes" / mode
            result = _saved_json(mode_root / "result.json", sources, {"status": "missing", "decision": "insufficient_data"})
            costs = _saved_json(mode_root / "costs.json", sources, {})
            evidence = _saved_json(mode_root / "evidence.json", sources, [])
            mode_panels = []
            items = evidence.get("evidence", evidence.get("items", [])) if isinstance(evidence, dict) else evidence
            if isinstance(items, dict):
                items = list(items.values())
            for item in items if isinstance(items, list) else []:
                # Only explicit preview images; never open arbitrary archive keys.
                preview = item.get("preview_path") or item.get("overlay_path")
                if not preview:
                    preview = next((artifact.get("path") for artifact in item.get("artifacts", [])
                                    if artifact.get("role") == "overlay"), None)
                if preview:
                    try:
                        mode_panels.append({"label": str(item.get("evidence_id", "tool output")),
                                            "src": _copy(mode_root, preview, output, sources)})
                    except (OSError, ValueError) as error:
                        warnings.append(f"{identifier}/{mode}: evidence preview: {error}")
            from .evaluation import cost_allocation
            sections.append({"title": f"{identifier} · Qwen {mode}",
                             "caption": _json({"result": result, "costs": costs,
                                               "cost_allocation": cost_allocation(case, mode, costs)}), "panels": mode_panels})
            for name in ("result.json", "costs.json", "evidence.json", "trace.jsonl", "messages.json"):
                if (mode_root / name).is_file():
                    attachments.append((f"{identifier} / {mode} / {name}", _copy(mode_root, name, output, sources)))
    notes = ["Saved outputs only: this report does not run models or change decisions.",
             "Geometric support is not physical visibility. Gray regions are not negative predictions.",
             "Each continuous map has its own stated display scale; this is not calibrated confidence.",
             "Missing, abstained and failed cases remain visible. Pixel masks are never inferred from VLM text.",
             "Preparation is shared once per case. Mode costs are incremental; context/agent allocations include preparation, while pair allocation includes only separately measured geometry. Alternative allocations must not be added together.",
             "Costs reflect the recorded cache and worker state. Original cached source usage is retained but not charged again; no cold-start timing is implied.",
             "Actual run summary: " + _json(run_summary),
             "Availability policy: " + _json(manifest.get("availability_policy", {})),
             "Modes requested: " + _json(config.get("modes", []))]
    gallery = write_static_gallery(output, "Facade change · evidence report", sections, notes=notes)
    links = ''.join('<li><a href="' + html.escape(quote(path, safe="/"), quote=True) + '">'
                    + html.escape(label) + '</a></li>' for label, path in attachments)
    extra = '<section><h2>Saved evidence and traces</h2><ul>' + links + '</ul></section>'
    if warnings:
        extra += '<section><h2>Report warnings</h2><pre>' + html.escape('\n'.join(warnings)) + '</pre></section>'
    path = output / "report.html"
    path.write_text(gallery.read_text(encoding="utf-8").replace("</html>", extra + "</html>"), encoding="utf-8")
    for source, expected in sources.items():
        if sha256(source) != expected:
            raise ValueError(f"Report source changed during rendering: {source}")
    write_json(output / "report_sources.json", {"schema_version": 1, "run": str(run),
                                              "source_sha256": sources, "warnings": warnings})
    return path


def render_comparison(out, result):
    """Small static companion for comparison.json, without filtering failures."""
    out = Path(out)
    columns = ("run_id", "mode", "split", "cases", "facades", "recall_macro", "precision_macro",
               "fpr_macro", "coverage_macro", "abstention_rate_macro", "error_rate_macro", "pixel_iou_macro",
               "region_box_iou_macro", "seconds", "incremental_seconds", "shared_preparation_seconds",
               "allocated_with_shared_seconds", "pair_geometry_plus_increment_seconds",
               "model_calls", "tool_calls", "input_tokens", "output_tokens",
               "images", "model_loads", "cache_hits", "peak_reserved_bytes")
    header = ''.join('<th>' + html.escape(key) + '</th>' for key in columns)
    rows = []
    for row in result.get("summary", []):
        values = ["unknown" if row.get(key) is None else format(row[key], ".5g") if isinstance(row[key], float)
                  else str(row[key]) for key in columns]
        rows.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in values) + '</tr>')
    notes = ''.join('<p>' + html.escape(note) + '</p>' for note in result.get("interpretation", []))
    notes += '<pre>' + html.escape(_json({"source_runs": result.get("source_runs", [])})) + '</pre>'
    content = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Facade comparison</title>'
               '<style>body{font:15px system-ui;padding:24px;color:#ddd;background:#15191e}'
               'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #52606c;text-align:left}'
               'main{overflow:auto}a{color:#9bcaff}p{max-width:1100px}</style>'
               '<h1>Saved run comparison</h1>' + notes + '<main><table><thead><tr>' + header
               + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></main>'
               '<p><a href="comparison.json">Full metrics and cases</a> · '
               '<a href="comparison.csv">Summary CSV</a> · <a href="cases.csv">All case statuses</a> · '
               '<a href="facades.csv">Building-level metrics</a></p></html>')
    path = out / "report.html"
    path.write_text(content, encoding="utf-8")
    return path
