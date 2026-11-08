"""Inspect saved synthetic bbox replies and make exactly two independent Qwen calls."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade_change.agentic.cli import TerminalProgress
from facade_change.agentic.config import load_config
from facade_change.agentic.coordinates import REGION_CROP, coordinate_frame, normalize_coordinate_system
from facade_change.agentic.models import DEFAULTS, ModelPool
from facade_change.agentic.reasoning import validate_budgets
from facade_change.agentic.runner import complete_run, new_run
from facade_change.geometry import transform_points
from facade_change.io import load_rgb, read_json, sha256, write_json


INSTRUCTION = "Сравни изображения A и B. Кратко опиши видимые различия. Если различий не видно, так и напиши. Ответь по-русски"
DISCLAIMER = "Common-support validation is NOT localization accuracy."


def remember(path, inputs, expected=None):
    path = Path(path).expanduser().resolve()
    actual = sha256(path)
    if (expected is not None and expected != actual) or inputs.get(str(path), actual) != actual:
        raise ValueError(f"Saved input changed: {path}")
    inputs[str(path)] = actual
    return path


def saved_json(path, inputs):
    return read_json(remember(path, inputs))


def rgb_artifact(record, inputs):
    artifacts = [a for a in record.get("artifacts", []) if a.get("role") == "rgb"]
    if len(artifacts) != 1:
        raise ValueError("Cited evidence has no unique RGB artifact; no other image was substituted")
    artifact = artifacts[0]
    if not artifact.get("sha256"):
        raise ValueError("Cited RGB has no recorded hash")
    return remember(artifact["path"], inputs, artifact["sha256"])


def frame_to_crop(frame, record):
    """Use the saved transforms and owners, including the validator's anchor fallback."""
    geometry = record.get("geometry", {})
    name = normalize_coordinate_system(frame.get("coordinate_system"))
    owner_key, owner = {
        REGION_CROP: ("region_id", record.get("region_id")),
        "observation_native": ("observation_id", record.get("observation_id")),
        "anchor_native": ("anchor_sha256", geometry.get("anchor_sha256")),
    }.get(name, (None, None))
    if owner is None:
        raise ValueError(f"No unambiguous owner for {name}")
    resolved = {"coordinate_system": name, owner_key: owner, **frame}
    if coordinate_frame(resolved) != coordinate_frame({"coordinate_system": name, owner_key: owner}):
        raise ValueError("Coordinate owner differs from cited RGB evidence")
    if name == REGION_CROP:
        matrix = np.eye(3)
    elif name == "observation_native":
        matrix = np.asarray(geometry.get("source_to_crop"), dtype=float)
    elif geometry.get("anchor_to_crop") is not None:
        matrix = np.asarray(geometry["anchor_to_crop"], dtype=float)
    elif geometry.get("crop_to_anchor") is not None:
        matrix = np.linalg.inv(np.asarray(geometry["crop_to_anchor"], dtype=float))
    else:
        bbox = geometry.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("No saved anchor-to-crop transform or region bbox")
        matrix = np.array([[1., 0., -bbox[0]], [0., 1., -bbox[1]], [0., 0., 1.]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Missing or invalid saved coordinate transform")
    return matrix


def bbox_on_rgb(localization, record, shape):
    bbox = localization.get("bbox_xyxy")
    if (not isinstance(bbox, list) or len(bbox) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox)):
        raise ValueError("Cannot display bbox without four finite coordinates")
    left, top, right, bottom = bbox
    if right < left or bottom < top:
        raise ValueError("Negative bbox width/height; coordinates were not reordered")
    rgb_frame = record.get("artifact_coordinates", {}).get("rgb")
    if not rgb_frame:
        raise ValueError("Cited RGB lacks an explicit artifact coordinate frame")
    transform = np.linalg.inv(frame_to_crop(rgb_frame, record)) @ frame_to_crop(localization, record)
    corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
    denominator = np.column_stack([corners, np.ones(4)]) @ transform[2]
    if not (np.all(denominator > 1e-10) or np.all(denominator < -1e-10)):
        raise ValueError("BBox crosses a projective horizon")
    points = transform_points(corners, transform)
    if np.any(points < -1e-7) or np.any(points > np.array([shape[1], shape[0]]) + 1e-7):
        raise ValueError("Mapped bbox lies outside the cited RGB; it was not clipped or expanded")
    return points, right == left or bottom == top


def draw_bbox(value, evidence, output, *, mode, origin, inputs):
    """Draw only on the cited RGB's grid; a cross is a marker, never a repaired box."""
    row = {"mode": mode, "origin": origin, "decision": value.get("decision"),
           "localization": value.get("localization"), "accuracy_note": DISCLAIMER}
    try:
        location = value.get("localization")
        if not isinstance(location, dict) or "bbox_xyxy" not in location:
            raise ValueError("No predicted bbox to display (localization may be null)")
        record = evidence[location["evidence_id"]]
        path = rgb_artifact(record, inputs)
        rgb, _ = load_rgb(path)
        points, zero_area = bbox_on_rgb(location, record, rgb.shape)
        support = location.get("support_validation", {})
        validation = ("invalid: zero-area bbox" if zero_area else
                      f"saved status={value.get('status', 'raw reply')}; common_fraction={support.get('common_fraction', 'not recorded')}")
        picture = Image.fromarray(rgb)
        draw = ImageDraw.Draw(picture)
        if zero_area:
            x, y = points.mean(axis=0)
            for color, width in (("white", 7), ("black", 5), ("red", 2)):
                draw.line([(x-8, y), (x+8, y)], fill=color, width=width)
                draw.line([(x, y-8), (x, y+8)], fill=color, width=width)
        else:
            draw.line([tuple(p) for p in [*points, points[0]]], fill="red", width=2)
        labels = [f"{mode} / {origin}; decision={value.get('decision')}",
                  f"bbox_xyxy={location['bbox_xyxy']}; frame={location.get('coordinate_system')}",
                  validation, DISCLAIMER]
        width = max(640, picture.width)
        lines = [part for label in labels for part in textwrap.wrap(label, width=max(40, width // 7))]
        header = 18 * len(lines) + 12
        canvas = Image.new("RGB", (width, picture.height + header), "white")
        canvas.paste(picture, (0, header))
        caption = ImageDraw.Draw(canvas)
        for index, line in enumerate(lines):
            caption.text((6, 6 + 18 * index), line, fill="black")
        canvas.save(output)
        row.update(status="rendered", validation=validation, rgb_path=str(path), rgb_sha256=sha256(path),
                   polygon_on_rgb=points.tolist(), rgb_offset_in_figure=[0, header],
                   visualization=str(output), visualization_sha256=sha256(output))
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        row.update(status="not_rendered", reason=f"{type(exc).__name__}: {exc}")
    return row


def visualize(analysis, out, inputs):
    out.mkdir()
    rows = []
    for mode in ("pair", "fixed_context", "agent"):
        directory = analysis / "cases/synthetic-2025/modes" / mode
        try:
            evidence = {r["evidence_id"]: r for r in saved_json(directory / "evidence.json", inputs)["evidence"]}
            if mode != "agent":
                value = saved_json(directory / "result.json", inputs)
                rows.append(draw_bbox(value, evidence, out / f"{mode}.png", mode=mode,
                                      origin="saved result", inputs=inputs))
                continue
            messages = saved_json(directory / "messages.json", inputs)
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                # Keep every original reply; do not repair invalid JSON or choose a preferred answer.
                text = "".join(block["text"] for block in message["content"] if block["type"] == "text")
                try:
                    value = json.loads(text)
                    if not isinstance(value, dict) or value.get("type") != "final":
                        continue
                    rows.append(draw_bbox(value, evidence, out / f"agent-message-{index}.png", mode=mode,
                                          origin=f"raw message {index}", inputs=inputs))
                except ValueError as exc:
                    rows.append({"mode": mode, "message_index": index, "status": "not_rendered", "reason": str(exc)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append({"mode": mode, "status": "not_rendered", "reason": f"{type(exc).__name__}: {exc}"})
    write_json(out / "index.json", {"accuracy_note": DISCLAIMER, "results": rows})
    return rows


def saved_pair(analysis, inputs):
    directory = analysis / "cases/synthetic-2020/modes/pair"
    case = saved_json(analysis / "cases/synthetic-2020/case.json", inputs)
    messages = saved_json(directory / "messages.json", inputs)
    state = saved_json(directory / "state.json", inputs)
    evidence = {r["evidence_id"]: r for r in saved_json(directory / "evidence.json", inputs)["evidence"]}
    initial = next(m for m in messages if m["role"] == "user")
    images = [b for b in initial["content"] if b["type"] == "image"]
    selected = []
    for role in ("reference", "source"):
        candidates = [b for b in images if evidence[b["evidence_id"]].get("observation_id") == case[role + "_observation_id"]
                      and evidence[b["evidence_id"]].get("kind") == "observation"]
        if len(candidates) != 1:
            raise ValueError(f"No unique actually supplied {role} RGB in saved pair transcript")
        block = candidates[0]
        record = evidence[block["evidence_id"]]
        if normalize_coordinate_system(record["coordinate_system"]) != REGION_CROP:
            raise ValueError("Diagnostic requires the saved RGB crops")
        path = rgb_artifact(record, inputs)
        if Path(block["path"]).resolve() != path or state["image_sha256"].get(str(path)) != sha256(path):
            raise ValueError("Saved RGB does not match the actual transcript image/hash")
        selected.append({"role": role, "path": str(path), "sha256": sha256(path),
                         "evidence_id": block["evidence_id"], "date": record.get("date")})
    return selected


def requests_for_pair(images, config):
    budgets = validate_budgets(config["budgets"])
    requests = []
    for name, other in (("reference-source", images[1]), ("reference-reference", images[0])):
        content = [{"type": "text", "text": INSTRUCTION}]
        for label, image in (("A", images[0]), ("B", other)):
            content.extend([{"type": "text", "text": label}, {"type": "image", "path": image["path"]}])
        requests.append((name, {"op": "generate", "messages": [{"role": "user", "content": content}],
            "max_input_tokens": min(budgets["max_input_tokens"], budgets["max_total_input_tokens"]),
            "max_new_tokens": min(budgets["max_output_tokens"], budgets["max_total_output_tokens"]),
            "max_images": budgets["max_images"],
            "timeout_seconds": min(config["worker_timeout"], budgets["max_seconds"])}))
    return requests


def run(source_run, config_path, out, *, pool_factory=ModelPool, progress=None):
    source_run, out = Path(source_run).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == source_run or source_run in out.parents:
        raise ValueError("Diagnostic output must be outside the immutable source run")
    analysis = source_run / "analysis"
    config = load_config(config_path)
    journal = new_run(out, {"configuration": config, "source_run": str(source_run),
                           "script_sha256": sha256(__file__)}, "qwen_saved_pair_diagnostic", progress=progress)
    inputs, queries = {}, []
    started = time.monotonic()
    try:
        remember(config_path, inputs)
        saved_config = saved_json(analysis / "config.json", inputs)
        if ({**DEFAULTS["qwen"], **saved_config["models"]["qwen"]}
                != {**DEFAULTS["qwen"], **config["models"]["qwen"]}
                or validate_budgets(saved_config["budgets"]) != validate_budgets(config["budgets"])):
            raise ValueError("Qwen options/budgets differ from the saved run; diagnostics must preserve them")
        visualizations = visualize(analysis, out / "visualizations", inputs)
        images = saved_pair(analysis, inputs)
        write_json(out / "inputs.json", {"files": inputs, "images": images, "selection": "synthetic-2020 / pair initial user message"})
        # One pool, no cache or close between queries. Only the existing worker's technical retries apply.
        with pool_factory({"qwen": config["models"]["qwen"]}, out / "worker_logs", gpu_mode=config["gpu_mode"],
                          timeout=config["worker_timeout"], cache_dir=None) as pool:
            pool.progress = journal.event
            write_json(out / "model_options.json", pool.models)
            for name, request in requests_for_pair(images, config):
                directory = out / name
                directory.mkdir()
                write_json(directory / "request.json", request)
                journal.event("mode_started", case_id="synthetic-2020", mode=name)
                call_started = time.monotonic()
                response = pool.call("qwen", copy.deepcopy(request))
                elapsed = time.monotonic() - call_started
                write_json(directory / "response.json", response)
                if isinstance(response.get("text"), str):
                    (directory / "response.txt").write_text(response["text"], encoding="utf-8")
                row = {"query": name, "status": response.get("status"), "elapsed_seconds": elapsed,
                       "usage": response.get("usage", {}), "generation": response.get("generation"),
                       "error": response.get("error")}
                queries.append(row)
                write_json(directory / "measurement.json", row)
                journal.event("mode_completed", **row)
        for path, expected in inputs.items():
            remember(path, inputs, expected)
        counts = {"completed": sum(q["status"] == "ok" for q in queries), "abstained": 0,
                  "error": sum(q["status"] != "ok" for q in queries)}
        summary = {"status": "completed" if not counts["error"] else "failed", "queries": queries,
                   "visualizations": visualizations, "outcome_counts": counts, "count_unit": "запросы",
                   "report_path": None, "elapsed_seconds": time.monotonic() - started,
                   "scope": "Two independent descriptive requests, not an accuracy evaluation", "accuracy_note": DISCLAIMER}
        write_json(out / "summary.json", summary)
        complete_run(journal, summary["status"], summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    progress = TerminalProgress()
    try:
        progress("command_started", task="qwen-diagnostic", out=args.out)
        result = run(args.run, args.config, args.out, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return int(result["status"] != "completed")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"qwen-diagnostic: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
