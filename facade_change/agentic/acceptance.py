"""Small procedural fixtures and explicit local GPU acceptance, never benchmarks."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..io import new_directory, read_json, write_json
from .runner import complete_run, new_run, run_cases


def fixture(out):
    """Three dated RGB observations. No temporal labels are given to inference."""
    out = new_directory(out)
    rng = np.random.default_rng(20261108)
    base = Image.fromarray(rng.integers(40, 190, (256, 320, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(base)
    for x in range(20, 300, 40):
        for y in range(15, 240, 45):
            draw.rectangle((x, y, x + 22, y + 26), outline=(230, 230, 210), width=3)
    images = []
    for index, year in enumerate((2010, 2020, 2025)):
        image = base.copy()
        if index == 1:
            ImageDraw.Draw(image).rectangle((130, 95, 175, 130), fill=(230, 35, 35))
        if index == 2:
            ImageDraw.Draw(image).line((135, 96, 152, 128, 148, 150), fill=(15, 15, 15), width=4)
        path = out / f"synthetic-{year}.png"
        image.save(path)
        images.append({"path": str(path), "date": str(year), "date_status": "confirmed",
                       "available_at": str(year), "image_id": f"synthetic-{year}"})
    pairs = [{"case_id": f"synthetic-{source['date']}", "building_id": "synthetic-building",
              "view_id": "synthetic-view", "split": "train", "reference": images[0],
              "source": source, "bbox_xyxy": [0, 0, 320, 256]} for source in images[1:]]
    write_json(out / "pairs.json", {"purpose": "Code fixture only, not facade research evidence", "cases": pairs})
    return pairs


def run_smoke(config, out, kind="cpu"):
    out = new_directory(out)
    pairs = fixture(out / "fixture")
    config = copy.deepcopy(config)
    config.update(store_root=str(out / "store"), knowledge_root=str(out / "knowledge"),
                  cache_root=str(out / "cache"), methods=["ssim", "lzma_abs"], method_options={},
                  fusion=None, thresholds={"ssim": .15}, diagnostic_baseline="ssim",
                  alignment={"method": "sift", "device": "cpu", "max_side": 512})
    if kind == "cpu":
        config["modes"] = []
        result = run_cases(config, pairs, out / "analysis")
        result["scope"] = "Real CPU SIFT/SSIM/LZMA + store/history/report; Qwen/SAM3 not executed"
    elif kind == "models":
        config["modes"] = ["pair", "fixed_context", "agent"]
        result = run_cases(config, pairs, out / "analysis")
        result["scope"] = "Real configured Qwen on procedural images, not accuracy evaluation"
    else:
        result = model_acceptance(config, pairs[0], out / "acceptance", kind)
    write_json(out / "verification.json", result)
    return result


def model_acceptance(config, pair, out, kind):
    from .models import ModelPool
    if kind not in {"handoff", "recovery", "context"}:
        raise ValueError("Unknown acceptance kind")
    journal = new_run(out, {"config": config, "kind": kind, "pair": pair}, "agentic_model_acceptance")
    pool = ModelPool(config["models"], journal.out / "worker_logs",
                     gpu_mode="sequential" if kind == "recovery" else config["gpu_mode"],
                     timeout=config["worker_timeout"], cache_dir=None)
    started = time.monotonic()
    responses = []
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Compare these dated synthetic facade crops (2010, 2020). Describe only visible differences. Do not infer causes."},
        {"type": "image", "path": pair["reference"]["path"]},
        {"type": "image", "path": pair["source"]["path"]}]}]
    budgets = config["budgets"]
    def invoke(role, payload):
        write_json(journal.out / "messages.json", messages)
        journal.event("acceptance_call", role=role, op=payload["op"])
        value = pool.call(role, payload)
        responses.append({"role": role, "op": payload["op"], "response": value})
        write_json(journal.out / "responses.json", responses)
        if value.get("status") == "error":
            raise RuntimeError(f"{role}: {value.get('error')}")
        return value
    options = {"max_input_tokens": budgets["max_input_tokens"], "max_images": budgets["max_images"],
               "max_new_tokens": budgets["max_output_tokens"]}
    try:
        context_measurement = None
        if kind == "context":
            content = messages[0]["content"]
            while sum(x["type"] == "image" for x in content) < budgets["max_images"]:
                content.append({"type": "image", "path": pair["source"]["path"]})
            original = content[0]["text"]
            low, high = 0, budgets["max_input_tokens"] * 2
            # Measure the real processor's tokens. No hand-written tokenizer estimate.
            for _ in range(16):
                if low >= high:
                    break
                middle = (low + high + 1) // 2
                content[0]["text"] = original + " Background context." * middle
                measured = pool.call("qwen", {"op": "measure", "messages": messages, **options})
                responses.append({"role": "qwen", "op": "measure", "phase": "context_sizing",
                                  "text_repetitions": middle, "response": measured})
                journal.event("context_size_probe", text_repetitions=middle,
                              status=measured.get("status"), usage=measured.get("usage", {}),
                              input_tokens=measured.get("input_tokens"))
                write_json(journal.out / "responses.json", responses)
                count = measured.get("input_tokens", measured.get("usage", {}).get("input_tokens"))
                if measured.get("status") == "ok" and isinstance(count, int) and count <= budgets["max_input_tokens"]:
                    low, context_measurement = middle, measured
                else:
                    high = middle - 1
            content[0]["text"] = original + " Background context." * low
            context_measurement = invoke("qwen", {"op": "measure", "messages": messages, **options})
        first = invoke("qwen", {"op": "generate", "messages": messages, **options})
        if kind != "context":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": first.get("text", "")} ]})
            # Persist state before ownership transfer; reloading requires no KV-cache migration.
            write_json(journal.out / "messages.json", messages)
            segmented = invoke("sam3", {"op": "segment", "image_path": pair["source"]["path"],
                "text": "red patch", "out": str(journal.out / "sam3"),
                "geometry": {"coordinate_system": "source_native", "native_shape": [256, 320]}})
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "The segmentation tool completed with status " + segmented["status"]
                 + f"; masks={len(segmented.get('scores', []))}; scores={segmented.get('scores', [])}. "
                 + "Reinspect the crop and state limitations; this semantic segmentation is not proof of temporal change."},
                {"type": "image", "path": pair["source"]["path"]}]})
            write_json(journal.out / "messages.json", messages)
            if kind == "recovery":
                # Restore the complete CPU transcript from disk, then reconstruct
                # multimodal inputs in a freshly loaded Qwen worker.
                messages = read_json(journal.out / "messages.json")
                journal.event("cpu_transcript_restored", path=str(journal.out / "messages.json"),
                              messages=len(messages), kv_cache_transfer=False)
            invoke("qwen", {"op": "generate", "messages": messages, **options})
        result = {"status": "completed", "kind": kind, "elapsed_seconds": time.monotonic() - started,
                  "responses": responses, "context_measurement": context_measurement,
                  "requested_limits": options,
                  "scope": "Only measured input/output lengths are verified; early EOS does not verify a full output-token budget"}
        write_json(journal.out / "summary.json", result)
        complete_run(journal, "completed")
        return result
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        pool.close()
        journal.close()
