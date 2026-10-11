"""Small frozen-feature training loop with native-grid, facade-level validation."""
from __future__ import annotations

import contextlib
import html
import importlib.metadata
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..io import finish_record, new_directory, read_json, run_record, sha256, write_json
from .data import PreparedData
from .metrics import (aggregate_metrics, confusion_counts, group_confusion_counts,
                      loss_statistics, merge_loss_statistics, resolve_groups, write_reports)
from .model import load_model, masked_multilabel_loss
from .prepared import PreparedAugmentations


def dino_paths(methods_config=None, source_root=None, checkpoint=None):
    """Read only DINO paths from an existing methods/benchmark config."""
    seen, hashes = set(), {}

    def options(path):
        path = Path(path).expanduser().resolve()
        if path in seen:
            raise ValueError("Recursive methods_config reference")
        seen.add(path)
        hashes[str(path)] = sha256(path)
        value = read_json(path)
        result = {}
        if value.get("methods_config"):
            nested = Path(value["methods_config"]).expanduser()
            result.update(options(nested if nested.is_absolute() else path.parent / nested))
        for key in ("methods", "method_options"):
            mapping = value.get(key, {})
            if isinstance(mapping, dict):
                for name, item in mapping.get("dinov2", {}).items():
                    if name in {"source_root", "checkpoint_path", "dino_root", "dino_checkpoint"} and item:
                        candidate = Path(item).expanduser()
                        result[name] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
        return result

    configured = options(methods_config) if methods_config else {}
    root = source_root or configured.get("source_root") or configured.get("dino_root")
    weights = checkpoint or configured.get("checkpoint_path") or configured.get("dino_checkpoint")
    if not root or not weights:
        raise ValueError("Provide DINOv2 paths using --methods-config or both --source-root and --checkpoint")
    root, weights = Path(root).expanduser().resolve(), Path(weights).expanduser().resolve()
    if not (root / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError(f"Existing DINOv2 source/weights not found: {root}, {weights}")
    return str(root), str(weights), hashes


def gpu_preflight(device):
    target = torch.device(device)
    if target.type == "cpu":
        return {"device": "cpu"}
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {device}")
    torch.cuda.set_device(target)
    free, total = torch.cuda.mem_get_info(target)
    report = {"device": str(target), "name": torch.cuda.get_device_name(target),
              "free_bytes": free, "total_bytes": total}
    try:
        report["compute_processes"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["process_query_error"] = str(error)
    print(f"GPU: {report['name']}; свободно {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    if free < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB GPU memory available; free memory or choose another device. Other processes were not changed.")
    return report


def ownership_masks(rows, image_width, image_height, crop_size):
    """Partition overlapping grid crops at overlap midpoints, scoring each pixel once.

    No full-resolution C-channel prediction array is needed. This is a fixed
    centre-ownership protocol, not logit averaging or a prediction-dependent mask.
    """
    axes = []
    for position, extent, maximum in (("x", "width", image_width), ("y", "height", image_height)):
        values = {}
        for row in rows:
            key, size = row[position], row[extent]
            if key in values and values[key] != size:
                raise ValueError("Inconsistent crop grid extent")
            values[key] = size
        starts = sorted(values)
        if not starts or starts[0] != 0 or starts[-1] + values[starts[-1]] != maximum:
            raise ValueError("Prepared grid does not cover image boundaries")
        if any(a + values[a] < b for a, b in zip(starts, starts[1:])):
            raise ValueError("Prepared grid has uncovered gaps")
        edges = [0] + [(a + values[a] + b) // 2 for a, b in zip(starts, starts[1:])] + [maximum]
        axes.append({start: (edges[i], edges[i + 1]) for i, start in enumerate(starts)})
    if len(rows) != len(axes[0]) * len(axes[1]) or len({(r['x'], r['y']) for r in rows}) != len(rows):
        raise ValueError("Prepared crops must be a complete rectangular grid")
    result = {}
    for row in rows:
        x0, x1 = axes[0][row["x"]]
        y0, y1 = axes[1][row["y"]]
        mask = np.zeros((crop_size, crop_size), dtype=bool)
        mask[y0-row["y"]:y1-row["y"], x0-row["x"]:x1-row["x"]] = True
        result[row["crop_id"]] = mask
    return result


def _tensor_batch(crops, device):
    rgb = torch.from_numpy(np.stack([c[0] for c in crops])).permute(0, 3, 1, 2)
    target = torch.from_numpy(np.stack([c[1] for c in crops]))
    valid = torch.from_numpy(np.stack([c[2] for c in crops]))
    return (rgb.to(device=device, dtype=torch.float32) / 255.,
            target.to(device=device, dtype=torch.float32), valid.to(device))


def _autocast(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def fixed_batches(prepared, records, batch_size, rng):
    """Visit every supervised prepared sample once; shuffle order only."""
    order = rng.permutation(len(records))
    for offset in range(0, len(order), batch_size):
        yield [prepared.load(records[int(index)]) for index in order[offset:offset + batch_size]]


def _save_gallery_crop(out, number, data, image_id, row, crop, prediction):
    folder = out / "previews"
    folder.mkdir(exist_ok=True)
    rgb, target, valid = crop
    stem = f"val-{number:02d}"
    Image.fromarray(rgb).save(folder / (stem + "-rgb.jpg"))
    # One panel per class preserves overlaps instead of rendering an argmax map.
    paths = []
    for channel, category in enumerate(data.categories):
        panels = []
        for values, color in ((target[channel], [30, 220, 80]), (prediction[channel], [255, 80, 40])):
            panel = (rgb.astype(np.float32) * .35).astype(np.uint8)
            panel[valid] = rgb[valid]
            selected = values & valid
            panel[selected] = (.45 * rgb[selected] + .55 * np.asarray(color)).astype(np.uint8)
            panels.append(panel)
        name = f"{stem}-class-{category['id']}.jpg"
        Image.fromarray(np.concatenate(panels, axis=1)).save(folder / name)
        paths.append((category["name"], "previews/" + name))
    return {"image_id": image_id, "building_id": data.images[image_id]["building_id"],
            "crop_id": row["crop_id"], "rgb": "previews/" + stem + "-rgb.jpg", "classes": paths}


def _gallery(out, previews):
    sections = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>DINOv2 — VAL</title>",
                "<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;background:#f4f4f4}img{max-width:100%}section{background:white;padding:20px;margin:20px 0}summary{cursor:pointer;padding:8px}</style>",
                "<h1>DINOv2: сегментация VAL</h1><p>Фиксированный первый размеченный кроп каждого фасада. Это подборка, а не все ошибки. Слева разметка, справа прогноз. Затемнённые пиксели не участвуют в оценке. Классы показаны отдельно, пересечения сохранены.</p>"]
    for preview in previews:
        sections.append(f"<section><h2>{html.escape(str(preview['building_id']))}</h2><p>{html.escape(preview['crop_id'])}</p><img width='512' src='{preview['rgb']}' alt='RGB'>")
        for name, path in preview["classes"]:
            sections.append(f"<details><summary>{html.escape(name)}</summary><img src='{path}' alt='GT слева, прогноз справа'></details>")
        sections.append("</section>")
    sections.append("</html>")
    (out / "gallery.html").write_text("\n".join(sections), encoding="utf-8")


def evaluate(model, data, ids, device, batch_size, amp=False, smoke=False, gallery_out=None,
             split="val", groups=None, prepared=None):
    model.eval()
    groups = groups if groups is not None else resolve_groups(data.categories)
    if split not in {"train", "val"} or any(data.images[key]["split"] != split for key in ids):
        raise ValueError("Evaluation IDs must belong to the named TRAIN or VAL split")
    records, previews, shown = [], [], set()
    for image_id in tqdm(ids, desc=f"DINO {split.upper()} оценка", unit="фото"):
        image = data.images[image_id]
        all_rows = data.crops_by_image[image_id]
        ownership = ownership_masks(all_rows, image["width"], image["height"], data.crop_size)
        rows = all_rows
        counts = np.zeros((len(data.categories), 3), dtype=np.int64)
        group_counts = np.zeros((len(groups), 3), dtype=np.int64)
        losses = []
        candidates = []

        def process_batch(items):
            rgb, _, _ = _tensor_batch([item[1] for item in items], device)
            with torch.no_grad(), _autocast(device, amp):
                logits = model(rgb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite validation logits")
            probabilities = logits.float().sigmoid().cpu().numpy()
            raw_logits = logits.float().cpu().numpy()
            predictions = raw_logits >= 0
            for (row, crop, valid), prediction, probability, values in zip(items, predictions, probabilities, raw_logits):
                counts[:] += confusion_counts(prediction, crop[1], valid)
                group_counts[:] += group_confusion_counts(prediction, crop[1], valid, groups)
                losses.append(loss_statistics(probability, crop[1], valid, logits=values))
                building = image["building_id"]
                if gallery_out is not None and building not in shown:
                    previews.append(_save_gallery_crop(gallery_out, len(previews), data, image_id, row, crop, prediction))
                    shown.add(building)

        supervised_crops = 0
        for row in rows:
            crop = prepared.original_crop(image_id, row) if prepared is not None else data.crop(image_id, row)
            valid = crop[2] & ownership[row["crop_id"]]
            if not valid.any():
                continue
            supervised_crops += 1
            candidates.append((row, crop, valid))
            if len(candidates) == batch_size:
                process_batch(candidates)
                candidates = []
            if smoke and supervised_crops >= 4:
                break
        if candidates:
            process_batch(candidates)
        if not losses:
            empty = np.zeros((len(data.categories), 0, 0), dtype=np.float32)
            losses.append(loss_statistics(empty, empty.astype(bool), np.zeros((0, 0), dtype=bool)))
        records.append({"image_id": image_id, "building_id": image["building_id"], "counts": counts.tolist(),
                        "group_counts": group_counts.tolist(),
                        "loss_stats": merge_loss_statistics(losses)})
    report = aggregate_metrics(records, data.categories, groups=groups, allow_unknown_buildings=split == "train")
    report.update(smoke=smoke, threshold=.5, split=split, test_evaluated=False,
                  prediction_source="original unaugmented crops, model.eval() after epoch",
                  crop_overlap_accounting="each native pixel once; fixed midpoint ownership",
                  evaluation_scope="annotated opaque non-crowd union only; not full-image background evaluation",
                  image_records=records)
    if gallery_out is not None:
        _gallery(gallery_out, previews)
    return report


def _save_checkpoint(path, model, optimizer, epoch, config, provenance, score):
    value = {"schema_version": 1, "kind": "frozen_dinov2_semantic_head", "epoch": epoch,
             "head_state_dict": {key: tensor.detach().cpu() for key, tensor in model.head.state_dict().items()},
             "optimizer_state_dict": optimizer.state_dict(), "config": config,
             "backbone_provenance": provenance, "val_miou": score}
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)



def _format_metrics(values):
    parts = []
    for key, label in (("loss", "loss"), ("f1", "F1"), ("miou", "mIoU"),
                       ("precision", "Precision"), ("recall", "Recall")):
        value = values.get(key)
        text = "n/a" if value is None else (f"{value:.4f}" if key == "loss" else f"{value:.2%}")
        parts.append(f"{label}={text}")
    return "; ".join(parts)


def _print_epoch_report(split, report):
    print(f"{split.upper()} | {_format_metrics(report['primary'])}", flush=True)
    for group in report.get("per_group", []):
        print(f"  {group['name']}: {_format_metrics(group)}", flush=True)
    auxiliary = report.get("auxiliary_image_average", {})
    if auxiliary.get("image_count"):
        print(f"  TRAIN без установленного фасада, {auxiliary['image_count']} фото "
              f"(диагностика по изображениям): {_format_metrics(auxiliary)}", flush=True)


def train(dataset, out, augmentations=None, methods_config=None, source_root=None,
          checkpoint=None, device="cuda:0", epochs=20, batch_size=4,
          learning_rate=1e-3, seed=42, no_amp=False, smoke=False,
          trust_checkpoint=False, class_groups=None):
    if min(epochs, batch_size) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if not augmentations:
        raise ValueError("Training requires --augmentations from a completed --prepare-only run")
    root, weights, config_hashes = dino_paths(methods_config, source_root, checkpoint)
    data = PreparedData(dataset)
    prepared = PreparedAugmentations(augmentations, data)
    groups = resolve_groups(data.categories, class_groups)
    train_ids, val_ids = data.image_ids("train"), data.image_ids("val")
    if not train_ids or not val_ids:
        raise ValueError("Both TRAIN and VAL are required")
    if smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:2]
        epochs = 1
    selected_train = set(train_ids)
    all_samples = [item for item in prepared.records
                   if item["split"] == "train" and item["image_id"] in selected_train]
    samples = [item for item in all_samples if item["valid_pixels"] > 0]
    if smoke:
        # Only the run is partial; the reusable preparation always remains complete.
        samples = samples[:min(12, len(samples))]
    if not samples:
        raise ValueError("Prepared TRAIN has no supervised samples")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu = gpu_preflight(device)
    amp = str(device).startswith("cuda") and not no_amp
    config = {"dataset": str(data.dataset_path), "dataset_provenance": data.metadata,
              "source_root": root, "checkpoint": weights, "methods_config_sha256": config_hashes,
              "device": str(device), "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "seed": seed,
              "amp": amp, "smoke": smoke, "trust_checkpoint": trust_checkpoint,
              "categories": data.categories, "category_to_channel": data.metadata["category_to_channel"],
              "class_groups": groups,
              "class_groups_file_sha256": sha256(class_groups) if class_groups else None,
              "training_image_ids": train_ids, "validation_image_ids": val_ids,
              "sampler": "all fixed original+two-augmentation TRAIN samples once per epoch; shuffled order only",
              "training_samples": len(samples), "prepared_train_samples": len(all_samples),
              "skipped_unannotated_samples": sum(item["valid_pixels"] == 0 for item in all_samples),
              "augmentations": prepared.metadata,
              "epoch_evaluation": "original TRAIN and VAL; fixed native-pixel ownership; no augmentation",
              "unknown_train_aggregation": "separate equal-image diagnostic, never counted as known facades",
              "decision": "independent sigmoid >=0.5; checkpoint by equal-facade VAL mIoU",
              "supervision_assumption": "inside annotated union, unmarked categories are negative; outside union all categories unknown"}
    destination = new_directory(out)
    record = run_record("dino_semantic_training", config)
    record["gpu_preflight"] = gpu
    record["environment"]["packages"]["pycocotools"] = importlib.metadata.version("pycocotools")
    write_json(destination / "run.json", record)
    write_json(destination / "config.json", config)
    try:
        print(f"TRAIN: {len(train_ids)} фото, {len(samples)} сохранённых примеров/эпоха; "
              f"VAL: {len(val_ids)} фото; классов: {len(data.categories)}; TEST не оценивается.", flush=True)
        print("DINOv2 заморожен. Аугментации читаются с диска. TRAIN/VAL метрики — на оригиналах.", flush=True)
        model, provenance = load_model(root, weights, len(data.categories), device, trust_checkpoint)
        record["model"] = provenance
        write_json(destination / "run.json", record)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = (torch.amp.GradScaler("cuda", enabled=amp)
                  if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler(enabled=amp))
        best, best_epoch, history = -1., 0, []
        for epoch in range(1, epochs + 1):
            start = time.monotonic()
            model.train()
            losses = []
            batches = fixed_batches(prepared, samples, batch_size, np.random.default_rng(seed + epoch))
            progress = tqdm(batches, total=math.ceil(len(samples) / batch_size),
                            desc=f"DINO TRAIN {epoch}/{epochs}", unit="batch")
            for crops in progress:
                rgb, target, valid = _tensor_batch(crops, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp):
                    logits = model(rgb)
                    loss = masked_multilabel_loss(logits, target, valid)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; reduce LR or try --no-amp")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite training gradients; try --no-amp")
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                progress.set_postfix(loss=f"{np.mean(losses):.4f}")
            # Both reports use the same frozen end-of-epoch model state and originals.
            reports = {}
            for split, ids in (("train", train_ids), ("val", val_ids)):
                reports[split] = evaluate(model, data, ids, device, batch_size, amp, smoke,
                                          split=split, groups=groups, prepared=prepared)
                write_json(destination / f"{split}-epoch-{epoch:03d}.json", reports[split])
                write_reports(destination / f"epoch-{epoch:03d}" / split, reports[split])
                _print_epoch_report(split, reports[split])
            score = reports["val"]["primary"]["miou"]
            if score is None:
                raise RuntimeError("VAL has no defined annotated-support IoU")
            entry = {"epoch": epoch, "optimizer_loss": float(np.mean(losses)),
                     "optimizer_loss_aggregation": "mean augmented mini-batch objectives during weight updates",
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "seconds": time.monotonic() - start,
                     "train": reports["train"]["primary"], "val": reports["val"]["primary"]}
            for split, report in reports.items():
                for metric in ("miou", "f1", "precision", "recall", "loss"):
                    entry[f"{split}_{metric}"] = report["primary"][metric]
            history.append(entry)
            write_json(destination / "history.json", history)
            _save_checkpoint(destination / "last.pth", model, optimizer, epoch, config, provenance, score)
            if score > best:
                best, best_epoch = score, epoch
                _save_checkpoint(destination / "best.pth", model, optimizer, epoch, config, provenance, score)
                write_json(destination / "best-val.json", reports["val"])
                write_json(destination / "best-train.json", reports["train"])
            scheduler.step()
            print(f"Эпоха {epoch}: лучший VAL mIoU={best:.2%} (эпоха {best_epoch})", flush=True)
        saved = torch.load(destination / "best.pth", map_location="cpu", weights_only=True)
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        final = evaluate(model, data, val_ids, device, batch_size, amp, smoke,
                         gallery_out=destination, split="val", groups=groups, prepared=prepared)
        write_json(destination / "metrics.json", final)
        write_reports(destination, final)
        best_train = read_json(destination / "best-train.json")
        write_json(destination / "train-metrics.json", best_train)
        write_reports(destination / "train", best_train)
        summary = {"status": "completed_smoke" if smoke else "completed_exploratory", "out": str(destination),
                   "best_epoch": best_epoch, "best_val_miou": best, "metrics": final["primary"],
                   "train_metrics": best_train["primary"], "training_samples_per_epoch": len(samples),
                   "augmentations": str(prepared.path), "test_evaluated": False,
                   "scope": "semantic multilabel metrics on annotated union; no background or temporal-change claim",
                   "physical_disjointness_verified": data.metadata["leakage_check"]["physical_disjointness_verified"]}
        write_json(destination / "summary.json", summary)
        finish_record(destination, record, summary["status"])
        return summary
    except BaseException as error:
        finish_record(destination, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise
