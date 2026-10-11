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
from .data import PreparedData, augment_crop, cutmix_crop, cutout_crop
from .metrics import aggregate_metrics, confusion_counts, write_reports
from .model import load_model, masked_multilabel_loss


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


def eligible_crops(data, image_id, cache):
    if image_id not in cache:
        _, labels, opaque = data.load_image(image_id)
        rows = []
        for row in data.crops_by_image[image_id]:
            x, y, w, h = (row[k] for k in ("x", "y", "width", "height"))
            if np.any((labels[y:y+h, x:x+w] != 0) & opaque[y:y+h, x:x+w]):
                rows.append(row)
        cache[image_id] = rows
    return cache[image_id]


def training_groups(data, ids, crops_per_image, rng, eligible, cutmix_prob, cutout_prob, counters):
    """Equal crop budget per image; consecutive crops reuse its decoded RGB.

    Donors come only from another TRAIN image. Their crop is loaded first so the
    receiver image stays cached while constructing its group of training crops.
    """
    for index in rng.permutation(len(ids)):
        image_id = ids[int(index)]
        donor = None
        mix = rng.random(crops_per_image) < cutmix_prob
        if mix.any() and len(ids) > 1:
            donor_index = int(rng.integers(len(ids) - 1))
            donor_id = ids[donor_index + (donor_index >= index)]
            donor_rows = eligible_crops(data, donor_id, eligible)
            if donor_rows:
                donor = augment_crop(*data.crop(donor_id, donor_rows[int(rng.integers(len(donor_rows)))]), rng)
        rows = eligible_crops(data, image_id, eligible)
        if not rows:
            counters["images_without_supervision"].add(image_id)
            continue
        selected = rng.choice(len(rows), size=crops_per_image, replace=len(rows) < crops_per_image)
        crops = []
        for i in range(crops_per_image):
            crop = augment_crop(*data.crop(image_id, rows[int(selected[i])]), rng)
            if mix[i] and donor is not None:
                crop = cutmix_crop(crop, donor, rng)
                counters["cutmix"] += 1
            if rng.random() < cutout_prob:
                crop = cutout_crop(*crop, rng)
                counters["cutout"] += 1
            if crop[2].any():
                crops.append(crop)
            else:
                counters["empty_augmented_crops"] += 1
        if crops:
            yield crops


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


def evaluate(model, data, ids, device, batch_size, amp=False, smoke=False, gallery_out=None):
    model.eval()
    records, previews, shown = [], [], set()
    for image_id in tqdm(ids, desc="DINO VAL", unit="фото"):
        image = data.images[image_id]
        all_rows = data.crops_by_image[image_id]
        ownership = ownership_masks(all_rows, image["width"], image["height"], data.crop_size)
        rows = all_rows
        counts = np.zeros((len(data.categories), 3), dtype=np.int64)
        candidates = []

        def process_batch(items):
            rgb, _, _ = _tensor_batch([item[1] for item in items], device)
            with torch.no_grad(), _autocast(device, amp):
                logits = model(rgb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite validation logits")
            predictions = (logits.float() >= 0).cpu().numpy()
            for (row, crop, valid), prediction in zip(items, predictions):
                counts[:] += confusion_counts(prediction, crop[1], valid)
                building = image["building_id"]
                if gallery_out is not None and building not in shown:
                    previews.append(_save_gallery_crop(gallery_out, len(previews), data, image_id, row, crop, prediction))
                    shown.add(building)

        supervised_crops = 0
        for row in rows:
            crop = data.crop(image_id, row)
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
        records.append({"image_id": image_id, "building_id": image["building_id"], "counts": counts.tolist()})
    report = aggregate_metrics(records, data.categories)
    report.update(smoke=smoke, threshold=.5, split="val", test_evaluated=False,
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


def train(dataset, out, methods_config=None, source_root=None, checkpoint=None, device="cuda:0",
          epochs=20, batch_size=4, crops_per_image=4, learning_rate=1e-3, seed=42,
          cutmix_prob=.5, cutout_prob=.25, no_amp=False, smoke=False, trust_checkpoint=False):
    if min(epochs, batch_size, crops_per_image) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size, crop budget and learning rate must be positive")
    if any(not 0 <= value <= 1 for value in (cutmix_prob, cutout_prob)):
        raise ValueError("Augmentation probabilities must be in [0, 1]")
    root, weights, config_hashes = dino_paths(methods_config, source_root, checkpoint)
    data = PreparedData(dataset)
    train_ids, val_ids = data.image_ids("train"), data.image_ids("val")
    if not train_ids or not val_ids:
        raise ValueError("Both TRAIN and VAL are required")
    if smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:2]
        epochs, crops_per_image = 1, min(2, crops_per_image)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Seeds and deterministic cuDNN recorded; do not claim cross-device bit identity.
    gpu = gpu_preflight(device)
    amp = str(device).startswith("cuda") and not no_amp
    config = {"dataset": str(data.dataset_path), "dataset_provenance": data.metadata,
              "source_root": root, "checkpoint": weights, "methods_config_sha256": config_hashes,
              "device": str(device), "epochs": epochs, "batch_size": batch_size,
              "crops_per_image": crops_per_image, "learning_rate": learning_rate, "seed": seed,
              "amp": amp, "smoke": smoke, "trust_checkpoint": trust_checkpoint,
              "categories": data.categories, "category_to_channel": data.metadata["category_to_channel"],
              "training_image_ids": train_ids, "validation_image_ids": val_ids,
              "sampler": "shuffled TRAIN images, equal random annotated-crop budget per image per epoch",
              "augmentations": {"outdoor": "exposure +/-0.5 stops, contrast 0.85–1.15, RGB gains 0.9–1.1, soft shadow, horizontal flip, affine +/-5deg/scale0.95–1.05/translation3%",
                                "mask_interpolation": "nearest-neighbor labels and support; bilinear RGB",
                                "blur": False, "cutmix_probability": cutmix_prob, "cutmix_beta_alpha": 1.,
                                "cutmix_donor": "different TRAIN image; copy RGB, multi-label targets and valid support",
                                "cutout_probability": cutout_prob, "cutout_side_fraction": [.15, .30],
                                "cutout": "mean RGB fill; occluded pixels ignored in loss"},
              "decision": "independent sigmoid >=0.5; checkpoint by equal-facade VAL mIoU",
              "supervision_assumption": "inside annotated union, unmarked categories are negative; outside union all categories unknown"}
    destination = new_directory(out)
    data.raster_cache_dir = destination / "mask-cache"
    record = run_record("dino_semantic_training", config)
    record["gpu_preflight"] = gpu
    record["environment"]["packages"]["pycocotools"] = importlib.metadata.version("pycocotools")
    write_json(destination / "run.json", record)
    write_json(destination / "config.json", config)
    try:
        print(f"TRAIN: {len(train_ids)} фото; VAL: {len(val_ids)}; классов: {len(data.categories)}; TEST не оценивается.", flush=True)
        print("DINOv2 заморожен. Обучается BN + линейная голова. Метрики только на размеченной поддержке.", flush=True)
        model, provenance = load_model(root, weights, len(data.categories), device, trust_checkpoint)
        record["model"] = provenance
        write_json(destination / "run.json", record)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp)
        best, best_epoch, history, eligible = -1., 0, [], {}
        counters = {"cutmix": 0, "cutout": 0, "empty_augmented_crops": 0, "images_without_supervision": set()}
        for epoch in range(1, epochs + 1):
            start = time.monotonic()
            model.train()
            rng = np.random.default_rng(seed + epoch)
            losses = []
            groups = training_groups(data, train_ids, crops_per_image, rng, eligible,
                                     cutmix_prob, cutout_prob, counters)
            progress = tqdm(groups, total=len(train_ids), desc=f"DINO TRAIN {epoch}/{epochs}", unit="фото")
            for crops in progress:
                for offset in range(0, len(crops), batch_size):
                    rgb, target, valid = _tensor_batch(crops[offset:offset+batch_size], device)
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
            if not losses:
                raise RuntimeError("No supervised training crops available")
            report = evaluate(model, data, val_ids, device, batch_size, amp, smoke)
            score = report["primary"]["miou"]
            if score is None:
                raise RuntimeError("VAL has no defined annotated-support IoU")
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_miou": score,
                            "val_recall": report["primary"]["recall"], "learning_rate": optimizer.param_groups[0]["lr"],
                            "seconds": time.monotonic() - start})
            write_json(destination / "history.json", history)
            write_json(destination / f"val-epoch-{epoch:03d}.json", report)
            _save_checkpoint(destination / "last.pth", model, optimizer, epoch, config, provenance, score)
            if score > best:
                best, best_epoch = score, epoch
                _save_checkpoint(destination / "best.pth", model, optimizer, epoch, config, provenance, score)
                write_json(destination / "best-val.json", report)
            scheduler.step()
            print(f"Эпоха {epoch}: VAL mIoU={score:.2%}; Recall={report['primary']['recall']:.2%}; лучший={best:.2%} (эпоха {best_epoch})", flush=True)
        saved = torch.load(destination / "best.pth", map_location="cpu", weights_only=True)
        model.head.load_state_dict(saved["head_state_dict"], strict=True)
        final = evaluate(model, data, val_ids, device, batch_size, amp, smoke, destination)
        write_json(destination / "metrics.json", final)
        write_reports(destination, final)
        counters["images_without_supervision"] = sorted(counters["images_without_supervision"])
        summary = {"status": "completed_smoke" if smoke else "completed_exploratory", "out": str(destination),
                   "best_epoch": best_epoch, "best_val_miou": best, "metrics": final["primary"],
                   "augmentation_counts": counters, "test_evaluated": False,
                   "scope": "semantic multilabel IoU on annotated union; no background or temporal-change claim",
                   "physical_disjointness_verified": data.metadata["leakage_check"]["physical_disjointness_verified"]}
        write_json(destination / "summary.json", summary)
        finish_record(destination, record, summary["status"])
        return summary
    except BaseException as error:
        finish_record(destination, record, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
        raise
