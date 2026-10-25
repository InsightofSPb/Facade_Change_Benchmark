"""Multi-label semantic metrics on annotated pixels, with equal facade weights.

The caller must evaluate each native image pixel once. Overlapping training or
inference crops are not statistical units and must be merged before this module.
"""
from __future__ import annotations

import csv
import colorsys
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_CLASS_METRICS = ("iou", "f1", "precision", "recall", "loss", "bce", "dice")
_SUMMARY_METRICS = ("miou", "f1", "precision", "recall", "loss", "bce", "dice")
_LOSS_ARRAYS = ("bce_sum", "probability_sum", "target_sum", "intersection")


def confusion_counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return C x 3 int64 counts (TP, FP, FN); overlaps remain multi-label."""
    prediction, target, valid = map(np.asarray, (prediction, target, valid))
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have identical C,H,W shapes")
    if valid.shape not in (prediction.shape[1:], prediction.shape):
        raise ValueError("valid must have shape H,W or C,H,W")
    if any(array.dtype != np.bool_ for array in (prediction, target, valid)):
        raise ValueError("prediction, target and valid must be boolean arrays")
    support = valid[None, :, :] if valid.ndim == 2 else valid
    axes = (1, 2)
    return np.stack((
        np.sum(prediction & target & support, axis=axes, dtype=np.int64),
        np.sum(prediction & ~target & support, axis=axes, dtype=np.int64),
        np.sum(~prediction & target & support, axis=axes, dtype=np.int64),
    ), axis=1)


def resolve_groups(categories: list[dict], path: str | Path | None = None) -> list[dict]:
    """Resolve operational reporting groups to model channels, never reorder them.

    The shipped groups are a reporting convenience, not a new defect ontology.
    Defaults intersect the available categories and report unmatched categories
    individually. An explicit configuration rejects unknown category names.
    """
    explicit = path is not None
    source = Path(path) if explicit else Path(__file__).resolve().parents[2] / "configs/semantic_class_groups.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("groups"), list):
        raise ValueError("class groups require schema_version=1 and a groups list")
    names = [category["name"] for category in categories]
    if len(names) != len(set(names)):
        raise ValueError("category names must be unique for class groups")
    by_name = {name: channel for channel, name in enumerate(names)}
    seen, used, result = set(), set(), []
    for group in config["groups"]:
        name, members = group.get("name"), group.get("categories")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("group names must be nonempty and unique")
        if (not isinstance(members, list) or not members
                or any(not isinstance(item, str) for item in members)
                or len(members) != len(set(members))):
            raise ValueError(f"group {name}: categories must be a nonempty list without duplicates")
        seen.add(name)
        unknown = sorted(set(members) - set(by_name))
        if explicit and unknown:
            raise ValueError(f"group {name}: unknown categories: {unknown}")
        channels = [by_name[member] for member in members if member in by_name]
        if not channels:
            continue
        used.update(channels)
        result.append({"name": name, "channels": channels,
                       "category_ids": [categories[c]["id"] for c in channels],
                       "category_names": [names[c] for c in channels]})
    if explicit and set(by_name.values()) - used:
        missing = [names[c] for c in range(len(names)) if c not in used]
        raise ValueError(f"class group configuration does not cover categories: {missing}")
    if not explicit:
        for channel, category in enumerate(categories):
            if channel not in used:
                result.append({"name": f"class:{category['name']}", "channels": [channel],
                               "category_ids": [category["id"]],
                               "category_names": [category["name"]]})
    return result


def group_confusion_counts(prediction, target, valid, groups) -> np.ndarray:
    """OR members before counting: overlapping labels count once per group."""
    # Validate even for an empty reporting-group list.
    confusion_counts(prediction, target, valid)
    if not groups:
        return np.zeros((0, 3), dtype=np.int64)
    group_prediction, group_target = [], []
    for group in groups:
        channels = group["channels"]
        if not channels or any(c < 0 or c >= len(target) for c in channels):
            raise ValueError("invalid group channels")
        group_prediction.append(np.any(prediction[channels], axis=0))
        group_target.append(np.any(target[channels], axis=0))
    return confusion_counts(np.stack(group_prediction), np.stack(group_target), valid)


def loss_statistics(probabilities, target, valid, logits=None) -> dict:
    """Additive masked BCE/soft-Dice statistics on owned native pixels.

    Pass logits for numerically stable BCE. Statistics can be merged across
    tiles before computing Dice, avoiding dependence on evaluation batch size.
    """
    probabilities, target, valid = map(np.asarray, (probabilities, target, valid))
    if probabilities.ndim != 3 or target.shape != probabilities.shape or valid.shape != target.shape[1:]:
        raise ValueError("loss inputs require C,H,W probabilities/target and H,W valid")
    if target.dtype != np.bool_ or valid.dtype != np.bool_:
        raise ValueError("loss target and valid must be boolean")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite and between zero and one")
    probability = probabilities[:, valid].astype(np.float64)
    truth = target[:, valid].astype(np.float64)
    if logits is not None:
        logits = np.asarray(logits)
        if logits.shape != probabilities.shape or not np.all(np.isfinite(logits)):
            raise ValueError("logits must be finite and have the same shape as probabilities")
        scores = logits[:, valid].astype(np.float64)
        bce = np.logaddexp(0., scores) - scores * truth
    else:
        clipped = np.clip(probability, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
        bce = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
    return {"bce_sum": bce.sum(axis=1).tolist(),
            "probability_sum": probability.sum(axis=1).tolist(),
            "target_sum": truth.sum(axis=1).tolist(),
            "intersection": (probability * truth).sum(axis=1).tolist(),
            "valid_pixels": int(valid.sum())}


def merge_loss_statistics(records: Iterable[dict | None]) -> dict | None:
    """Merge additive statistics; absent legacy statistics remain undefined."""
    result = None
    for record in records:
        if record is None:
            continue
        arrays = {key: np.asarray(record[key], dtype=np.float64) for key in _LOSS_ARRAYS}
        shape = arrays["bce_sum"].shape
        if (len(shape) != 1 or any(value.shape != shape or not np.all(np.isfinite(value))
                                  or np.any(value < 0) for value in arrays.values())):
            raise ValueError("loss statistics require equal nonnegative finite channel arrays")
        pixels = record["valid_pixels"]
        if not isinstance(pixels, (int, np.integer)) or pixels < 0:
            raise ValueError("loss valid_pixels must be a nonnegative integer")
        if result is None:
            result = {**{key: np.zeros(shape, dtype=np.float64) for key in _LOSS_ARRAYS},
                      "valid_pixels": 0}
        if result["bce_sum"].shape != shape:
            raise ValueError("loss statistic channel counts differ")
        for key, value in arrays.items():
            result[key] += value
        result["valid_pixels"] += int(pixels)
    if result is None:
        return None
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _loss_metrics(stats, channels=None):
    if stats is None or not stats["valid_pixels"]:
        return {"loss": None, "bce": None, "dice": None}
    arrays = {key: np.asarray(stats[key], dtype=np.float64) for key in _LOSS_ARRAYS}
    if channels is not None:
        arrays = {key: value[channels] for key, value in arrays.items()}
    if not len(arrays["bce_sum"]):
        return {"loss": None, "bce": None, "dice": None}
    bce = float(arrays["bce_sum"].mean() / stats["valid_pixels"])
    present = arrays["target_sum"] > 0
    per_class = 1 - (2 * arrays["intersection"] + 1e-6) / (
        arrays["probability_sum"] + arrays["target_sum"] + 1e-6)
    dice = float(per_class[present].mean()) if present.any() else 0.
    return {"loss": .5 * bce + .5 * dice, "bce": bce, "dice": dice}


def _class_metrics(counts, categories, stats=None):
    result = []
    for channel, (category, class_counts) in enumerate(zip(categories, counts)):
        tp, fp, fn = map(int, class_counts)
        result.append({"channel": channel, "category_id": category["id"], "name": category["name"],
                       "tp": tp, "fp": fp, "fn": fn,
                       "iou": _ratio(tp, tp + fp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
                       "recall": _ratio(tp, tp + fn), "precision": _ratio(tp, tp + fp),
                       **_loss_metrics(stats, [channel])})
    return result


def _group_metrics(counts, groups, stats=None):
    result = []
    for group, values in zip(groups, counts):
        row = _class_metrics(np.asarray([values]), [{"id": 0, "name": group["name"]}])[0]
        row.pop("channel")
        row.pop("category_id")
        row.update(group, miou=row["iou"], **_loss_metrics(stats, group["channels"]))
        result.append(row)
    return result


def _summary(classes, stats=None):
    return {"miou": _mean(row["iou"] for row in classes),
            **{metric: _mean(row[metric] for row in classes) for metric in ("f1", "recall", "precision")},
            **_loss_metrics(stats),
            "eligible_classes": {metric: sum(row[metric] is not None for row in classes)
                                 for metric in _CLASS_METRICS}}


def _counts(values, channels):
    counts = np.asarray(values)
    if counts.shape != (channels, 3):
        raise ValueError("image counts must have one [TP, FP, FN] row per channel")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("confusion counts must be nonnegative integers")
    if np.any(counts > np.iinfo(np.int64).max):
        raise ValueError("confusion counts exceed int64 capacity")
    return counts.astype(np.int64)


def _average_rows(rows, identity, unit):
    return {**identity, **{metric: _mean(row[metric] for row in rows) for metric in _CLASS_METRICS},
            f"eligible_{unit}": {metric: sum(row[metric] is not None for row in rows)
                                 for metric in _CLASS_METRICS}}


def aggregate_metrics(image_records: Iterable[dict[str, Any]], categories: list[dict],
                      groups=None, allow_unknown_buildings=False) -> dict:
    """Pool native-image counts within each facade, then weight facades equally.

    Unknown TRAIN membership can be reported separately, never as invented
    facade IDs. Optional loss_stats contain additive per-channel sums. Groups
    require group_counts made from the binary union, not summed class counts.
    """
    if not categories or any("id" not in item or "name" not in item for item in categories):
        raise ValueError("categories must contain id and name in model channel order")
    if len({item["id"] for item in categories}) != len(categories):
        raise ValueError("category IDs must be unique")
    groups = groups or []
    grouped, seen_images, auxiliary, stats_list = {}, set(), [], []
    pooled = np.zeros((len(categories), 3), dtype=np.int64)
    pooled_groups = np.zeros((len(groups), 3), dtype=np.int64)
    presence = set()
    for record in image_records:
        image_id = record["image_id"]
        if image_id in seen_images:
            raise ValueError(f"duplicate native image record: {image_id}")
        seen_images.add(image_id)
        building_id = record.get("building_id")
        known = isinstance(building_id, str) and bool(building_id.strip())
        if not known and not allow_unknown_buildings:
            raise ValueError(f"confirmed building_id required for image {image_id}")
        counts = _counts(record["counts"], len(categories))
        group_counts = _counts(record["group_counts"], len(groups)) if groups else pooled_groups.copy()
        stats = merge_loss_statistics([record.get("loss_stats")])
        if stats is not None and len(stats["bce_sum"]) != len(categories):
            raise ValueError("loss statistics require one value per category")
        presence.add(stats is not None)
        stats_list.append(stats)
        pooled += counts
        pooled_groups += group_counts
        if not known:
            classes = _class_metrics(counts, categories, stats)
            auxiliary.append({"image_id": image_id, "building_id": None,
                              **_summary(classes, stats), "classes": classes,
                              "groups": _group_metrics(group_counts, groups, stats)})
            continue
        group = grouped.setdefault(building_id, {"counts": np.zeros_like(pooled),
                                   "group_counts": np.zeros_like(pooled_groups), "stats": [], "images": 0})
        group["counts"] += counts
        group["group_counts"] += group_counts
        group["stats"].append(stats)
        group["images"] += 1
    if len(presence) > 1:
        raise ValueError("loss statistics must be supplied for every image or none")

    per_building = []
    for building_id, group in sorted(grouped.items()):
        stats = merge_loss_statistics(group["stats"])
        classes = _class_metrics(group["counts"], categories, stats)
        per_building.append({"building_id": building_id, "image_count": group["images"],
                             **_summary(classes, stats), "classes": classes,
                             "groups": _group_metrics(group["group_counts"], groups, stats)})

    def mean_classes(rows, unit):
        return [_average_rows([row["classes"][c] for row in rows],
                              {"channel": c, "category_id": category["id"], "name": category["name"]}, unit)
                for c, category in enumerate(categories)]

    def mean_groups(rows, unit):
        values = [_average_rows([row["groups"][g] for row in rows], group, unit)
                  for g, group in enumerate(groups)]
        for row in values:
            row["miou"] = row["iou"]
        return values

    def mean_summary(rows):
        return {metric: _mean(row[metric] for row in rows) for metric in _SUMMARY_METRICS}

    pooled_stats = merge_loss_statistics(stats_list)
    pooled_classes = _class_metrics(pooled, categories, pooled_stats)
    return {
        "aggregation": "sum native-image counts within each known facade; equal mean of facade metrics",
        "evaluation_support": "annotated opaque native pixels only; unannotated pixels ignored",
        "undefined_metric_policy": "null when denominator is zero; excluded from that metric's mean",
        "group_definition": "binary union of member predictions and labels before counting; group mIoU is one-channel union IoU",
        "loss_definition": "0.5 mean masked BCE + 0.5 mean soft-Dice loss on positive classes; statistics pooled within facade; group loss restricts original class channels, not a union-mask loss; dice means Dice loss",
        "image_count": len(seen_images),
        "primary": {**mean_summary(per_building), "building_count": len(per_building),
                    "eligible_buildings": {metric: sum(row[metric] is not None for row in per_building)
                                           for metric in _SUMMARY_METRICS}},
        "per_class": mean_classes(per_building, "buildings"),
        "per_group": mean_groups(per_building, "buildings"),
        "per_building": per_building,
        "auxiliary_image_average": {"aggregation": "equal native-image weights; unknown membership, not facade statistics",
                                    "image_count": len(auxiliary), **mean_summary(auxiliary),
                                    "per_class": mean_classes(auxiliary, "images"),
                                    "per_group": mean_groups(auxiliary, "images")},
        "per_auxiliary_image": auxiliary,
        "pixel_pooled_secondary": {**_summary(pooled_classes, pooled_stats), "classes": pooled_classes,
                                   "groups": _group_metrics(pooled_groups, groups, pooled_stats)},
    }


def write_reports(out: str | Path, metrics: dict) -> dict[str, str]:
    """Write class, facade and operational-group CSVs with undefined cells empty."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    def write(name, rows, fields):
        path = out / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
        paths[name] = str(path)

    def flattened(rows, unit):
        return [{**row, **{f"{key}_{unit}": value for key, value in row[f"eligible_{unit}"].items()}}
                for row in rows]

    write("per_class", flattened(metrics["per_class"], "buildings"),
          ["channel", "category_id", "name", *_CLASS_METRICS,
           *[f"{metric}_buildings" for metric in _CLASS_METRICS]])
    write("per_building", flattened(metrics["per_building"], "classes"),
          ["building_id", "image_count", *_SUMMARY_METRICS,
           *[f"{metric}_classes" for metric in _CLASS_METRICS]])
    group_rows = flattened(metrics.get("per_group", []), "buildings")
    for row in group_rows:
        row["category_names"] = "|".join(row["category_names"])
    write("per_group", group_rows, ["name", "category_names", "miou", *_CLASS_METRICS,
                                    *[f"{metric}_buildings" for metric in _CLASS_METRICS]])
    write("per_facade_group", [{**group, "building_id": building["building_id"]}
                               for building in metrics["per_building"] for group in building.get("groups", [])],
          ["building_id", "name", "tp", "fp", "fn", "miou", *_CLASS_METRICS])
    if metrics.get("per_auxiliary_image"):
        write("auxiliary_per_image", metrics["per_auxiliary_image"], ["image_id", *_SUMMARY_METRICS])
    return paths


_CLASS_NAMES_RU = {
    "WATER_STAIN": "Следы воды", "MISSING_ELEMENT": "Утраченный элемент",
    "DELAMINATION": "Отслоение", "SPALLING": "Выкрашивание",
    "ORNAMENT_INTACT": "Сохранный декор", "CRACK": "Трещина",
    "REPAIRS": "Ремонт", "EFFLORESCENCE": "Высолы",
    "TEXT_OR_IMAGES": "Надписи и изображения", "CORROSION": "Коррозия",
    "ADVERTISEMENTS": "Реклама",
}


def _gallery_colors(categories):
    """Use COCO colors, or a stable category identity rather than channel order."""
    colors = []
    for category in categories:
        color = category.get("color")
        if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
            try:
                colors.append(tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)))
                continue
            except ValueError:
                pass
        identity = f"{category['id']}:{category['name']}".encode("utf-8")
        hue = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") / 2**32
        colors.append(tuple(round(255 * value) for value in colorsys.hsv_to_rgb(hue, .72, .94)))
    return np.asarray(colors, dtype=np.uint8)


def _gallery_panels(rgb, labels, valid, colors):
    """Render all active channels with stripes; per-class panels remain exact.

    A single RGB pixel cannot encode arbitrary overlapping labels unambiguously.
    The overview alternates the colors of its active classes spatially, never
    selects the highest-scoring class, and is complemented by separate channels.
    """
    labels = labels & valid[None]
    count = labels.sum(axis=0)
    yy, xx = np.indices(valid.shape)
    checker = np.where(((xx // 8 + yy // 8) % 2)[..., None], 74, 94)
    colored = np.zeros((*valid.shape, 3), dtype=np.uint8)
    ordinal = np.cumsum(labels, axis=0) - 1
    stripe = ((xx + yy) // 3) % np.maximum(count, 1)
    for channel, color in enumerate(colors):
        selected = labels[channel] & (ordinal[channel] == stripe)
        colored[selected] = color
    overlay = rgb.copy()
    selected = count > 0
    overlay[selected] = np.rint(.45 * rgb[selected] + .55 * colored[selected]).astype(np.uint8)
    overlay[~valid] = np.broadcast_to(checker, rgb.shape)[~valid]
    colored[~valid] = np.broadcast_to(checker, rgb.shape)[~valid]
    return overlay, colored


def write_gallery(out: str | Path, previews: Iterable[dict], categories: list[dict],
                  split="val", epoch=None, checkpoint_name=None) -> Path:
    """Save already computed original crops beside a selected checkpoint.

    Previews contain RGB H,W,3 uint8, boolean target/prediction C,H,W, boolean
    valid H,W, and image_id/building_id/crop_id. The caller chooses a small fixed
    sample without consulting prediction quality. No inference runs here.
    """
    from PIL import Image

    if split not in {"train", "val"}:
        raise ValueError("gallery split must be train or val")
    previews = list(previews)
    colors = _gallery_colors(categories)
    for preview in previews:
        rgb, target, valid, prediction = (np.asarray(preview[key])
                                          for key in ("rgb", "target", "valid", "prediction"))
        confusion_counts(prediction, target, valid)
        if (rgb.shape != (*valid.shape, 3) or rgb.dtype != np.uint8
                or len(target) != len(categories)):
            raise ValueError("gallery requires RGB uint8 and one target/prediction channel per category")
    out = Path(out)
    folder = out / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    escape = lambda value: html.escape(str(value), quote=True)
    title = f"DINOv2: {split.upper()}"
    if epoch is not None:
        title += f" · эпоха {epoch}"
    sections = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'>",
                "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                f"<title>{escape(title)}</title>",
                "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;background:#f2f4f7;color:#17212b}section{background:white;padding:20px;margin:24px 0;border-radius:10px}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-weight:600;margin-bottom:8px}img{max-width:100%;height:auto}.legend{display:flex;flex-wrap:wrap;gap:10px 20px;padding:16px;background:white;border-radius:10px}.key{display:inline-block;width:16px;height:16px;vertical-align:-3px;margin-right:7px;border:1px solid #555}.class-pair{max-width:1024px}summary{cursor:pointer;padding:8px 0}.muted{color:#53606f}@media(max-width:1000px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head><body>",
                f"<h1>{escape(title)}</h1>"]
    if checkpoint_name:
        sections.append(f"<p>Снимок весов: <code>{escape(checkpoint_name)}</code>.</p>")
    sections.append("<p>Фиксированные первые размеченные кропы выбранных фасадов; "
                    "это иллюстрации, а не подборка всех ошибок. Решения по каждому классу: sigmoid ≥ 0.5. "
                    "Серая клетка — пиксели вне размеченной поддержки: они не оценивались и не считаются фоном. "
                    "Чёрный на карте прогноза — ни один класс не предсказан на допустимом пикселе. "
                    "Пересекающиеся классы чередуются полосами; точное наложение каждого класса видно "
                    "в отдельных панелях ниже. RGB показан без маскирования.</p><div class='legend'>")
    for category, color in zip(categories, colors):
        css_color = "#" + "".join(f"{value:02x}" for value in color)
        name = category["name"]
        label = _CLASS_NAMES_RU.get(name, name)
        suffix = f" ({name})" if label != name else ""
        sections.append(f"<span><i class='key' style='background:{css_color}'></i>"
                        f"{escape(label + suffix)} · ID {escape(category['id'])}</span>")
    sections.append("</div>")
    for number, preview in enumerate(previews):
        rgb, target, valid, prediction = (np.asarray(preview[key])
                                          for key in ("rgb", "target", "valid", "prediction"))
        stem = f"{split}-{number:02d}"
        gt_overlay, _ = _gallery_panels(rgb, target, valid, colors)
        pred_overlay, pred_mask = _gallery_panels(rgb, prediction, valid, colors)
        panels = [("Оригинальный RGB", "rgb.jpg", rgb),
                  ("Разметка на RGB", "target.png", gt_overlay),
                  ("Прогноз на RGB", "prediction.png", pred_overlay),
                  ("Предсказанные классы", "classes.png", pred_mask)]
        building = preview.get("building_id") or "Адрес не подтверждён — TRAIN"
        sections.append(f"<section><h2>{escape(building)}</h2><p class='muted'>"
                        f"Фото {escape(preview.get('image_id'))} · кроп {escape(preview.get('crop_id'))}"
                        "</p><div class='panels'>")
        for caption, suffix, pixels in panels:
            filename = f"{stem}-{suffix}"
            Image.fromarray(pixels).save(folder / filename)
            sections.append(f"<figure><figcaption>{caption}</figcaption>"
                            f"<a href='previews/{filename}'><img src='previews/{filename}' "
                            f"alt='{caption}' loading='lazy'></a></figure>")
        sections.append("</div><h3>По классам: разметка слева, прогноз справа</h3>")
        for channel, (category, color) in enumerate(zip(categories, colors)):
            name = category["name"]
            label = _CLASS_NAMES_RU.get(name, name)
            filename = f"{stem}-class-{category['id']}.jpg"
            pair = [_gallery_panels(rgb, labels[channel:channel + 1], valid, color[None])[0]
                    for labels in (target, prediction)]
            Image.fromarray(np.concatenate(pair, axis=1)).save(folder / filename, quality=95)
            sections.append(f"<details><summary>{escape(label)} · {escape(name)}</summary>"
                            f"<img class='class-pair' src='previews/{filename}' "
                            "alt='Разметка слева, прогноз справа' loading='lazy'></details>")
        sections.append("</section>")
    if not previews:
        sections.append("<p>Нет подходящих размеченных кропов для этой подборки.</p>")
    sections.append("</body></html>")
    path = out / "gallery.html"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def semantic_legend(categories: list[dict]) -> list[dict]:
    """A shared, complete legend, including classes absent from an image."""
    result = []
    for category, color in zip(categories, _gallery_colors(categories)):
        name = category["name"]
        translated = _CLASS_NAMES_RU.get(name, name)
        label = f"{translated} ({name})" if translated != name else name
        result.append({"name": f"{label} · ID {category['id']}",
                       "color": "#" + "".join(f"{value:02x}" for value in color)})
    return result


def _write_full_semantic_view(rgb, bits, support, colors, mask_path, overlay_path):
    """Render row blocks, without a C,H,W expansion or an int64 cumsum."""
    from PIL import Image

    height, width = bits.shape
    rendered = np.empty_like(rgb)
    xx = np.arange(width, dtype=np.uint32)[None, :]
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        block, valid = bits[top:bottom], support[top:bottom]
        yy = np.arange(top, bottom, dtype=np.uint32)[:, None]
        counts = np.zeros(block.shape, np.uint8)
        for channel in range(len(colors)):
            counts += ((block & (1 << channel)) != 0) & valid
        stripe = ((xx + yy) // 3) % np.maximum(counts, 1)
        ordinal = np.zeros(block.shape, np.uint8)
        result = rendered[top:bottom]
        result[:] = 0
        for channel, color in enumerate(colors):
            active = ((block & (1 << channel)) != 0) & valid
            result[active & (ordinal == stripe)] = color
            ordinal += active
        checker = np.where((xx // 8 + yy // 8) % 2, 74, 94).astype(np.uint8)
        result[~valid] = checker[~valid, None]
    Image.fromarray(rendered).save(mask_path, compress_level=1)
    # Reuse the color-map allocation for the overlay. Outside labelled support,
    # leave the source photograph visible; the separate mask marks unknowns.
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        selected = (bits[top:bottom] != 0) & support[top:bottom]
        result, source = rendered[top:bottom], rgb[top:bottom]
        result[~selected] = source[~selected]
        result[selected] = np.rint(.45 * source[selected] + .55 * result[selected]).astype(np.uint8)
    Image.fromarray(rendered).save(overlay_path, compress_level=1)


def write_full_image_panels(out: str | Path, *, rgb: np.ndarray, target: np.ndarray,
                            prediction: np.ndarray, valid: np.ndarray, categories: list[dict],
                            image_id: int | str, building_id: str | None = None,
                            split: str = "val", source_name: str | None = None,
                            prediction_valid: np.ndarray | None = None) -> dict:
    """Save native-photo panels immediately and return a small gallery section.

    Target/prediction are H,W uint16 bit fields: bit i is category channel i.
    They preserve multilabel overlaps exactly. ``valid`` is annotation support
    for evaluation; predictions remain visible outside it. Optional prediction
    support masks transparent/padded source pixels, never unannotated regions.
    Native exports remain exact; only browser previews may be resized.
    No inference, EXIF transform, or threshold calibration occurs here.
    """
    from PIL import Image

    rgb, target, prediction, valid = map(np.asarray, (rgb, target, prediction, valid))
    if split not in {"train", "val", "test"}:
        raise ValueError("full-image gallery split must be train, val or test")
    if (not 1 <= len(categories) <= 16
            or len({category["id"] for category in categories}) != len(categories)):
        raise ValueError("full-image gallery requires 1 to 16 unique category IDs")
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or target.shape != rgb.shape[:2] or prediction.shape != target.shape
            or valid.shape != target.shape or not all(target.shape)
            or target.dtype != np.uint16 or prediction.dtype != np.uint16
            or valid.dtype != np.bool_):
        raise ValueError("full-image gallery requires RGB uint8, H,W uint16 bit masks and boolean valid")
    allowed = (1 << len(categories)) - 1
    if np.any(target > allowed) or np.any(prediction > allowed):
        raise ValueError("mask contains bits outside the category channels")
    if prediction_valid is None:
        prediction_valid = np.ones(target.shape, dtype=bool)
    else:
        prediction_valid = np.asarray(prediction_valid)
    if prediction_valid.dtype != np.bool_ or prediction_valid.shape != target.shape:
        raise ValueError("prediction_valid must be boolean H,W")
    if not isinstance(image_id, (str, int)) or isinstance(image_id, bool):
        raise ValueError("image_id must be a string or integer")
    identity = json.dumps([split, image_id], ensure_ascii=False, separators=(",", ":"))
    key = f"{split}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    out = Path(out)
    folder = out / "photos" / key
    # An already exported identity is a caller error; never overwrite panels.
    folder.mkdir(parents=True, exist_ok=False)
    colors = _gallery_colors(categories)
    Image.fromarray(rgb).save(folder / "original.png", compress_level=1)
    _write_full_semantic_view(rgb, target, valid, colors,
                             folder / "gt-masks.png", folder / "gt-overlay.png")
    _write_full_semantic_view(rgb, prediction, prediction_valid, colors,
                             folder / "predicted-masks.png", folder / "predicted-overlay.png")
    np.savez_compressed(folder / "masks.npz", target_bits=target, prediction_bits=prediction,
                        valid=valid, prediction_valid=prediction_valid,
                        category_ids=np.asarray([category["id"] for category in categories]),
                        category_names=np.asarray([category["name"] for category in categories]))
    panels = [{"label": label, "src": f"photos/{key}/{filename}"}
              for label, filename in (("ORIGINAL · исходное фото", "original.png"),
                                       ("GT · разметка на фото", "gt-overlay.png"),
                                       ("GT MASKS · все классы", "gt-masks.png"),
                                       ("PREDICTED · прогноз на фото", "predicted-overlay.png"),
                                       ("PREDICTED MASKS · все классы", "predicted-masks.png"))]
    # Browsers would decode five potentially huge native images per section.
    # Bound page previews, while every link still opens its native PNG.
    for panel in panels:
        panel["href"] = panel["src"]
        if max(rgb.shape[:2]) > 1600:
            native = out / panel["src"]
            preview = native.with_name(native.stem + "-preview.jpg")
            with Image.open(native) as image:
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                image.save(preview, quality=90, subsampling=0)
            panel["src"] = preview.relative_to(out).as_posix()
    name = building_id or "Адрес не подтверждён — TRAIN"
    source = f" · {source_name}" if source_name else ""
    return {"title": f"{split.upper()} · {name} · фото {image_id}",
            "caption": (f"{rgb.shape[1]} × {rgb.shape[0]} пикселей, исходное разрешение{source}. "
                        f"Точные пересекающиеся маски: photos/{key}/masks.npz"),
            "panels": panels}


def write_full_image_gallery(out: str | Path, sections: Iterable[dict], categories: list[dict],
                             split: str = "val", checkpoint_name: str | None = None) -> Path:
    """Write a static all-class gallery for already exported full photographs."""
    from ..visualization import write_static_gallery

    if split not in {"train", "val", "test"}:
        raise ValueError("full-image gallery split must be train, val or test")
    notes = [
        "Все фотографии показаны целиком. Для быстрой загрузки страницы большие фото "
        "имеют уменьшенное превью (до 1600 пикселей по длинной стороне). Нажмите на изображение, "
        "чтобы открыть исходное разрешение PNG; масштаб меняется средствами браузера. "
        "Исходные фото, маски и метрики не уменьшаются.",
        "ORIGINAL — исходный RGB без маскирования. GT показывает известную разметку; "
        "серая клетка в GT MASKS означает неизвестную разметку, а не здоровый фон. "
        "В GT-наложении эти области остаются исходным RGB.",
        "PREDICTED показан на всём фото: sigmoid ≥ 0.5 независимо для каждого класса. "
        "Метрики считаются только на размеченной поддержке. Прогноз за её пределами "
        "не оценён. Чёрный в PREDICTED MASKS означает отсутствие классов выше порога, "
        "а не подтверждённое отсутствие повреждений. Серая клетка в прогнозе, если она есть, "
        "обозначает недопустимые пиксели исходного фото, например прозрачность.",
        "Пересечения классов показаны чередующимися диагональными полосами с цветами "
        "активных каналов. В очень малом пересечении не каждый цвет будет виден. "
        "Это обзор, не argmax и не взаимоисключающая разметка. "
        "Точные битовые маски сохранены рядом в masks.npz; бит i соответствует каналу i "
        "в category_ids/category_names. Легенда включает все классы.",
    ]
    if checkpoint_name:
        notes.insert(0, f"Снимок весов: {checkpoint_name}")
    return write_static_gallery(out, f"Семантическая сегментация · {split.upper()}",
                                sections, legend=semantic_legend(categories), notes=notes)
