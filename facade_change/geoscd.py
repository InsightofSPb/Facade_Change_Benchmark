"""GeoSCD's VGGT geometry branch; native RGB diagnostics, no SAM or change scores.

The output domain is the original reference grid, not a homography union canvas.
Low-resolution rounded projections are interpolated as displacement fields; RGB
is sampled once from the original source. Predicted occlusion stays independent
of geometric support. A completed projection requires manual review.
"""
from __future__ import annotations

import csv
import html
import importlib.metadata
import importlib.util
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from .batch import batch_pairs, comparison_counts
from .geometry import absolute_residual
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json

GEOSCD_COMMIT = "dd31369654e96d6843cc4bbcebce854a8bc2159b"
GEOMETRY_RESOLUTION = 518


def source_provenance(root):
    root = Path(root).expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     text=True, timeout=10).strip()
    if commit != GEOSCD_COMMIT:
        raise ValueError(f"GeoSCD must be checked out at {GEOSCD_COMMIT}, got {commit}")
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
                      check=False, timeout=10).returncode:
        raise ValueError("GeoSCD tracked source differs from the pinned commit")
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "src"],
                                    timeout=10).decode().split("\0")
    hashes = {name: sha256(root / name) for name in names if name and (root / name).is_file()}
    return {"repository": "https://github.com/ZilingLiu/GeoSCD", "commit": commit,
            "root": str(root), "source_sha256": hashes}


def relative_camera_transform(reference_extrinsic, source_extrinsic):
    """World→source @ inverse(world→reference); not an assumption E_reference=I."""
    reference = np.eye(4, dtype=np.float64)
    source = np.eye(4, dtype=np.float64)
    for target, value in ((reference, reference_extrinsic), (source, source_extrinsic)):
        value = np.asarray(value)
        if value.shape != (3, 4) or not np.isfinite(value).all():
            raise ValueError("Expected finite 3x4 camera extrinsics")
        target[:3] = value
    result = source @ np.linalg.inv(reference)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite relative camera transform")
    return result[:3]


def native_remap(coordinates, reference_shape, source_shape, projection_valid=None):
    """Lift a ref-grid→source-grid field to original pixels without border drift.

    Both grids are the same 518² model domain. Upsampling absolute coordinates
    would clamp identity coordinates at the first/last native pixels. Instead,
    resize the displacement and add the analytic affine resize baseline.
    """
    import cv2
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[2] != 2 or min(coordinates.shape[:2]) < 2:
        raise ValueError("Expected a correspondence field [H,W,2] with H,W >=2")
    h, w = map(int, reference_shape[:2])
    sh, sw = map(int, source_shape[:2])
    if min(h, w, sh, sw) < 1 or max(h, w, sh, sw) >= 32767:
        raise ValueError("Native dimensions outside the OpenCV remap range")
    gh, gw = coordinates.shape[:2]
    finite = np.isfinite(coordinates).all(axis=2)
    if projection_valid is not None:
        if np.shape(projection_valid) != (gh, gw):
            raise ValueError("Projection validity must share the correspondence grid")
        finite &= np.asarray(projection_valid, dtype=bool)
    yy, xx = np.mgrid[:gh, :gw]
    displacement = coordinates - np.stack((xx, yy), axis=2)
    displacement[~finite] = 0
    displacement = cv2.resize(displacement, (w, h), interpolation=cv2.INTER_LINEAR)
    validity = cv2.resize(finite.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 1 - 1e-6
    map_x = ((np.arange(w, dtype=np.float32) + .5) * sw / w - .5)[None, :] + displacement[..., 0] * sw / gw
    map_y = ((np.arange(h, dtype=np.float32) + .5) * sh / h - .5)[:, None] + displacement[..., 1] * sh / gh
    validity &= np.isfinite(map_x) & np.isfinite(map_y)
    map_x[~validity] = -1
    map_y[~validity] = -1
    return map_x.astype(np.float32), map_y.astype(np.float32), validity


def dense_warp(reference_rgb, reference_opaque, source_rgb, source_opaque,
               coordinates, projection_valid=None):
    import cv2
    for rgb, opaque in ((reference_rgb, reference_opaque), (source_rgb, source_opaque)):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or opaque.shape != rgb.shape[:2]:
            raise ValueError("Expected native uint8 RGB and same-grid opacity")
    map_x, map_y, projection_support = native_remap(
        coordinates, reference_rgb.shape, source_rgb.shape, projection_valid)
    source = cv2.remap(source_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.remap(source_opaque.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    source_support = projection_support & (coverage >= 1 - 1e-6)
    overlap = reference_opaque & source_support
    return {"reference": reference_rgb, "source": source, "reference_support": reference_opaque,
            "source_support": source_support, "overlap": overlap,
            "reference_only": reference_opaque & ~source_support}


class OfficialGeometry:
    """One lazy VGGT model per batch. No from_pretrained or weight download."""
    def __init__(self, root, checkpoint, device, seed):
        import torch
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("GeoSCD geometry requires an available CUDA device in its separate environment")
        self.torch = torch
        self.device = torch.device(device if ":" in device else f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        root = Path(root)
        sys.path.insert(0, str(root / "src"))
        spec = importlib.util.spec_from_file_location("_facade_geoscd_pixel_match", root / "src/pixel_match.py")
        self.pixel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pixel)
        import vggt.models.vggt as imported_vggt
        if not Path(imported_vggt.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError("A different VGGT package is already imported; start GeoSCD in a fresh process")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = self.pixel.VGGT()
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("Use the separate GeoSCD PyTorch 2.6 environment; safe weights_only loading is required") from exc
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(self.device)
        self.metadata = {"device": str(self.device), "dtype": str(self.dtype),
                         "model_weights_dtype": "torch.float32; autocast only the official aggregator",
                         "device_name": torch.cuda.get_device_name(self.device),
                         "camera_adaptation": "E_source @ inverse(E_reference); upstream assumes E_reference=identity",
                         "proxy": "native RGB → square 1024 bicubic → square 518 bilinear; no Retinex/color transfer",
                         "weights_loading": "torch.load(weights_only=True), strict state_dict"}
        self.metadata["dependencies"] = {}
        for name in ("torch", "torchvision", "einops", "huggingface_hub", "pycolmap", "lightglue", "matplotlib", "scipy"):
            try:
                self.metadata["dependencies"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.metadata["dependencies"][name] = None

    def __call__(self, reference_path, source_path):
        torch, pixel = self.torch, self.pixel
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start = time.monotonic()
        with torch.inference_mode():
            images = pixel.load_and_resize_images([str(reference_path), str(source_path)], 1024,
                                                   light=False, transfer=False).to(self.device)
            extrinsic, intrinsic, depth, confidence = pixel.run_VGGT(self.model, images, self.dtype, 518)
            transform = relative_camera_transform(extrinsic[0, 0].float().cpu().numpy(),
                                                  extrinsic[0, 1].float().cpu().numpy())
            transform = torch.as_tensor(transform, dtype=torch.float32, device=self.device)
            left_depth, right_depth = depth[0, 0].float(), depth[0, 1].float()
            left_k, right_k = intrinsic[0, 0].float(), intrinsic[0, 1].float()
            forward, forward_scatter, _, forward_z = pixel.matching_and_project(transform, left_k, left_depth, right_k)
            reverse, reverse_scatter, _, reverse_z = pixel.matching_project_right2left(transform, right_k, right_depth, left_k)
            left_occ = pixel.dilate_occlusion_mask(pixel.compute_occlusion_mask(forward, right_depth, forward_z))
            right_occ = pixel.dilate_occlusion_mask(pixel.compute_occlusion_mask(reverse, left_depth, reverse_z))
            convert = lambda value: value.detach().float().cpu().numpy()
            result = {"reference_to_source": convert(forward), "source_to_reference": convert(reverse),
                      "reference_occlusion": left_occ, "source_occlusion": right_occ,
                      "reference_projected_z": convert(forward_z), "source_projected_z": convert(reverse_z),
                      "source_scattered_depth_from_reference": convert(forward_scatter),
                      "reference_scattered_depth_from_source": convert(reverse_scatter),
                      "camera_reference_to_source": convert(transform),
                      "predicted_depth": convert(depth[0]), "predicted_depth_confidence": convert(confidence[0])}
        torch.cuda.synchronize(self.device)
        result["inference_seconds"] = time.monotonic() - start
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        return result


def _preview(path, rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((1400, 1000))
    image.save(path, quality=92)


def _save_pair(out, reference, source, arrays, prediction, elapsed):
    import cv2
    ref_rgb, ref_opaque, src_rgb, src_opaque = arrays
    forward, reverse = (np.asarray(prediction[key]) for key in ("reference_to_source", "source_to_reference"))
    expected = (GEOMETRY_RESOLUTION, GEOMETRY_RESOLUTION)
    for coordinates in (forward, reverse):
        if coordinates.shape != expected + (2,):
            raise ValueError("GeoSCD correspondence output must use the 518x518 camera/depth grid")
    valid = []
    for role, coordinates in (("reference", forward), ("source", reverse)):
        z = np.asarray(prediction[f"{role}_projected_z"])
        if z.shape != expected:
            raise ValueError("Projected Z must share the 518x518 correspondence grid")
        valid.append(np.isfinite(coordinates).all(axis=2) & np.isfinite(z) & (z > 0)
                     & (coordinates[..., 0] >= 0) & (coordinates[..., 0] < expected[1])
                     & (coordinates[..., 1] >= 0) & (coordinates[..., 1] < expected[0]))
    warped = dense_warp(*arrays, forward, valid[0])
    overlap = warped["overlap"]
    count = int(overlap.sum())
    # Retain original previews even if the projection has no usable intersection.
    _preview(out / "original_reference_preview.jpg", ref_rgb)
    _preview(out / "original_source_preview.jpg", src_rgb)
    if not count:
        raise ValueError("No fully supported finite, positive-depth geometric overlap")
    archive = {key: np.asarray(value) for key, value in prediction.items()
               if key not in {"inference_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}}
    archive.update(reference_projection_valid=valid[0], source_projection_valid=valid[1])
    np.savez_compressed(out / "correspondences.npz", **archive)
    for role in ("reference", "source"):
        Image.fromarray(warped[role]).save(out / f"{role}_rgb.png")
    for name in ("reference_support", "source_support", "overlap", "reference_only"):
        Image.fromarray(warped[name].astype(np.uint8) * 255).save(out / f"{name}.png")
    occ_fractions = {}
    for role, shape in (("reference", ref_rgb.shape), ("source", src_rgb.shape)):
        occlusion = np.asarray(prediction[f"{role}_occlusion"], dtype=bool)
        if occlusion.shape != expected:
            raise ValueError("Predicted occlusion must share the 518x518 correspondence grid")
        native = cv2.resize(occlusion.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(native * 255).save(out / f"predicted_{role}_occlusion.png")
        occ_fractions[role] = float(occlusion.mean())
    residual = absolute_residual(warped["source"], warped["reference"])
    residual[~overlap] = 0
    Image.fromarray(residual).save(out / "absolute_residual_rgb.png")
    score = residual.mean(axis=2, dtype=np.float32)
    score[~overlap] = np.nan
    np.save(out / "rgb_mean_absolute_difference.npy", score, allow_pickle=False)
    mean_difference = float(score[overlap].mean())
    del score
    overlay = ((ref_rgb.astype(np.uint16) + warped["source"]) // 2).astype(np.uint8)
    overlay[~overlap] = 40
    _preview(out / "overlay_preview.jpg", overlay)
    _preview(out / "residual_preview.jpg", residual)
    yy, xx = np.ogrid[:ref_rgb.shape[0], :ref_rgb.shape[1]]
    checker = np.where(((xx // 128 % 2) == (yy // 128 % 2))[..., None], ref_rgb, warped["source"])
    checker[warped["reference_only"]] = ref_rgb[warped["reference_only"]]
    checker[~ref_opaque] = 40
    _preview(out / "checkerboard_preview.jpg", checker)
    geometry = {"type": "dense_reference_grid", "reference_native_shape": list(ref_rgb.shape[:2]),
                "source_native_shape": list(src_rgb.shape[:2]), "model_grid": list(expected),
                "direction": "inverse sampling: reference-native pixel → source-native pixel",
                "coordinate_lift": "(model_coordinate + 0.5) * native_dimension / 518 - 0.5",
                "native_field": "bilinear model displacement + analytic half-pixel native affine baseline",
                "native_rgb_interpolations": 1, "predicted_occlusion_used_as_support": False,
                "outside_reference_domain": "not projected/scored here; complete source retained by original path/hash and preview",
                "originals": {role: {"image_path": row["image_path"], "sha256": row["sha256"]}
                              for role, row in (("reference", reference), ("source", source))},
                "upstream_adaptations": ["518 camera/depth grid avoids 512 intrinsic-resize origin mismatch",
                                         "relative camera transform E_source @ inverse(E_reference)",
                                         "finite positive projected depth required for geometric support"]}
    write_json(out / "geometry.json", geometry)
    metrics = {"selected_method": "geoscd_geometry", "overlap_pixels": count,
               "overlap_fraction_of_reference_support": count / max(1, int(ref_opaque.sum())),
               "mean_rgb_difference_on_overlap": mean_difference,
               "predicted_occlusion_fraction_on_model_grid": occ_fractions,
               "inference_and_input_seconds": elapsed, "inference_seconds": prediction.get("inference_seconds"),
               "peak_cuda_allocated_bytes": prediction.get("peak_cuda_allocated_bytes"),
               "peak_cuda_reserved_bytes": prediction.get("peak_cuda_reserved_bytes"),
               "review_status": "pending", "interpretation": "Dense geometry computed, not accepted; no homography routing gate or damage accuracy"}
    write_json(out / "diagnostics.json", metrics)
    figures = [("Исходная опора", "original_reference_preview.jpg"), ("Исходный поздний снимок", "original_source_preview.jpg"),
               ("Наложение: геометрическое пересечение", "overlay_preview.jpg"), ("Шахматное совмещение", "checkerboard_preview.jpg"),
               ("RGB-разность", "residual_preview.jpg"), ("Предсказанная окклюзия опоры; отдельно от overlap", "predicted_reference_occlusion.png")]
    markup = ''.join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{name}"></figure>' for title, name in figures)
    (out / "gallery.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style>'
                                      '<h1>GeoSCD: только геометрическая ветвь</h1><p>Нужна ручная проверка. Области позднего кадра вне исходной опоры здесь не оцениваются. '
                                      'Маска предсказанных окклюзий не исключает пиксели из overlap.</p>' + markup + '</html>', encoding="utf-8")
    return metrics


def _comparison_inputs(comparison_run, manifest_hash, selected):
    if comparison_run is None:
        return None, []
    root = Path(comparison_run).expanduser().resolve()
    record = read_json(root / "run.json")
    if record.get("status") not in {"completed_needs_review", "completed_with_issues", "interrupted"}:
        raise ValueError("Comparison run must have a final status")
    if record.get("config", {}).get("manifest_sha256") != manifest_hash:
        raise ValueError("Comparison run uses a different prepared manifest")
    for name in ("results.json", "selected_pairs.json"):
        if record.get("artifact_sha256", {}).get(name) != sha256(root / name):
            raise ValueError(f"Comparison artifact changed or lacks a recorded hash: {name}")
    old = {row["pair_id"]: row for row in read_json(root / "selected_pairs.json")}
    for pair in selected:
        if pair["pair_id"] not in old or any(str(old[pair["pair_id"]][key]) != str(pair[key])
                                             for key in ("reference_id", "source_id", "split", "view_id")):
            raise ValueError("GeoSCD selection is absent from or differs from the comparison run")
    return root, read_json(root / "results.json")


def _write_batch_outputs(out, selected, rows, comparison_root, previous_rows):
    computed = sum(row["status"] == "computed_needs_review" for row in rows)
    summary = {"eligible_pairs": len(selected), "attempted_runs": len(rows), "computed_needs_review": computed,
               "computed_runs": computed,
               "failed_runs": sum(row["status"] == "failed" for row in rows),
               "not_attempted": len(selected) - len(rows), "accepted_by_manual_review": 0,
               "method": "geoscd_geometry", "interpretation": "Computed is not the SIFT/LoFTR passed gate; no damage accuracy"}
    if comparison_root:
        methods = sorted({row["method"] for row in previous_rows})
        summary["previous_routing_gates_on_same_selected_pairs"] = comparison_counts(selected, methods, previous_rows)
    write_json(out / "results.json", rows)
    write_json(out / "summary.json", summary)
    (out / "summary.txt").write_text(f"GeoSCD geometry: computed {computed}/{len(selected)}; failed {summary['failed_runs']}; not attempted {summary['not_attempted']}\n"
                                     "Computed requires manual review; not comparable to a passed homography routing gate.\n"
                                     "No SAM, change detector, crops or damage GT were run. Inspect comparison.html.\n", encoding="utf-8")
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["pair_id", "reference_file", "source_file", "method", "path", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lookup = {row["pair_id"]: row for row in rows}
    old_lookup = {(row["pair_id"], row["method"]): row for row in previous_rows}
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}td,th{padding:12px;vertical-align:top;border:1px solid #ccc}img{width:320px;max-width:100%}small{display:block;max-width:320px}</style>',
             '<h1>SIFT / LoFTR / GeoSCD geometry</h1><p>GeoSCD: вычислено ≠ принято. Холст GeoSCD — исходная опора; гомографии — объединённый холст. Маски окклюзий не исключают ошибки из сравнения.</p><table><tr><th>Пара</th><th>SIFT</th><th>LoFTR</th><th>GeoSCD geometry</th></tr>']
    for pair in selected:
        parts.append('<tr><th>' + html.escape(f"{pair['reference_file']} → {pair['source_file']}") + '<br>' + html.escape(pair['split']) + '</th>')
        for method in ("sift", "loftr", "geoscd_geometry"):
            row = lookup.get(pair["pair_id"]) if method == "geoscd_geometry" else old_lookup.get((pair["pair_id"], method))
            if not row:
                parts.append('<td>Не запущено</td>')
                continue
            status = html.escape(row["status"])
            if row["status"] == "failed":
                parts.append('<td>' + status + '<small>' + html.escape(row.get("error", "")) + '</small></td>')
                continue
            root = out if method == "geoscd_geometry" else comparison_root
            relative = quote(os.path.relpath(root / row["path"], out), safe="/")
            parts.append(f'<td>{status}<a href="{relative}/gallery.html"> Подробности</a><br><img loading="lazy" src="{relative}/overlay_preview.jpg"><br><img loading="lazy" src="{relative}/residual_preview.jpg"></td>')
        parts.append('</tr>')
    (out / "comparison.html").write_text('\n'.join(parts) + '</table></html>', encoding="utf-8")
    return summary


def run_geoscd(manifest_path, out, geoscd_root, checkpoint, pairs=None, limit=3, split="all",
               device="cuda:0", resolution=518, seed=42, comparison_run=None, *, _backend=None):
    """Sequential geometry trial on existing prepared pairs; never alter their split."""
    if resolution != GEOMETRY_RESOLUTION:
        raise ValueError("Use resolution=518: the upstream 512 intrinsic resize lacks a half-pixel correction")
    manifest_path = Path(manifest_path).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    geoscd_root = Path(geoscd_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError("A local VGGT model.pt checkpoint is required; no automatic download")
    manifest_hash = sha256(manifest_path)
    manifest = read_json(manifest_path)
    selected = batch_pairs(manifest, pairs, limit, split)
    by_id = {str(row["image_id"]): row for row in manifest["images"]}
    for pair in selected:
        pair.update(reference_file=by_id[str(pair["reference_id"])]["file_name"],
                    source_file=by_id[str(pair["source_id"])]["file_name"])
    comparison_root, previous_rows = _comparison_inputs(comparison_run, manifest_hash, selected)
    config = {"manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
              "geoscd_root": str(geoscd_root), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "device": device, "resolution": resolution, "seed": seed, "pairs": pairs, "limit": limit, "split": split,
              "comparison_run": str(comparison_root) if comparison_root else None}
    if comparison_root:
        config["comparison_input_sha256"] = {name: sha256(comparison_root / name)
                                             for name in ("run.json", "selected_pairs.json", "results.json")}
    out = new_directory(out)
    record = run_record("geoscd_geometry_batch", config)
    rows = []
    write_json(out / "selected_pairs.json", selected)
    write_json(out / "run.json", record)
    try:
        record["upstream"] = source_provenance(geoscd_root)
        try:
            record["gpu_before"] = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            record["gpu_before"] = f"Unavailable: {type(exc).__name__}"
        backend = _backend if _backend is not None else OfficialGeometry(geoscd_root, checkpoint, device, seed)
        if sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("VGGT checkpoint changed while loading the model")
        record["backend"] = getattr(backend, "metadata", {"id": "injected CPU test backend"})
        write_json(out / "run.json", record)
        with (out / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "reference_file", "source_file", "review_status", "notes"])
            writer.writeheader()
            writer.writerows({**{key: pair[key] for key in ("pair_id", "reference_file", "source_file")},
                              "review_status": "pending", "notes": ""} for pair in selected)
        for index, pair in enumerate(selected, 1):
            print(f"GeoSCD {index}/{len(selected)}: {pair['reference_file']} → {pair['source_file']}", flush=True)
            row = {**pair, "method": "geoscd_geometry", "path": f"geoscd/pair-{pair['pair_id']}"}
            child = new_directory(out / row["path"])
            child_record = run_record("geoscd_geometry_pair", {**config, "reference_id": str(pair["reference_id"]), "source_id": str(pair["source_id"])})
            reference, source = by_id[str(pair["reference_id"])], by_id[str(pair["source_id"])]
            child_record["observations"] = {"reference": reference, "source": source}
            child_record["upstream"] = record["upstream"]
            child_record["backend"] = record["backend"]
            write_json(child / "run.json", child_record)
            start = time.monotonic()
            try:
                arrays = []
                for image in (reference, source):
                    if sha256(image["image_path"]) != image["sha256"]:
                        raise ValueError(f"Image changed since prepared manifest: {image['file_name']}")
                    rgb, opaque = load_rgb(image["image_path"])
                    if rgb.shape[:2] != (image["height"], image["width"]):
                        raise ValueError("Decoded native dimensions differ from prepared manifest")
                    arrays.extend((rgb, opaque))
                prediction = backend(reference["image_path"], source["image_path"])
                for image in (reference, source):
                    if sha256(image["image_path"]) != image["sha256"]:
                        raise ValueError(f"Image changed during geometry inference: {image['file_name']}")
                row["diagnostics"] = _save_pair(child, reference, source, arrays, prediction, time.monotonic() - start)
                row["diagnostics"]["elapsed_seconds_with_artifacts"] = time.monotonic() - start
                write_json(child / "diagnostics.json", row["diagnostics"])
                row["status"] = "computed_needs_review"
                finish_record(child, child_record, "completed_needs_review")
            except KeyboardInterrupt:
                finish_record(child, child_record, "interrupted", "Interrupted by user")
                raise
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                finish_record(child, child_record, "failed", row["error"])
            rows.append(row)
            print(f"  geoscd_geometry: {row['status']}{'; ' + row['error'] if 'error' in row else ''}", flush=True)
            summary = _write_batch_outputs(out, selected, rows, comparison_root, previous_rows)
        record["summary"] = summary
        if source_provenance(geoscd_root) != record["upstream"]:
            raise ValueError("Pinned GeoSCD source changed during geometry inference")
        finish_record(out, record, "completed_with_issues" if summary["failed_runs"] else "completed_needs_review")
        return summary
    except KeyboardInterrupt:
        _write_batch_outputs(out, selected, rows, comparison_root, previous_rows)
        finish_record(out, record, "interrupted", "Interrupted by user; completed geometry runs preserved")
        raise
    except Exception as exc:
        _write_batch_outputs(out, selected, rows, comparison_root, previous_rows)
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
