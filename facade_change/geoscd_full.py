"""Full pinned GeoSCD: two ordered VGGT projections plus bundled SAM/GeSCF.

The official detector operates on square 512-pixel inputs. Its binary output is
lifted to the original reference grid; SAM cosine diagnostics are scores, not
probabilities. RGB diagnostics still sample the unmodified original source once.
"""
from __future__ import annotations

import csv
import ast
import html
import importlib.util
import inspect
import subprocess
import sys
import time
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from .batch import batch_pairs
from .geoscd import OfficialGeometry, _preview, source_provenance
from .geometry import absolute_residual
from .io import finish_record, load_rgb, new_directory, read_json, run_record, sha256, write_json

FULL_RESOLUTION = 512


def skip_unused_sam_attention(forward):
    """Keep the pinned Block result, omit its discarded first full-grid attention.

    The first attention output is used only by return_qkv=True. This changes no
    active output path; it avoids a redundant attention matrix in normal calls.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(forward)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or len(function.body) < 3:
        raise ValueError("Unexpected bundled SAM Block.forward")
    first, second = function.body[:2]
    if (ast.unparse(first) != "qkv = self.attn(self.norm1(x), return_qkv)"
            or ast.unparse(second) != "if return_qkv:\n    return qkv"):
        raise ValueError("Bundled SAM unused-attention prefix differs from pinned source")
    function.body[:2] = [ast.If(test=second.test, body=[ast.Return(value=first.value)], orelse=[])]
    ast.fix_missing_locations(tree)
    namespace = dict(forward.__globals__)
    exec(compile(tree, forward.__code__.co_filename, "exec"), namespace)
    return namespace[forward.__name__]


def validate_sam_vit_h_state(state):
    """SAM 3 and SAM B/L are not substitutes for the upstream SAM 1 ViT-H."""
    expected = {"image_encoder.patch_embed.proj.weight": (1280, 3, 16, 16),
                "image_encoder.blocks.31.attn.qkv.weight": (3840, 1280)}
    if not isinstance(state, Mapping) or any(tuple(getattr(state.get(key), "shape", ())) != shape
                                             for key, shape in expected.items()):
        raise ValueError("Full GeoSCD requires SAM 1 ViT-H sam_vit_h_4b8939.pth; SAM 3 / sam3.pt and SAM B/L are incompatible")


def merge_directional_masks(reference_mask, source_mask, reference_to_source):
    """Exact upstream gather + OR; not forward scatter of source pixels."""
    reference_mask, source_mask = np.asarray(reference_mask, bool), np.asarray(source_mask, bool)
    coordinates = np.asarray(reference_to_source)
    if reference_mask.shape != source_mask.shape or coordinates.shape != reference_mask.shape + (2,):
        raise ValueError("Directional masks and correspondence grid must share the same shape")
    h, w = source_mask.shape
    finite = np.isfinite(coordinates).all(axis=2)
    safe = np.where(finite[..., None], coordinates, -1).astype(np.int64)
    x, y = safe[..., 0], safe[..., 1]
    valid = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    sampled = np.zeros_like(reference_mask)
    sampled[valid] = source_mask[y[valid], x[valid]]
    return reference_mask | sampled


def native_camera_warp(reference_rgb, reference_opaque, source_rgb, source_opaque,
                       coordinates, projected_z):
    """Adapt the official zero-origin 518→512 K scaling for native diagnostics.

    Model endpoints lift as (q * 518 / 512 + .5) * native_size / 518 -.5.
    This differs from ordinary half-pixel resizing of the detector's 512 raster.
    """
    import cv2
    coordinates = np.asarray(coordinates, dtype=np.float32)
    z = np.asarray(projected_z)
    if coordinates.shape != (512, 512, 2) or z.shape != (512, 512):
        raise ValueError("Expected the official 512x512 correspondence and projected-depth grid")
    h, w = reference_rgb.shape[:2]
    sh, sw = source_rgb.shape[:2]
    if min(h, w, sh, sw) < 1 or max(h, w, sh, sw) >= 32767:
        raise ValueError("Native image dimensions outside OpenCV remap range")
    finite = np.isfinite(coordinates).all(axis=2) & np.isfinite(z) & (z > 0)
    finite &= (coordinates[..., 0] >= 0) & (coordinates[..., 0] < 512) & (coordinates[..., 1] >= 0) & (coordinates[..., 1] < 512)
    yy, xx = np.mgrid[:512, :512]
    displacement = coordinates - np.stack((xx, yy), axis=2)
    displacement[~finite] = 0
    qx = (((np.arange(w, dtype=np.float32) + .5) * 518 / w - .5) * 512 / 518)
    qy = (((np.arange(h, dtype=np.float32) + .5) * 518 / h - .5) * 512 / 518)
    query_x, query_y = np.meshgrid(qx, qy)
    flow = cv2.remap(displacement.astype(np.float32), query_x, query_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    support = cv2.remap(finite.astype(np.float32), query_x, query_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) >= 1 - 1e-6
    map_x = ((np.arange(w, dtype=np.float32) + .5) * sw / w - .5)[None, :] + flow[..., 0] * sw / 512
    map_y = ((np.arange(h, dtype=np.float32) + .5) * sh / h - .5)[:, None] + flow[..., 1] * sh / 512
    map_x[~support], map_y[~support] = -1, -1
    warped = cv2.remap(source_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.remap(source_opaque.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    source_support = support & (coverage >= 1 - 1e-6)
    return warped, source_support, reference_opaque & source_support


class OfficialFull:
    """Reuse two CPU resident models, putting only the active model on CUDA."""
    def __init__(self, root, checkpoint, sam_checkpoint, device, seed, mode, points_per_side, iou_thresh, sem_filter):
        import torch
        root = Path(root).resolve()
        sys.path.insert(0, str(root / "src"))
        # Inspect SAM safely before allocating either model on the GPU.
        state = torch.load(str(sam_checkpoint), map_location="cpu", weights_only=True)
        validate_sam_vit_h_state(state)
        import segment_anything_model
        if not Path(segment_anything_model.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Full GeoSCD must use its bundled modified segment_anything_model")
        sam = segment_anything_model.sam_model_registry["vit_h"](checkpoint=None)
        sam.load_state_dict(state, strict=True)
        del state
        sam.eval()
        self.sam = sam  # Explicitly move all SAM components, including the AMG's shared model.
        from segment_anything_model.modeling.image_encoder import Block
        self.sam_block_class = Block
        self.sam_forward = skip_unused_sam_attention(Block.forward)
        self.geometry = OfficialGeometry(root, checkpoint, device, seed)
        self.geometry.model.cpu()
        torch.cuda.empty_cache()
        spec = importlib.util.spec_from_file_location("_facade_geoscd_full_framework", root / "src/framework.py")
        self.framework = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.framework)
        for name in ("utils", "pseudo_generator", "registration"):
            if not Path(sys.modules[name].__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"Unrelated {name} module is already imported; use a fresh process")
        self.args = SimpleNamespace(test_dataset="Random", output_size=512, feature_facet="key", feature_layer=17,
                                    embedding_layer=32, sam_backbone="vit_h", pseudo_backbone="vit_h",
                                    points_per_side=points_per_side, pred_iou_thresh=.7, stability_score_thresh=.7,
                                    mode=mode, iou_thresh=iou_thresh, sem_filter=sem_filter)
        # The upstream same-backbone path shares one SAM instance already.
        with patch.object(self.framework, "load_backbones", return_value=(sam, sam)):
            self.segmenter = self.framework.GeSCF(self.args)
        self.segmenter.eval().cpu()
        self.torch = torch
        geometry_metadata = {**self.geometry.metadata,
                             "camera_adaptation": "none: official full baseline uses E_source assuming E_reference=identity",
                             "correspondence_grid": [512, 512], "intrinsics": "official zero-origin scaling from 518 to 512"}
        self.metadata = {"method": "full GeoSCD VGGT + bundled SAM 1 ViT-H GeSCF",
                         "geometry": geometry_metadata, "model_grid": [512, 512],
                         "full_detector": vars(self.args), "geometry_runs_per_pair": 2,
                         "camera_convention": "faithful upstream E_source only; assumes first camera is identity",
                         "intrinsic_resize": "faithful upstream zero-origin 518→512 scaling",
                         "mask_fusion": "reference directional mask OR source mask gathered with reference→source projection",
                         "device_schedule": "VGGT CUDA then CPU; shared SAM CUDA then CPU, same weights, no SAM3 substitution",
                         "sam_memory_adaptation": "omit discarded first Block attention when return_qkv=False; active output and QKV paths unchanged",
                         "safe_sam_loading": "bundled vit_h(checkpoint=None), weights_only=True, strict state_dict",
                         "score": "1 - official multi-head SAM key cosine, not calibrated probability",
                         "ground_truth": "not used; no metrics computed"}
        self.metadata["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(self.geometry.device).total_memory

    def _geometry(self, reference_path, source_path):
        pixel, torch = self.geometry.pixel, self.torch
        captured = {}
        original = pixel.matching_and_project
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured["projected_z"] = result[3].detach().float().cpu().numpy()
            return result
        with patch.object(pixel, "matching_and_project", side_effect=capture):
            output = pixel.run_dense_match([str(reference_path), str(source_path)], self.geometry.model,
                                           self.geometry.device, self.geometry.dtype, resolution=512, light=False, transfer=False)
        coordinates, occlusion, scatter, _, _, _, extrinsic = output
        return {"coordinates": coordinates.detach().cpu().numpy(), "occlusion": np.asarray(occlusion, bool),
                "scattered_depth": scatter.detach().float().cpu().numpy(),
                "camera_extrinsic": extrinsic.detach().float().cpu().numpy(), **captured}

    def _detect(self, reference_path, source_path, geometry):
        torch = self.torch
        captured = {}
        original = self.framework.match_multihead_key_avg
        def capture(*args, **kwargs):
            similarity, valid = original(*args, **kwargs)
            scores = similarity.detach().float().cpu().numpy()
            support = valid.detach().cpu().numpy().astype(bool)
            if not support.any() or not np.isfinite(scores[support]).all():
                raise ValueError("No finite valid SAM correspondence similarities")
            captured.update(score=(1 - scores).astype(np.float32), feature_valid=support)
            values = scores[support]
            captured["statistics"] = {"valid_feature_pixels": int(support.sum()),
                                      "similarity_std": float(values.std()),
                                      "similarity_mad": float(np.median(np.abs(values - np.median(values)))),
                                      "zero_std_or_mad": bool(values.std() == 0 or np.median(np.abs(values - np.median(values))) == 0)}
            return similarity, valid
        coordinates = torch.as_tensor(geometry["coordinates"], dtype=torch.int64, device=self.geometry.device)
        depth = torch.as_tensor(geometry["scattered_depth"], device=self.geometry.device)
        with patch.object(self.framework, "match_multihead_key_avg", side_effect=capture):
            mask = self.segmenter(str(reference_path), str(source_path), self.args, coordinates, depth,
                                  ignore_left=geometry["occlusion"], debug=False)
        if "score" not in captured:
            raise RuntimeError("Official GeSCF did not produce its SAM key similarity map")
        return {"mask": np.asarray(mask, bool), **captured}

    def __call__(self, reference_path, source_path):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(self.geometry.device)
        torch.cuda.synchronize(self.geometry.device)
        start = time.monotonic()
        try:
            with torch.cuda.device(self.geometry.device), torch.inference_mode():
                self.geometry.model.to(self.geometry.device)
                print("  VGGT: reference → source", flush=True)
                forward = self._geometry(reference_path, source_path)
                print("  VGGT: source → reference", flush=True)
                reverse = self._geometry(source_path, reference_path)
                torch.cuda.synchronize(self.geometry.device)
                geometry_seconds = time.monotonic() - start
                self.geometry.model.cpu()
                torch.cuda.empty_cache()
                segment_start = time.monotonic()
                self.sam.to(self.geometry.device)
                self.segmenter.to(self.geometry.device)
                with patch.object(self.sam_block_class, "forward", self.sam_forward):
                    print(f"  SAM: reference → source (VGGT {geometry_seconds:.2f}s)", flush=True)
                    left = self._detect(reference_path, source_path, forward)
                    print("  SAM: source → reference", flush=True)
                    right = self._detect(source_path, reference_path, reverse)
                torch.cuda.synchronize(self.geometry.device)
                sam_seconds = time.monotonic() - segment_start
                print(f"  SAM finished: {sam_seconds:.2f}s", flush=True)
            final = merge_directional_masks(left["mask"], right["mask"], forward["coordinates"])
            return {"reference_geometry": forward, "source_geometry": reverse,
                    "reference_detection": left, "source_detection": right, "final_reference_mask": final,
                    "geometry_seconds": geometry_seconds, "sam_seconds": sam_seconds,
                    "inference_seconds": time.monotonic() - start,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.geometry.device),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.geometry.device)}
        finally:
            self.geometry.model.cpu()
            self.segmenter.automatic_mask_generator.predictor.reset_image()
            self.segmenter.cpu()
            self.sam.cpu()
            torch.cuda.empty_cache()


def _save_prediction(out, reference, source, arrays, prediction):
    import cv2
    ref_rgb, ref_opaque, src_rgb, src_opaque = arrays
    forward, reverse = prediction["reference_geometry"], prediction["source_geometry"]
    warped, source_support, overlap = native_camera_warp(*arrays, forward["coordinates"], forward["projected_z"])
    if not overlap.any():
        raise ValueError("No finite positive-depth native geometric overlap")
    source_warped, ref_support_reverse, reverse_overlap = native_camera_warp(src_rgb, src_opaque, ref_rgb, ref_opaque,
                                                                           reverse["coordinates"], reverse["projected_z"])
    archive = {}
    for role, geometry, detection, rgb, geometric in (
        ("reference", forward, prediction["reference_detection"], ref_rgb, overlap),
        ("source", reverse, prediction["source_detection"], src_rgb, reverse_overlap),
    ):
        h, w = rgb.shape[:2]
        for name in ("mask", "score", "feature_valid"):
            array = np.asarray(detection[name])
            if array.shape != (512, 512):
                raise ValueError("Official SAM outputs must use the 512x512 detector grid")
            archive[f"{role}_{name}"] = array
        for name, value in geometry.items():
            archive[f"{role}_{name}"] = np.asarray(value)
        mask = cv2.resize(np.asarray(detection["mask"], np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(mask * 255).save(out / f"directional_{role}_change_mask_native.png")
        feature_valid = np.asarray(detection["feature_valid"], bool) & np.isfinite(detection["score"])
        values = np.where(feature_valid, detection["score"], 0).astype(np.float32)
        score = cv2.resize(values, (w, h), interpolation=cv2.INTER_LINEAR)
        score_valid = cv2.resize(feature_valid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 1 - 1e-6
        score[~(score_valid & geometric)] = np.nan
        np.save(out / f"{role}_sam_key_change_score_native.npy", score, allow_pickle=False)
        preview_score = np.nan_to_num(score, nan=0).clip(0, 2) / 2
        heatmap = cv2.applyColorMap((preview_score * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        heatmap[~np.isfinite(score)] = 40
        _preview(out / f"{role}_score_preview.jpg", heatmap)
        occlusion = cv2.resize(np.asarray(geometry["occlusion"], np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(occlusion * 255).save(out / f"predicted_{role}_occlusion.png")
        _preview(out / f"original_{role}_preview.jpg", rgb)
    final = np.asarray(prediction["final_reference_mask"], bool)
    if final.shape != (512, 512):
        raise ValueError("Full GeoSCD final binary mask must use the 512x512 detector grid")
    archive["final_reference_mask"] = final
    np.savez_compressed(out / "full_predictions_model_grid.npz", **archive)
    native_mask = cv2.resize(final.astype(np.uint8), (ref_rgb.shape[1], ref_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(native_mask * 255).save(out / "final_change_mask_reference_native.png")
    Image.fromarray((native_mask.astype(bool) & overlap).astype(np.uint8) * 255).save(out / "final_change_mask_on_geometric_overlap.png")
    for role, rgb in (("reference", ref_rgb), ("source", warped)):
        Image.fromarray(rgb).save(out / f"{role}_rgb.png")
    for name, mask in (("reference_support", ref_opaque), ("source_support", source_support),
                       ("overlap", overlap), ("source_native_geometric_overlap", reverse_overlap)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{name}.png")
    residual = absolute_residual(warped, ref_rgb)
    residual[~overlap] = 0
    Image.fromarray(residual).save(out / "absolute_residual_rgb.png")
    overlay = ((ref_rgb.astype(np.uint16) + warped) // 2).astype(np.uint8)
    overlay[~overlap] = 40
    _preview(out / "alignment_overlay_preview.jpg", overlay)
    change_overlay = ref_rgb.copy()
    changed = native_mask.astype(bool)
    change_overlay[changed] = (ref_rgb[changed].astype(np.float32) * .55 + np.array([255, 40, 100]) * .45).astype(np.uint8)
    _preview(out / "change_overlay_preview.jpg", change_overlay)
    metadata = {"method": "full GeoSCD", "native_reference_shape": list(ref_rgb.shape[:2]),
                "native_source_shape": list(src_rgb.shape[:2]), "detector_grid": [512, 512],
                "mask_native_lift": "nearest resize of official reference-grid binary mask; not native-resolution inference",
                "native_rgb_warp": "zero-origin camera-grid endpoint lift; one interpolation from original source",
                "camera_grid_lift": "(coordinate * 518/512 + .5) * native_dimension/518 -.5",
                "source_regions_outside_reference": "retained in original and directional source-grid outputs",
                "geometric_support": "finite positive projected depth, correspondence bounds, original alpha; no predicted occlusion exclusion",
                "occlusion_use": "part of official occupy detector, never an evaluation-support mask",
                "scores": "1 - SAM multi-head key cosine; diagnostic, not probability or final-mask confidence",
                "originals": {role: {"image_path": row["image_path"], "sha256": row["sha256"]}
                              for role, row in (("reference", reference), ("source", source))}}
    write_json(out / "prediction_contract.json", metadata)
    metrics = {"method": "geoscd_full", "review_status": "pending", "overlap_pixels": int(overlap.sum()),
               "overlap_fraction_of_reference_support": float(overlap.sum() / max(1, ref_opaque.sum())),
               "predicted_changed_reference_pixels": int(native_mask.sum()),
               "reference_similarity_statistics": prediction["reference_detection"].get("statistics", {}),
               "source_similarity_statistics": prediction["source_detection"].get("statistics", {}),
               **{name: prediction.get(name) for name in ("geometry_seconds", "sam_seconds", "inference_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")},
               "interpretation": "Full detector output computed; no change GT, F1/AP, acceptance or homography routing gate"}
    write_json(out / "diagnostics.json", metrics)
    figures = [("Исходная опора", "original_reference_preview.jpg"), ("Исходный поздний снимок", "original_source_preview.jpg"),
               ("Совмещение оригинальных RGB", "alignment_overlay_preview.jpg"), ("Предсказанное изменение полного GeoSCD", "change_overlay_preview.jpg"),
               ("SAM key: 1−cosine; шкала 0…2, не вероятность", "reference_score_preview.jpg"),
               ("Предсказанная окклюзия; отдельно от геометрической поддержки", "predicted_reference_occlusion.png")]
    (out / "gallery.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px}img{max-width:100%}</style>'
                                     '<h1>Полный GeoSCD: VGGT + SAM</h1><p>Предсказание изменения не равно подтверждённому повреждению. '
                                     'Сетка модели 512×512; вывод перенесён на исходную опору. Истинная временная разметка не использовалась.</p>'
                                     + ''.join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{filename}"></figure>' for title, filename in figures)
                                     + '</html>', encoding="utf-8")
    return metrics


def _write_outputs(out, selected, rows):
    computed = sum(row["status"] == "computed_needs_review" for row in rows)
    summary = {"method": "geoscd_full", "eligible_pairs": len(selected), "attempted_runs": len(rows),
               "computed_runs": computed, "failed_runs": sum(row["status"] == "failed" for row in rows),
               "eligible_building_count": len({pair["building_id"] for pair in selected}),
               "pair_count_by_split": {name: sum(pair["split"] == name for pair in selected)
                                       for name in sorted({pair["split"] for pair in selected})},
               "not_attempted": len(selected) - len(rows), "evaluation_ready": False, "temporal_ground_truth": False,
               "interpretation": "Full VGGT+SAM detector computed; visual review and independent temporal GT required"}
    write_json(out / "results.json", rows)
    write_json(out / "summary.json", summary)
    (out / "summary.txt").write_text(f"Full GeoSCD VGGT+SAM: computed {computed}/{len(selected)}; failed {summary['failed_runs']}; not attempted {summary['not_attempted']}\n"
                                     "No F1/AP or homography routing gate: temporal GT absent. Inspect comparison.html.\n", encoding="utf-8")
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "reference_file", "source_file", "method", "status", "path", "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lookup = {row["pair_id"]: row for row in rows}
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px}img{width:360px;max-width:100%}td,th{padding:12px;border:1px solid #ddd;vertical-align:top}</style><h1>Полный GeoSCD VGGT + SAM</h1><p>Вычислено ≠ принято. Предсказания без истинной временной разметки.</p><table>']
    for pair in selected:
        parts.append('<tr><th>' + html.escape(f"{pair['reference_file']} → {pair['source_file']}") + '<br>' + html.escape(pair['split']) + '</th>')
        row = lookup.get(pair["pair_id"])
        if not row:
            parts.append('<td>Не запущено</td>')
        elif row["status"] == "failed":
            parts.append('<td>' + html.escape(row.get("error", "")) + '</td>')
        else:
            path = html.escape(row["path"], quote=True)
            parts.append(f'<td><a href="{path}/gallery.html">Подробности</a><br>{row["status"]}<br><img loading="lazy" src="{path}/change_overlay_preview.jpg"><br><img loading="lazy" src="{path}/reference_score_preview.jpg"></td>')
        parts.append('</tr>')
    (out / "comparison.html").write_text('\n'.join(parts) + '</table></html>', encoding="utf-8")
    return summary


def run_geoscd_full(manifest_path, out, geoscd_root, checkpoint, sam_checkpoint, device="cuda:0",
                    pairs=None, limit=3, split="all", seed=42, mode="occupy", points_per_side=32,
                    iou_thresh=.65, sem_filter=None, *, _backend=None):
    """Run the full official detector on already prepared, split-preserving pairs."""
    if mode not in {"initial", "occupy"} or points_per_side < 1 or not 0 <= iou_thresh <= 1:
        raise ValueError("Require mode initial/occupy, positive points-per-side and IoU threshold in [0,1]")
    if sem_filter is not None and not -1 <= sem_filter <= 1:
        raise ValueError("Semantic cosine filter must be in [-1,1]")
    manifest_path, root, checkpoint, sam_checkpoint = (Path(path).expanduser().resolve()
                                                      for path in (manifest_path, geoscd_root, checkpoint, sam_checkpoint))
    if "sam3" in sam_checkpoint.name.lower() or "sam_3" in sam_checkpoint.name.lower():
        raise ValueError("sam3.pt is incompatible: full GeoSCD requires SAM 1 ViT-H sam_vit_h_4b8939.pth")
    if not checkpoint.is_file() or not sam_checkpoint.is_file():
        raise ValueError("Full GeoSCD requires local VGGT and SAM 1 ViT-H checkpoints; no automatic downloads")
    manifest = read_json(manifest_path)
    selected = batch_pairs(manifest, pairs, limit, split)
    images = {str(row["image_id"]): row for row in manifest["images"]}
    for pair in selected:
        pair.update(reference_file=images[str(pair["reference_id"])]["file_name"], source_file=images[str(pair["source_id"])]["file_name"])
    config = {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path), "geoscd_root": str(root),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "sam_checkpoint": str(sam_checkpoint),
              "sam_checkpoint_sha256": sha256(sam_checkpoint), "device": device, "pairs": pairs, "limit": limit, "split": split,
              "seed": seed, "mode": mode, "points_per_side": points_per_side, "iou_thresh": iou_thresh, "sem_filter": sem_filter}
    out = new_directory(out)
    record = run_record("geoscd_full_batch", config)
    rows = []
    write_json(out / "run.json", record)
    write_json(out / "selected_pairs.json", selected)
    try:
        record["upstream"] = source_provenance(root)
        try:
            record["gpu_before"] = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            record["gpu_before"] = f"Unavailable: {type(exc).__name__}"
        backend = _backend if _backend is not None else OfficialFull(root, checkpoint, sam_checkpoint, device, seed,
                                                                    mode, points_per_side, iou_thresh, sem_filter)
        for path, key in ((checkpoint, "checkpoint_sha256"), (sam_checkpoint, "sam_checkpoint_sha256")):
            if sha256(path) != config[key]:
                raise ValueError("A model checkpoint changed while loading")
        record["backend"] = getattr(backend, "metadata", {"id": "injected CPU contract backend"})
        write_json(out / "run.json", record)
        with (out / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "reference_file", "source_file", "review_status", "notes"])
            writer.writeheader()
            writer.writerows({**{key: pair[key] for key in ("pair_id", "reference_file", "source_file")}, "review_status": "pending", "notes": ""} for pair in selected)
        for index, pair in enumerate(selected, 1):
            print(f"Full GeoSCD {index}/{len(selected)}: {pair['reference_file']} → {pair['source_file']}", flush=True)
            row = {**pair, "method": "geoscd_full", "path": f"geoscd-full/pair-{pair['pair_id']}"}
            child = new_directory(out / row["path"])
            child_record = run_record("geoscd_full_pair", {**config, "reference_id": str(pair["reference_id"]), "source_id": str(pair["source_id"])})
            child_record.update(upstream=record["upstream"], backend=record["backend"])
            reference, source = images[str(pair["reference_id"])], images[str(pair["source_id"])]
            child_record["observations"] = {"reference": reference, "source": source}
            write_json(child / "run.json", child_record)
            start = time.monotonic()
            try:
                arrays = []
                for image in (reference, source):
                    if sha256(image["image_path"]) != image["sha256"]:
                        raise ValueError(f"Image changed since prepared manifest: {image['file_name']}")
                    rgb, opaque = load_rgb(image["image_path"])
                    if rgb.shape[:2] != (image["height"], image["width"]):
                        raise ValueError("Decoded dimensions differ from prepared native grid")
                    arrays.extend((rgb, opaque))
                prediction = backend(reference["image_path"], source["image_path"])
                for image in (reference, source):
                    if sha256(image["image_path"]) != image["sha256"]:
                        raise ValueError("Original image changed during full GeoSCD inference")
                row.update(status="computed_needs_review", diagnostics=_save_prediction(child, reference, source, arrays, prediction))
                row["diagnostics"]["elapsed_seconds_with_artifacts"] = time.monotonic() - start
                write_json(child / "diagnostics.json", row["diagnostics"])
                finish_record(child, child_record, "completed_needs_review")
            except KeyboardInterrupt:
                finish_record(child, child_record, "interrupted", "Interrupted by user")
                raise
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                finish_record(child, child_record, "failed", row["error"])
            rows.append(row)
            print(f"  geoscd_full: {row['status']}{'; ' + row['error'] if 'error' in row else ''}", flush=True)
            summary = _write_outputs(out, selected, rows)
        if source_provenance(root) != record["upstream"]:
            raise ValueError("Pinned GeoSCD source changed during inference")
        record["summary"] = summary
        finish_record(out, record, "completed_with_issues" if summary["failed_runs"] else "completed_needs_review")
        return summary
    except KeyboardInterrupt:
        _write_outputs(out, selected, rows)
        finish_record(out, record, "interrupted", "Interrupted by user; completed predictions preserved")
        raise
    except Exception as exc:
        _write_outputs(out, selected, rows)
        finish_record(out, record, "failed", f"{type(exc).__name__}: {exc}")
        raise
