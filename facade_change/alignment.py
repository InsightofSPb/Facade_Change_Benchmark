"""Small interchangeable SIFT/LoFTR matchers, shared native-coordinate estimator."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Protocol
import numpy as np

from .geometry import proxy_transform, transform_points
from .io import sha256


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV missing: python -m pip install -e '.[align]'") from exc
    return cv2


def make_proxy(rgb, opaque, max_side):
    cv2 = opencv()
    if max_side < 8:
        raise ValueError("max_side must be >=8")
    h, w = rgb.shape[:2]
    scale = min(1., max_side / max(h, w))
    shape = (max(1, round(h * scale)), max(1, round(w * scale)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if shape != (h, w):
        gray = cv2.resize(gray, shape[::-1], interpolation=cv2.INTER_AREA)
        opaque = cv2.resize(opaque.astype(np.float32), shape[::-1], interpolation=cv2.INTER_AREA) >= 1 - 1e-6
    return gray, opaque, proxy_transform((h, w), shape)


def valid_locations(points, support):
    points = np.asarray(points).reshape(-1, 2)
    h, w = support.shape
    keep = np.isfinite(points).all(axis=1)
    keep &= (points[:, 0] >= 0) & (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
    idx = np.flatnonzero(keep)
    xy = np.rint(points[idx]).astype(int)
    keep[idx] &= support[xy[:, 1], xy[:, 0]]
    return keep


def ratio_matches(neighbors, ratio=.75):
    """OpenCV kNN can yield fewer than two neighbors for an entry."""
    return [pair[0] for pair in neighbors if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance]


@dataclass
class Matches:
    source: np.ndarray
    reference: np.ndarray
    metadata: dict


class Matcher(Protocol):
    def match(self, source_gray, reference_gray, source_support, reference_support) -> Matches: ...


class SIFTMatcher:
    def match(self, source_gray, reference_gray, source_support, reference_support) -> Matches:
        cv2 = opencv()
        detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.01, edgeThreshold=10)
        ks, ds = detector.detectAndCompute(source_gray, source_support.astype(np.uint8) * 255)
        kr, dr = detector.detectAndCompute(reference_gray, reference_support.astype(np.uint8) * 255)
        good = [] if ds is None or dr is None or len(dr) < 2 else ratio_matches(
            cv2.BFMatcher(cv2.NORM_L2).knnMatch(ds, dr, k=2))
        source = np.array([ks[m.queryIdx].pt for m in good], dtype=float).reshape(-1, 2)
        reference = np.array([kr[m.trainIdx].pt for m in good], dtype=float).reshape(-1, 2)
        keep = valid_locations(source, source_support) & valid_locations(reference, reference_support)
        return Matches(source[keep], reference[keep], {"id": "sift-v1", "ratio": .75,
                       "nfeatures": 8000, "contrast_threshold": .01, "edge_threshold": 10})


class LoFTRMatcher:
    def __init__(self, checkpoint, device="cpu", confidence=.4):
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("LoFTR requires --checkpoint pointing to a local compatible state_dict; no automatic download")
        if not 0 <= confidence <= 1:
            raise ValueError("LoFTR confidence must be in [0,1]")
        try:
            import torch
            from kornia.feature import LoFTR
        except ImportError as exc:
            raise RuntimeError("LoFTR requires torch and kornia; install in the local GPU environment") from exc
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("Device must be cpu or cuda[:index]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise RuntimeError("This LoFTR loader needs torch.load(weights_only=True); select a compatible existing environment. CPU manifest/SIFT do not need Torch.")
        self.torch, self.device, self.confidence = torch, device, confidence
        digest = sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        self.model = LoFTR(pretrained=None)
        self.model.load_state_dict(state, strict=True)
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint changed while loading")
        self.metadata = {"id": "kornia-loftr-v1", "checkpoint_sha256": digest,
                         "checkpoint_path": str(Path(checkpoint).resolve()), "device": device,
                         "confidence_threshold": confidence, "checkpoint_load": "strict; weights_only"}
        if device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(device)
            self.metadata["cuda_memory_before_model_bytes"] = {"free": free, "total": total}
            self.metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        self.model = self.model.to(device).eval()

    def match(self, source_gray, reference_gray, source_support, reference_support) -> Matches:
        torch = self.torch
        inputs = {}
        for i, (gray, support) in enumerate(((source_gray, source_support), (reference_gray, reference_support))):
            padding = ((0, (-gray.shape[0]) % 8), (0, (-gray.shape[1]) % 8))
            inputs[f"image{i}"] = torch.from_numpy(np.pad(gray, padding)).float()[None, None].to(self.device) / 255
            inputs[f"mask{i}"] = torch.from_numpy(np.pad(support, padding)).bool()[None].to(self.device)
        with torch.inference_mode():
            result = self.model(inputs)
        source = result["keypoints0"].detach().cpu().numpy()
        reference = result["keypoints1"].detach().cpu().numpy()
        confidence = result["confidence"].detach().cpu().numpy()
        keep = (confidence >= self.confidence) & valid_locations(source, source_support) & valid_locations(reference, reference_support)
        return Matches(source[keep], reference[keep], self.metadata)


@dataclass
class Alignment:
    source_to_reference: np.ndarray
    matches: Matches
    inliers: np.ndarray
    diagnostics: dict


def align(reference_rgb, reference_opaque, source_rgb, source_opaque, matcher: Matcher,
          max_side=1024, ransac_threshold=3., seed=42) -> Alignment:
    cv2 = opencv()
    if not np.isfinite(ransac_threshold) or ransac_threshold <= 0:
        raise ValueError("RANSAC threshold must be finite and positive")
    cv2.setRNGSeed(int(seed))
    cv2.setNumThreads(1)
    if not hasattr(cv2, "USAC_MAGSAC"):
        raise RuntimeError("This pipeline requires OpenCV USAC_MAGSAC")
    gs, ms, ps = make_proxy(source_rgb, source_opaque, max_side)
    gr, mr, pr = make_proxy(reference_rgb, reference_opaque, max_side)
    matches = matcher.match(gs, gr, ms, mr)
    if len(matches.source) != len(matches.reference) or len(matches.source) < 8:
        raise ValueError(f"Insufficient correspondences: {len(matches.source)}; need at least 8")
    matches.source = transform_points(matches.source, np.linalg.inv(ps))
    matches.reference = transform_points(matches.reference, np.linalg.inv(pr))
    h, mask = cv2.findHomography(matches.source, matches.reference, cv2.USAC_MAGSAC,
                                 ransacReprojThreshold=ransac_threshold, confidence=.999, maxIters=10000)
    if h is None or mask is None or not np.isfinite(h).all() or int(mask.sum()) < 4:
        raise ValueError("Homography estimation failed or has fewer than 4 inliers")
    inliers = mask.ravel().astype(bool)
    errors = np.linalg.norm(transform_points(matches.source, h) - matches.reference, axis=1)
    def coverage(points, shape):
        hull = cv2.convexHull(points.astype(np.float32))
        return float(cv2.contourArea(hull) / (shape[0] * shape[1]))
    diagnostics = {"opencv_runtime_version": cv2.__version__, "matches": len(inliers),
                   "inliers": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
                   "inlier_reprojection_median_px": float(np.median(errors[inliers])),
                   "inlier_reprojection_p95_px": float(np.percentile(errors[inliers], 95)),
                   "source_inlier_hull_fraction": coverage(matches.source[inliers], source_rgb.shape),
                   "reference_inlier_hull_fraction": coverage(matches.reference[inliers], reference_rgb.shape),
                   "source_native_to_proxy": ps.tolist(), "reference_native_to_proxy": pr.tolist(),
                   "source_proxy_shape": list(gs.shape), "reference_proxy_shape": list(gr.shape),
                   "ransac": {"method": "USAC_MAGSAC", "threshold_native_reference_px": ransac_threshold,
                              "confidence": .999, "max_iters": 10000, "seed": seed},
                   "review_status": "needs_visual_review"}
    return Alignment(h, matches, inliers, diagnostics)
