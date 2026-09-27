"""Pixel-centre geometry; H always maps source-native to reference-native."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    reference_to_canvas: np.ndarray
    source_to_reference: np.ndarray

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "reference_to_canvas": self.reference_to_canvas.tolist(),
                "source_to_reference": self.source_to_reference.tolist(),
                "source_to_canvas": (self.reference_to_canvas @ self.source_to_reference).tolist(),
                "coordinates": "pixel centres; x=column, y=row; native reference pixel scale"}


def proxy_transform(native_shape, proxy_shape) -> np.ndarray:
    """Adapted from LPOSS: actual rounded sizes and half-pixel resize offsets."""
    sy, sx = np.array(proxy_shape[:2]) / np.array(native_shape[:2])
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=float)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=float).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Nonfinite point or projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def image_edges(shape) -> np.ndarray:
    h, w = shape[:2]
    if min(h, w) < 1:
        raise ValueError("Empty image")
    return np.array([[-.5, -.5], [w - .5, -.5], [w - .5, h - .5], [-.5, h - .5]])


def expanded_canvas(reference_shape, source_shape, source_to_reference,
                    max_pixels: int = 50_000_000, max_side: int = 16000) -> Canvas:
    matrix = np.asarray(source_to_reference, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.any(matrix):
        raise ValueError("H must be a finite nonzero 3x3 matrix")
    matrix = matrix / np.max(np.abs(matrix))
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Singular homography")
    source_edges = image_edges(source_shape)
    denominators = np.column_stack([source_edges, np.ones(4)]) @ matrix[2]
    if np.min(np.abs(denominators)) < 1e-10 or np.min(denominators) * np.max(denominators) <= 0:
        raise ValueError("Projective horizon intersects source image")
    corners = np.vstack([image_edges(reference_shape), transform_points(source_edges, matrix)])
    low = np.floor(corners.min(axis=0) + .5 + 1e-9)
    high = np.ceil(corners.max(axis=0) + .5 - 1e-9)
    width, height = map(int, high - low)
    if min(width, height) <= 0 or max(width, height) > max_side or width * height > max_pixels:
        raise ValueError(f"Unsafe expanded canvas {width}x{height}; inspect H, do not downscale silently")
    shift = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], dtype=float)
    if abs(matrix[2, 2]) > 1e-10:
        matrix = matrix / matrix[2, 2]
    return Canvas(width, height, shift, matrix)


def warp_pair(reference_rgb, reference_opaque, source_rgb, source_opaque, canvas: Canvas) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the align extra: python -m pip install -e '.[align]'") from exc
    result = {}
    for label, rgb, opaque, matrix in (
        ("reference", reference_rgb, reference_opaque, canvas.reference_to_canvas),
        ("source", source_rgb, source_opaque, canvas.reference_to_canvas @ canvas.source_to_reference),
    ):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or opaque.shape != rgb.shape[:2]:
            raise ValueError("Expected native uint8 RGB and same-grid opacity mask")
        size = (canvas.width, canvas.height)
        result[label] = cv2.warpPerspective(rgb, matrix, size, flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        coverage = cv2.warpPerspective(opaque.astype(np.float32), matrix, size, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result[label + "_support"] = coverage >= 1 - 1e-6
    ref, src = result["reference_support"], result["source_support"]
    result.update(overlap=ref & src, reference_only=ref & ~src, source_only=src & ~ref)
    return result


def absolute_residual(current_rgb, previous_rgb) -> np.ndarray:
    """MasksComp arithmetic; avoid uint8 wraparound before absolute value."""
    if current_rgb.shape != previous_rgb.shape or current_rgb.dtype != np.uint8 or previous_rgb.dtype != np.uint8:
        raise ValueError("Residual expects matching uint8 RGB arrays")
    if current_rgb.ndim != 3 or current_rgb.shape[2] != 3:
        raise ValueError("Residual expects three RGB channels")
    return np.abs(current_rgb.astype(np.int16) - previous_rgb.astype(np.int16)).astype(np.uint8)
