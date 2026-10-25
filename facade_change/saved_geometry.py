"""CPU sampling of recorded homographies and GeoSCD fields; no model loading."""
from __future__ import annotations

import numpy as np

from .alignment_session import artifact, remember
from .geometry import expanded_canvas
from .io import read_json


def frame_geometry(match, fingerprints):
    path, record = match["path"], match["record"]
    geometry_path = artifact(path, record, "geometry.json", fingerprints)
    geometry = read_json(geometry_path)
    observations = record["observations"]
    ref, src = (observations[role] for role in ("reference", "source"))
    extra = {"alignment_run": str(path), "geometry_sha256": fingerprints[str(geometry_path)]}
    if match.get("selection"):
        extra["alignment_selection"] = {key: value for key, value in match["selection"].items() if key != "observations"}
    if geometry.get("type") == "dense_reference_grid":
        if (match.get("selection") or {}).get("review") != "accept":
            raise ValueError("Dense geometry requires explicit hash-bound acceptance")
        if (geometry.get("reference_native_shape") != [ref["height"], ref["width"]]
                or geometry.get("source_native_shape") != [src["height"], src["width"]]
                or geometry.get("model_grid") != [518, 518]
                or geometry.get("direction") != "inverse sampling: reference-native pixel → source-native pixel"
                or geometry.get("predicted_occlusion_used_as_support") is not False):
            raise ValueError("Dense geometry grid/direction/support contract differs")
        field = artifact(path, record, "correspondences.npz", fingerprints)
        return {**extra, "dense_geometry": {"type": "dense_reference_grid", "path": str(field),
                "sha256": fingerprints[str(field)], "reference_shape": [ref["height"], ref["width"]],
                "source_shape": [src["height"], src["width"]], "model_grid": [518, 518]}}
    matrix = np.asarray(geometry["source_to_reference"], dtype=float)
    expanded_canvas((ref["height"], ref["width"]), (src["height"], src["width"]), matrix)
    return {**extra, "source_to_reference": matrix.tolist()}


class FrameSampler:
    """Sample one original frame on the native earliest/reference grid."""

    def __init__(self, frame):
        self.dense = frame.get("dense_geometry")
        if self.dense:
            from .geoscd import native_remap
            info = self.dense
            path = remember(info["path"], {}, info["sha256"])
            with np.load(path, allow_pickle=False) as fields:
                coordinates = fields["reference_to_source"]
                valid = fields["reference_projection_valid"]
                if (coordinates.shape != (518, 518, 2) or valid.shape != (518, 518)
                        or valid.dtype != np.bool_):
                    raise ValueError("Malformed saved GeoSCD correspondence arrays")
                self.map_x, self.map_y, self.valid = native_remap(
                    coordinates, info["reference_shape"], info["source_shape"], valid)
        else:
            self.matrix = np.asarray(frame["source_to_reference"], dtype=float)
            if (self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all()
                    or np.linalg.matrix_rank(self.matrix) != 3):
                raise ValueError("Expected a finite nonsingular homography")

    def _window(self, array, crop, fill):
        x, y, w, h = crop
        result = np.full((h, w), fill, dtype=array.dtype)
        ah, aw = array.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(aw, x+w), min(ah, y+h)
        if x1 > x0 and y1 > y0:
            result[y0-y:y1-y, x0-x:x1-x] = array[y0:y1, x0:x1]
        return result

    def warp(self, array, crop, nearest=False):
        import cv2
        x, y, w, h = crop
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        if self.dense:
            if list(array.shape[:2]) != self.dense["source_shape"]:
                raise ValueError("Original frame dimensions differ from dense geometry")
            return cv2.remap(array, self._window(self.map_x, crop, -1), self._window(self.map_y, crop, -1),
                             flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shift = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
        return cv2.warpPerspective(array, shift @ self.matrix, (w, h), flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def support(self, opaque, crop):
        result = self.warp(opaque.astype(np.float32), crop) >= 1 - 1e-6
        if self.dense:
            result &= self._window(self.valid, crop, False)
        return result
