"""Coordinate names and owners shared by agentic producers and evaluation.

Only ``canonical_crop`` is an alias: it names the same fixed region pixel grid
as ``region_crop_pixels``. Native frames are never renamed or transformed here.
An observation frame needs its observation ID; an anchor frame needs its image
hash. A region grid can inherit the current case's region ID for legacy labels.
"""
from __future__ import annotations


REGION_CROP = "region_crop_pixels"


def normalize_coordinate_system(value):
    if not isinstance(value, str) or not value:
        raise ValueError("An explicit coordinate system is required")
    return REGION_CROP if value == "canonical_crop" else value


def coordinate_frame(record, *, default_region_id=None):
    """Return a frame name and its owner, without applying any transformation.

    Other legacy, explicitly named grids retain their literal identity. Missing
    owners on the two generic native frames cannot identify an image uniquely.
    """
    frame = normalize_coordinate_system(record.get("coordinate_system"))
    if frame == "observation_native":
        owner = record.get("observation_id")
    elif frame == "anchor_native":
        owner = record.get("anchor_sha256")
    elif frame == REGION_CROP:
        owner = record.get("region_id", default_region_id)
    else:
        owner = None
    if frame in {"observation_native", "anchor_native"} and (not isinstance(owner, str) or not owner):
        raise ValueError("An explicit native coordinate owner is required")
    if owner is not None and (not isinstance(owner, str) or not owner):
        raise ValueError("Invalid coordinate owner")
    if frame == "anchor_native" and (len(owner) != 64 or any(c not in "0123456789abcdef" for c in owner)):
        raise ValueError("The anchor coordinate owner must be its SHA-256")
    return frame, owner


def coordinate_mismatch(prediction, target, *, region_id=None):
    """Return a stable reason for incompatible frames, otherwise ``None``."""
    try:
        left_name = normalize_coordinate_system(prediction.get("coordinate_system"))
        right_name = normalize_coordinate_system(target.get("coordinate_system"))
    except ValueError:
        return "coordinate_system_mismatch"
    if left_name != right_name:
        return "coordinate_system_mismatch"
    try:
        left = coordinate_frame(prediction, default_region_id=region_id)
        right = coordinate_frame(target, default_region_id=region_id)
    except ValueError:
        return "coordinate_owner_required"
    if left != right:
        return "coordinate_owner_mismatch"
    return None
