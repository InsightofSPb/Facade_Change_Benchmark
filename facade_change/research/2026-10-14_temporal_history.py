"""Causal, nonlearned exemplars for the current local visual-state episode.

Each exemplar is an actual prior RGB observation with its own local validity
mask. Accepted changes invalidate earlier exemplars at those pixels, including
when the new observation resembles an older, pre-damage image. No annotations,
event labels, semantic masks, or future observations enter this module.

``max_exemplars`` bounds valid exemplars per pixel, not retained full frames.
Sparse visibility can retain more full frames; this avoids synthesizing RGB
reference mosaics, whose boundaries would alter contextual scorers.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from ..methods.base import integer_option, rgb_images


def _threshold(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError("Memory update threshold must be finite and nonnegative")
    return float(value)


def _readonly(array):
    view = array.view()
    view.flags.writeable = False
    return view


class TemporalMemory:
    """Keep up to K recent observations of each pixel's current local episode.

    Frame indices start at zero and advance on every successful call, including
    an observation with empty geometric support. All frames must already share
    one fixed aligned grid. The externally calibrated update threshold is fixed
    for this memory's lifetime and survives serialization.
    """

    def __init__(self, max_exemplars=3):
        self.max_exemplars = integer_option(max_exemplars, "max_exemplars", 1)
        self._shape = None
        self._next_frame = 0
        self._threshold = None
        self._rgb = []
        self._valid = []
        self._frame_ids = []
        self._episode_start = None

    def score_then_update(self, rgb, support, scorer, threshold):
        """Score current RGB against past exemplars, then admit the observation.

        ``scorer(reference_rgb, current_rgb, overlap)`` returns an HxW numeric
        score array, or ``(scores, metadata)``. Scores must be nonnegative where
        finite. Nonfinite scores are unavailable comparisons. The history score
        is the minimum finite score among current-episode exemplars at a pixel;
        ties select the oldest eligible frame. Reference RGB is never composited.

        ``support`` is geometric support, never an evaluation/GT visibility mask.
        Newly observed pixels initialize without a score or event. A scored
        pixel changes exactly when ``score > threshold``; all its previous
        exemplars are then invalidated. Unchanged scored pixels add an exemplar.
        Previously observed pixels with no finite comparator do not update.

        The returned ``support`` denotes finite scores, not observed coverage.
        Metadata is returned for the caller's provenance, not stored in memory.
        Scoring exceptions leave the previous memory unchanged.
        """
        rgb, _ = rgb_images(rgb, rgb)
        support = np.asarray(support)
        shape = rgb.shape[:2]
        if support.dtype != np.bool_ or support.shape != shape:
            raise ValueError("Memory requires same-grid boolean geometric support")
        if self._shape is not None and shape != self._shape:
            raise ValueError("Memory observations must share one fixed aligned RGB grid")
        threshold = _threshold(threshold)
        if self._threshold is not None and threshold != self._threshold:
            raise ValueError("Memory update threshold must remain fixed across frames")
        if not callable(scorer):
            raise TypeError("scorer must be callable")

        known = np.zeros(shape, bool)
        scores = np.full(shape, np.inf, np.float64)
        reference_frame = np.full(shape, -1, np.int64)
        winning_comparison = np.full(shape, -1, np.int32)
        reference_count = np.zeros(shape, np.int32)
        comparisons = []
        for image, valid, frame in zip(self._rgb, self._valid, self._frame_ids):
            known |= valid
            overlap = support & valid
            if not overlap.any():
                continue
            result = scorer(_readonly(image), _readonly(rgb), _readonly(overlap))
            values, metadata = result if isinstance(result, tuple) else (result, None)
            values = np.asarray(values)
            if values.shape != shape or values.dtype.kind not in "fiu":
                raise ValueError("History scorer must return a same-grid numeric HxW map")
            finite = overlap & np.isfinite(values)
            if np.any(values[finite] < 0):
                raise ValueError("History scores must be nonnegative where supported")
            reference_count += finite
            improved = finite & (values < scores)
            scores[improved] = values[improved]
            reference_frame[improved] = frame
            winning_comparison[improved] = len(comparisons)
            comparisons.append({"reference_frame": frame, "metadata": metadata})

        scored = reference_count > 0
        initialized = support & ~known
        accepted = scored & (scores > threshold)
        admitted = initialized | scored
        scores[~scored] = np.nan

        # Construct the next state after all callbacks succeed. Current RGB has
        # never been exposed as a reference to the current scoring operation.
        next_valid = [valid & ~accepted for valid in self._valid]
        next_rgb = list(self._rgb)
        next_frames = list(self._frame_ids)
        frame_index = self._next_frame
        if admitted.any():
            next_rgb.append(rgb.copy())
            next_valid.append(admitted.copy())
            next_frames.append(frame_index)
        counts = np.zeros(shape, np.int32)
        for valid in reversed(next_valid):
            valid &= counts < self.max_exemplars
            counts += valid
        keep = [index for index, valid in enumerate(next_valid) if valid.any()]
        episode_start = (np.full(shape, -1, np.int64) if self._episode_start is None
                         else self._episode_start.copy())
        episode_start[initialized | accepted] = frame_index

        self._rgb = [next_rgb[index] for index in keep]
        self._valid = [next_valid[index] for index in keep]
        self._frame_ids = [next_frames[index] for index in keep]
        self._shape = shape
        self._threshold = threshold
        self._episode_start = episode_start
        self._next_frame += 1
        return {
            "scores": scores, "support": scored,
            "accepted_change": accepted, "initialized_support": initialized,
            "reference_frame": reference_frame, "reference_count": reference_count,
            "winning_comparison": winning_comparison,
            "episode_start": episode_start.copy(), "frame_index": frame_index,
            "comparisons": comparisons,
        }

    def state_dict(self):
        """Return independent NumPy arrays suitable for pickle-free np.savez."""
        shape = self._shape or (0, 0)
        return {
            "schema_version": np.array(1, np.int64),
            "max_exemplars": np.array(self.max_exemplars, np.int64),
            "next_frame": np.array(self._next_frame, np.int64),
            "threshold": np.array(np.nan if self._threshold is None else self._threshold),
            "shape": np.array(self._shape or (), np.int64),
            "rgb": (np.stack(self._rgb) if self._rgb
                    else np.empty((0, *shape, 3), np.uint8)),
            "valid": (np.stack(self._valid) if self._valid
                      else np.empty((0, *shape), bool)),
            "frame_ids": np.array(self._frame_ids, np.int64),
            "episode_start": (self._episode_start.copy() if self._episode_start is not None
                              else np.empty(shape, np.int64)),
        }

    @classmethod
    def from_state_dict(cls, state):
        """Validate and restore state without trusting Python pickle payloads."""
        keys = {"schema_version", "max_exemplars", "next_frame", "threshold",
                "shape", "rgb", "valid", "frame_ids", "episode_start"}
        if not isinstance(state, Mapping) or set(state) != keys:
            raise ValueError("Temporal memory state has missing or unexpected fields")
        state = {key: np.asarray(value) for key, value in state.items()}
        for key in ("schema_version", "max_exemplars", "next_frame"):
            if state[key].shape != () or state[key].dtype.kind not in "iu":
                raise ValueError(f"Invalid integer scalar in memory state: {key}")
        if int(state["schema_version"]) != 1:
            raise ValueError("Unsupported temporal memory state schema")
        memory = cls(int(state["max_exemplars"]))
        next_frame = int(state["next_frame"])
        if next_frame < 0:
            raise ValueError("Invalid memory frame counter")
        shape = state["shape"]
        if shape.dtype.kind not in "iu" or shape.ndim != 1 or shape.size not in (0, 2):
            raise ValueError("Invalid memory grid shape")
        shape = tuple(int(value) for value in shape)
        if shape and min(shape) < 1:
            raise ValueError("Invalid memory grid dimensions")
        threshold = state["threshold"]
        if threshold.shape != () or threshold.dtype.kind != "f":
            raise ValueError("Invalid memory threshold scalar")
        if next_frame == 0:
            if shape or not np.isnan(threshold):
                raise ValueError("Uninitialized memory must have no grid or threshold")
        elif not shape:
            raise ValueError("Initialized memory must have a grid")
        else:
            memory._threshold = _threshold(float(threshold))
        grid = shape or (0, 0)
        frames, valid, images = state["frame_ids"], state["valid"], state["rgb"]
        count = frames.size
        if (frames.ndim != 1 or frames.dtype.kind not in "iu"
                or np.any(frames < 0) or np.any(frames >= next_frame)
                or np.any(frames[1:] <= frames[:-1])):
            raise ValueError("Memory frame indices must be unique, increasing, and past")
        if images.dtype != np.uint8 or images.shape != (count, *grid, 3):
            raise ValueError("Invalid saved RGB exemplar array")
        if valid.dtype != np.bool_ or valid.shape != (count, *grid):
            raise ValueError("Invalid saved exemplar support array")
        if count and (not valid.reshape(count, -1).any(axis=1).all()
                      or np.any(valid.sum(axis=0) > memory.max_exemplars)):
            raise ValueError("Saved exemplars violate nonempty support or memory capacity")
        episode = state["episode_start"]
        if episode.dtype.kind != "i" or episode.shape != grid:
            raise ValueError("Invalid local episode start array")
        known = valid.any(axis=0)
        if np.any(episode[~known] != -1) or np.any(episode[known] < 0):
            raise ValueError("Saved episodes disagree with observed support")
        for frame, mask in zip(frames, valid):
            if np.any(episode[mask] > frame):
                raise ValueError("An exemplar predates its active local episode")
        memory._shape = shape or None
        memory._next_frame = next_frame
        memory._rgb = [image.copy() for image in images]
        memory._valid = [mask.copy() for mask in valid]
        memory._frame_ids = [int(frame) for frame in frames]
        memory._episode_start = episode.copy() if shape else None
        return memory
