"""Full author GeoSCD mask on the input RGB grid, without confidence reinterpretation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..geoscd import source_provenance
from ..io import sha256
from .base import validate_rgb_inputs


class GeoSCDScorer:
    """Adapt OfficialFull paths to the label-free common RGB/support interface."""

    def __init__(self, source_root=None, checkpoint_path=None, sam_checkpoint=None,
                 device="cuda:0", seed=42, mode="occupy", points_per_side=32,
                 iou_thresh=.65, sem_filter=None, geoscd_root=None,
                 geoscd_checkpoint=None, **options):
        if options:
            raise ValueError("Unknown GeoSCD options: " + ", ".join(sorted(options)))
        source_root = source_root if source_root is not None else geoscd_root
        checkpoint_path = checkpoint_path if checkpoint_path is not None else geoscd_checkpoint
        if source_root is None or checkpoint_path is None or sam_checkpoint is None:
            raise ValueError("GeoSCD requires source_root, checkpoint_path (VGGT), and sam_checkpoint")
        root = Path(source_root).expanduser().resolve()
        checkpoints = {
            "vggt": Path(checkpoint_path).expanduser().resolve(),
            "sam": Path(sam_checkpoint).expanduser().resolve(),
        }
        for role, path in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(f"GeoSCD {role} checkpoint does not exist: {path}")
        provenance = source_provenance(root)
        weight_metadata = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in checkpoints.items()
        }
        # Import the existing full detector only in this method's selected worker.
        from ..geoscd_full import OfficialFull
        self.detector = OfficialFull(root, checkpoints["vggt"], checkpoints["sam"],
                                     device, seed, mode, points_per_side, iou_thresh, sem_filter)
        self.raw_scores = None
        self.native_prediction = None
        self.metadata = {
            **self.detector.metadata,
            "method": "geoscd", "output_kind": "native_mask",
            "source": provenance, "checkpoints": weight_metadata,
            "signal": "unmodified native uint8 RGB saved losslessly as PNG for the author loader",
            "output": "author final_reference_mask; nearest resize to the native reference grid; float32 0/1, NaN outside base support",
            "score": "binary author decision; internal SAM similarity is not exported as confidence",
            "native_mask_grid": "same as supplied reference RGB",
            "evaluation_support": "supplied base geometric support only; no ground-truth visibility or edit labels",
        }

    def __call__(self, reference, source, support):
        reference, source, support = validate_rgb_inputs(reference, source, support)
        with tempfile.TemporaryDirectory(prefix="facade-geoscd-pair-") as folder:
            left, right = Path(folder) / "reference.png", Path(folder) / "source.png"
            Image.fromarray(reference).save(left)
            Image.fromarray(source).save(right)
            prediction = self.detector(left, right)
        mask = np.asarray(prediction["final_reference_mask"])
        if mask.shape != (512, 512):
            raise ValueError("Full GeoSCD must return its 512x512 author final mask")
        h, w = reference.shape[:2]
        native = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
            (w, h), resample=Image.Resampling.NEAREST), dtype=bool)
        self.native_prediction = native.copy()
        self.raw_scores = None
        score = native.astype(np.float32)
        score[~support] = np.nan
        return score

    def close(self):
        detector = getattr(self, "detector", None)
        if detector is None:
            return
        try:
            detector.segmenter.automatic_mask_generator.predictor.reset_image()
            detector.geometry.model.cpu()
            detector.segmenter.cpu()
            detector.sam.cpu()
            if detector.torch.cuda.is_available():
                detector.torch.cuda.empty_cache()
        finally:
            self.detector = None
