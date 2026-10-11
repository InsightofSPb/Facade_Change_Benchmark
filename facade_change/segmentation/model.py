"""Frozen DINOv2 with a small, supervised multi-label segmentation head.

The head uses the ordinary BatchNorm + 1x1 classifier pattern of DINOv2's
linear segmentation probes, without importing the legacy mmseg dependency.
This is a project baseline, not a reproduction of the author's training recipe:
we use single-device BatchNorm, overlapping COCO labels and masked BCE + Dice.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenDinoSegmenter(nn.Module):
    """Train only a linear head on concatenated final DINOv2 feature maps.

    Inputs are RGB floats in [0, 1] on the stored image grid. The backbone sees
    ImageNet normalization and bottom/right padding to its patch size. Logits
    are interpolated to the padded grid before that padding is removed.
    """

    def __init__(self, backbone, num_classes, intermediate_layers=4):
        super().__init__()
        if num_classes < 1 or intermediate_layers < 1:
            raise ValueError("num_classes and intermediate_layers must be positive")
        patch_size = getattr(backbone, "patch_size", None)
        if patch_size != 14:
            raise ValueError("Expected a DINOv2 backbone with patch size 14")
        if not callable(getattr(backbone, "get_intermediate_layers", None)):
            raise TypeError("Backbone must provide get_intermediate_layers")
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.intermediate_layers = int(intermediate_layers)
        self.patch_size = int(patch_size)
        self.feature_channels = int(backbone.embed_dim) * self.intermediate_layers
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.head = nn.Sequential(
            nn.BatchNorm2d(self.feature_channels),
            nn.Conv2d(self.feature_channels, self.num_classes, kernel_size=1),
        )
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Calling model.train() must not activate dropout or update any
        # normalization state in the pretrained feature extractor.
        self.backbone.eval()
        return self

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected floating RGB tensor with shape B,3,H,W")
        height, width = rgb.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("RGB dimensions must be nonzero")
        value = (rgb - self.rgb_mean) / self.rgb_std
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        # Reflection requires the padding to be smaller than the dimension;
        # replication also supports tiny synthetic smoke-test images.
        if pad_w:
            value = F.pad(value, (0, pad_w, 0, 0),
                          mode="reflect" if pad_w < width else "replicate")
        if pad_h:
            value = F.pad(value, (0, 0, 0, pad_h),
                          mode="reflect" if pad_h < height else "replicate")
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                value, n=self.intermediate_layers, reshape=True)
        expected_shape = (rgb.shape[0], int(self.backbone.embed_dim),
                          value.shape[-2] // self.patch_size,
                          value.shape[-1] // self.patch_size)
        if (len(features) != self.intermediate_layers
                or any(tuple(feature.shape) != expected_shape for feature in features)):
            raise RuntimeError("Unexpected DINOv2 intermediate feature maps")
        logits = self.head(torch.cat(features, dim=1))
        logits = F.interpolate(logits, size=value.shape[-2:], mode="bilinear",
                               align_corners=False)
        return logits[..., :height, :width]


def load_model(source_root, checkpoint_path, num_classes, device,
               trust_checkpoint=False):
    """Load local author weights with the existing restricted-loading policy."""
    from ..methods.dinov2 import _load_dino

    backbone, provenance = _load_dino(source_root, checkpoint_path, trust_checkpoint)
    model = FrozenDinoSegmenter(backbone, num_classes).to(device)
    return model, {
        **provenance,
        "backbone": "dinov2_vits14",
        "backbone_frozen": True,
        "features": "last_four_normalized_intermediate_patch_maps",
        "head": "BatchNorm2d + Conv2d(kernel_size=1)",
        "feature_channels": model.feature_channels,
        "num_classes": int(num_classes),
        "prediction": "independent sigmoid probabilities; fixed threshold 0.5",
        "loss": "0.5 masked BCE + 0.5 positive-class soft Dice",
        "input_preparation": "ImageNet normalization; bottom/right reflect padding to 14 (replicate for tiny dimensions)",
        "output_preparation": "bilinear logits to padded RGB grid, then remove padding",
        "downloads": False,
        "device": str(device),
    }


def masked_multilabel_loss(logits, targets, valid):
    """Equal-weight BCE and soft Dice on annotated, opaque, non-padded support.

    Each class is an independent channel, so intersecting annotations remain
    positive in all applicable channels. Other classes on annotated support are
    supervised as negative; pixels outside that support provide no supervision.
    Dice averages only classes with positive valid targets in the batch. BCE
    still penalizes false positives of absent classes on valid support.
    An all-invalid batch returns a graph-connected zero.
    """
    if logits.ndim != 4 or targets.shape != logits.shape:
        raise ValueError("logits and targets must share shape B,C,H,W")
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if tuple(valid.shape) != (logits.shape[0], 1, *logits.shape[-2:]):
        raise ValueError("valid must have shape B,H,W or B,1,H,W")
    # Accumulate outside half precision, even when the forward pass uses AMP.
    scores = logits.float()
    truth = targets.to(device=logits.device, dtype=torch.float32)
    mask = valid.to(device=logits.device, dtype=torch.bool).float()
    valid_elements = mask.sum() * logits.shape[1]
    bce = (F.binary_cross_entropy_with_logits(scores, truth, reduction="none")
           * mask).sum() / valid_elements.clamp_min(1.0)
    probabilities = scores.sigmoid() * mask
    masked_truth = truth * mask
    axes = (0, 2, 3)
    positive_count = masked_truth.sum(dim=axes)
    intersection = (probabilities * masked_truth).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + positive_count
    per_class_dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    present = (positive_count > 0).float()
    dice = (per_class_dice_loss * present).sum() / present.sum().clamp_min(1.0)
    return 0.5 * bce + 0.5 * dice
