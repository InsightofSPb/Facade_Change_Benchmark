"""Small spatial change heads over frozen features; no labels enter forward."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("pair", "mean", "convgru", "convgru-last")


class ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels + 2, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels + 2, channels, 3, padding=1)

    def forward(self, value, state, time):
        reset, update = self.gates(torch.cat((value, state, time), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((value, reset * state, time), 1)).tanh()
        return (1 - update) * state + update * candidate


class SpatialChangeModel(nn.Module):
    """Last pair plus past-only memory, with independent multilabel outputs.

    A future feature adapter replaces Identity and returns the same spatial grid.
    Context/auxiliary maps have their own vocabulary and availability channels.
    The current observation is never passed to the temporal memory.
    """

    def __init__(self, feature_channels, target_channels, mode="pair", hidden=32,
                 extra_channels=0):
        super().__init__()
        if mode not in MODES or min(feature_channels, target_channels, hidden) < 1 or extra_channels < 0:
            raise ValueError("Invalid spatial model dimensions or mode")
        self.spec = dict(feature_channels=feature_channels, target_channels=target_channels,
                         mode=mode, hidden=hidden, extra_channels=extra_channels)
        self.mode = mode
        self.feature_adapter = nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(feature_channels + 2 * extra_channels, hidden, 1), nn.GELU())
        self.memory = ConvGRUCell(hidden) if mode.startswith("convgru") else None
        self.decoder = nn.Sequential(nn.Conv2d(hidden * 4 + 4, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, target_channels, 1))

    def forward(self, features, support, years, output_size, extra=None, available=None):
        if (features.ndim != 4 or features.shape[0] < 2
                or features.shape[1] != self.spec["feature_channels"]
                or support.shape != (features.shape[0], 1, *features.shape[-2:])
                or years.shape != (features.shape[0],) or torch.any(years[1:] <= years[:-1])):
            raise ValueError("Expected chronological T,C,H,W features, T,1,H,W support and T years")
        support = support.bool()
        values = self.feature_adapter(features.masked_fill(~support, 0))
        if values.shape != features.shape:
            raise ValueError("Feature adapter must preserve channels and spatial coordinates")
        count = self.spec["extra_channels"]
        if count:
            expected = (len(features), count, *features.shape[-2:])
            if extra is None or available is None or extra.shape != expected or available.shape != expected:
                raise ValueError("Context requires separate values and per-channel availability")
            values = torch.cat((values, extra.masked_fill(~available.bool(), 0), available.float()), 1)
        elif extra is not None or available is not None:
            raise ValueError("Checkpoint was created without auxiliary/context channels")
        values = self.project(values) * support
        previous, current = values[-2:-1], values[-1:]
        memory = torch.zeros_like(previous)
        memory_valid = torch.zeros_like(support[-1:])
        past, valid = values[:-1], support[:-1]
        if self.mode == "mean":
            memory = past.sum(0, keepdim=True) / valid.sum(0, keepdim=True).clamp_min(1)
            memory_valid = valid.any(0, keepdim=True)
        elif self.memory is not None:
            start = len(past) - 1 if self.mode == "convgru-last" else 0
            for index in range(start, len(past)):
                gap = years[index] - years[index - 1] if index > start else years.new_zeros(())
                age = years[-1] - years[index]
                time = torch.log1p(torch.stack((gap, age)).float()).view(1, 2, 1, 1)
                time = time.expand(1, 2, *features.shape[-2:])
                proposal = self.memory(past[index:index + 1], memory, time)
                mask = valid[index:index + 1]
                memory = torch.where(mask, proposal, memory)
                memory_valid = memory_valid | mask
        pair_gap = torch.log1p((years[-1] - years[-2]).float())
        gap_map = pair_gap.expand(1, 1, *features.shape[-2:])
        combined = torch.cat((previous, current, current - previous, memory,
                              support[-2:-1].float(), support[-1:].float(),
                              memory_valid.float(), gap_map), 1)
        logits = self.decoder(combined)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
