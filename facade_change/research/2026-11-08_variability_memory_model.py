"""Causal residual heads: past variability refines an explicit last-pair signal."""
from __future__ import annotations

import importlib

import torch
from torch import nn
from torch.nn import functional as F

ConvGRUCell = importlib.import_module(
    "facade_change.research.2026-10-25_spatial_change_model").ConvGRUCell
MODES = ("pair", "mean", "convgru", "convgru-last")


class VariabilityMemory(nn.Module):
    """History is context, never a hard veto on the current transition.

    Recompute the causal prefix for each event. A frame enters the memory of a
    later event only after its own transition has been evaluated. No GT, future
    image, or current image is passed to past_memory.
    """

    def __init__(self, channels=384, hidden=32, mode="pair", methods=7):
        super().__init__()
        if mode not in MODES or min(channels, hidden, methods) < 1:
            raise ValueError("Invalid memory mode or dimensions")
        self.mode, self.hidden, self.methods = mode, hidden, methods
        self.project = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU())
        self.transition = nn.Sequential(nn.Conv2d(2 * hidden, hidden, 1), nn.GELU())
        self.cell = ConvGRUCell(hidden)
        self.decoder = nn.Sequential(nn.Conv2d(4 * hidden + methods + 4, hidden, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(hidden, 1, 1))
        # Every head starts as exactly the frozen stack, not a random replacement.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def past_memory(self, past, support, years, current_year):
        if len(past) < 1:
            raise ValueError("Memory needs at least the previous observation")
        state = torch.zeros_like(past[-1:])
        available = torch.zeros_like(support[-1:])
        if self.mode == "pair":
            return state, available
        if self.mode == "convgru-last":
            # Same recurrence, parameters, and timestamps; no older image content.
            past = past[-1:].expand_as(past)
            support = support[-1:].expand_as(support)
        if self.mode == "mean":
            valid = support[1:] & support[:-1]
            delta = (past[1:] - past[:-1]).abs() * valid
            return delta.sum(0, keepdim=True) / valid.sum(0, keepdim=True).clamp_min(1), valid.any(0, keepdim=True)
        for index in range(len(past)):
            valid = support[index:index + 1]
            change = (past[index:index + 1] - past[index - 1:index]).abs() if index else state * 0
            if index:
                change = change * (valid & support[index - 1:index])
            value = self.transition(torch.cat((past[index:index + 1], change), 1))
            gap = years[index] - years[index - 1] if index else years.new_zeros(())
            age = current_year - years[index]
            time = torch.log1p(torch.stack((gap, age)).float()).view(1, 2, 1, 1)
            proposal = self.cell(value, state, time.expand(1, 2, *state.shape[-2:]))
            state = torch.where(valid, proposal, state)
            available = available | valid
        return state, available

    def forward(self, features, support, years, pair_scores):
        if (features.ndim != 4 or len(features) < 2
                or support.shape != (len(features), 1, *features.shape[-2:])
                or years.shape != (len(features),) or torch.any(years[1:] <= years[:-1])
                or pair_scores.ndim != 3 or pair_scores.shape[0] != self.methods):
            raise ValueError("Expected chronological features/support/years and last-pair scores")
        support = support.bool()
        features = F.normalize(features.masked_fill(~support, 0), dim=1)
        values = self.project(features) * support
        memory, available = self.past_memory(values[:-1], support[:-1], years[:-1], years[-1])
        previous, current = values[-2:-1], values[-1:]
        scores = F.adaptive_avg_pool2d(pair_scores[None], features.shape[-2:])
        gap = torch.log1p((years[-1] - years[-2]).float()).expand(1, 1, *features.shape[-2:])
        combined = torch.cat((previous, current, (current - previous).abs(), memory, scores,
                              support[-2:-1].float(), support[-1:].float(), available.float(), gap), 1)
        correction = self.decoder(combined)
        return F.interpolate(correction, size=pair_scores.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
