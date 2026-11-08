"""Causal semantic memory, history-supervised pair learning, and progression proxy.

Semantic probabilities are independent sigmoid channels in the *recorded* ontology
order. Targets are accepted by loss/target builders, never by a model forward.
Progression means a new any-damage episode observed at least twice, not severity.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEMANTIC_MODES = ("independent", "mean", "confidence", "learned", "repeat")
PAIR_MODES = ("pair", "regularized", "history")
PROGRESSION_MODES = ("latest", "first", "history")
DAMAGE_CLASSES = ("WATER_STAIN", "MISSING_ELEMENT", "DELAMINATION", "SPALLING",
                  "CRACK", "EFFLORESCENCE", "CORROSION")
PROGRESSION_RULE = dict(version="2026-11-08-appearance-persistence-v1",
    target_type="annotation-derived progression proxy",
    damage_classes=list(DAMAGE_CLASSES), minimum_observations=3,
    minimum_confirmations=2, missing_policy="unknown if any observation lacks known support",
    repair_policy="REPAIRS ends the previous episode; reappearance starts a new one",
    positive="previously absent any-damage, present in at least two latest observations",
    limitations="Does not establish severity growth, independently verified events, or occlusion absence")


def _support(values, support, padding=None):
    """Normalize B,T,H,W support; padding=True means a valid timestep.

    This follows the shared temporal-model contract: valid observations are a
    prefix and absent right-padding is False.
    """
    if values.ndim != 5:
        raise ValueError("Expected B,T,C,H,W spatial observations")
    b, t, _, h, w = values.shape
    if support.ndim == 4:
        support = support.unsqueeze(2)
    if support.shape != (b, t, 1, h, w):
        raise ValueError("Support must have shape B,T,H,W or B,T,1,H,W")
    result = support.bool()
    if padding is not None:
        if padding.shape != (b, t):
            raise ValueError("Padding must have shape B,T")
        valid = padding.bool()
        if torch.any(valid[:, 1:] & ~valid[:, :-1]) or torch.any(~valid.any(1)):
            raise ValueError("Valid timesteps must be a nonempty prefix before right padding")
        result = result & valid[:, :, None, None, None]
    return result


def _times(times, values, padding=None):
    b, t = values.shape[:2]
    if times.ndim == 1 and b == 1:
        times = times[None]
    if times.shape != (b, t) or not torch.isfinite(times).all():
        raise ValueError("Finite observation times B,T are required")
    valid = torch.ones_like(times, dtype=torch.bool) if padding is None else padding.bool()
    for n in range(b):
        selected = times[n, valid[n]]
        if len(selected) > 1 and torch.any(selected[1:] <= selected[:-1]):
            raise ValueError("Observation times must be strictly chronological")
    return times


def masked_bce(logits, targets, known):
    """Unknown targets never become unchanged; empty supervision gives differentiable zero."""
    if targets.ndim == logits.ndim - 1:
        targets = targets.unsqueeze(1)
    if known.ndim == logits.ndim - 1:
        known = known.unsqueeze(1)
    known = known.bool().expand_as(logits) & (targets >= 0) & (targets <= 1)
    safe = targets.float().masked_fill(~known, 0)
    values = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    return values.masked_fill(~known, 0).sum() / known.sum().clamp_min(1)


class SemanticMemory(nn.Module):
    """Multilabel state filter. Confident current evidence can reset old damage.

    ``repeat`` has the identical learned module, but each output prefix repeats
    its last *past* probabilities into older slots, preserving the current
    observation. The last pair remains intact and the gate remains trainable.
    """

    def __init__(self, classes, mode="learned", hidden=32):
        super().__init__()
        if mode not in SEMANTIC_MODES or classes < 1 or hidden < 1:
            raise ValueError("Invalid semantic memory mode or dimensions")
        self.classes, self.mode = int(classes), mode
        if mode in ("learned", "repeat"):
            self.update = nn.Sequential(nn.Conv2d(4 * classes + 2, hidden, 1),
                                        nn.GELU(), nn.Conv2d(hidden, classes, 1))
            nn.init.zeros_(self.update[-1].weight)
            nn.init.constant_(self.update[-1].bias, 1.5)

    def _prefix(self, values, support, times):
        b, t, k, h, w = values.shape
        state = torch.zeros((b, k, h, w), dtype=values.dtype, device=values.device)
        seen = torch.zeros((b, 1, h, w), dtype=torch.bool, device=values.device)
        total = torch.zeros_like(state)
        count = torch.zeros_like(seen, dtype=values.dtype)
        states = []
        for index in range(t):
            valid = support[:, index]
            observation = values[:, index].masked_fill(~valid, 0)
            if self.mode == "independent":
                proposal = observation
            elif self.mode == "mean":
                total = total + observation
                count = count + valid
                proposal = total / count.clamp_min(1)
            else:
                confidence = (2 * observation - 1).abs()
                if self.mode == "confidence":
                    # A high-confidence reversal updates rapidly; uncertain noise
                    # contributes little. No GT or fixed damage veto is involved.
                    gate = .05 + .9 * confidence.square()
                else:
                    gap = (times[:, index] - times[:, index - 1]).clamp_min(0) if index else times[:, index] * 0
                    age = (times[:, index] - times[:, 0]).clamp_min(0)
                    time = torch.log1p(torch.stack((gap, age), 1)).to(values.dtype)
                    time = time[:, :, None, None].expand(b, 2, h, w)
                    inputs = torch.cat((state, observation, confidence,
                                        observation - state, time), 1)
                    gate = self.update(inputs).sigmoid()
                proposal = (1 - gate) * state + gate * observation
            proposal = torch.where(seen, proposal, observation)
            state = torch.where(valid, proposal, state)
            seen = seen | valid
            states.append(state)
        return torch.stack(states, 1)

    def forward(self, probabilities, support, times, padding=None):
        support = _support(probabilities, support, padding)
        times = _times(times, probabilities, padding)
        if probabilities.shape[2] != self.classes:
            raise ValueError("Semantic channels differ from the recorded ontology")
        selected = probabilities.masked_select(support.expand_as(probabilities))
        if not torch.isfinite(selected).all() or torch.any((selected < 0) | (selected > 1)):
            raise ValueError("Supported predicted semantic probabilities must lie in [0,1]")
        values = probabilities.masked_fill(~support, 0)
        if self.mode != "repeat":
            return self._prefix(values, support, times)
        outputs = []
        for end in range(1, values.shape[1] + 1):
            previous = max(0, end - 2)
            repeated = values[:, :end].clone()
            repeated_support = support[:, :end].clone()
            if end > 2:
                repeated[:, :-2] = values[:, previous:previous + 1]
                # Common geometric support; retain exact last pair.
                repeated_support[:, :-2] &= support[:, previous:previous + 1]
            outputs.append(self._prefix(repeated, repeated_support, times[:, :end])[:, -1])
        return torch.stack(outputs, 1)


def semantic_change(previous, current):
    """Annotation-independent signal; this is not semantic class recognition."""
    return (current - previous).abs().amax(dim=1, keepdim=True)


class PairAuxDetector(nn.Module):
    """The public inference signature accepts exactly two observations."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.decoder = nn.Sequential(nn.Conv2d(3 * hidden + 3, hidden, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(hidden, 1, 1))
        # Shared by all B controls. Reconstruction anchors representations to
        # nonconstant frozen input features and prevents an auxiliary-only collapse.
        self.reconstruct = nn.Conv2d(hidden, channels, 1)

    def encode(self, features, support):
        if support.ndim == 3:
            support = support[:, None]
        return self.encoder(features.masked_fill(~support.bool(), 0)) * support

    def forward(self, previous, current, previous_support, current_support, gap):
        if previous.shape != current.shape or previous.ndim != 4:
            raise ValueError("Pair inference requires matching B,C,H,W endpoints")
        if previous_support.ndim == 3:
            previous_support = previous_support[:, None]
        if current_support.ndim == 3:
            current_support = current_support[:, None]
        p = self.encode(previous, previous_support)
        c = self.encode(current, current_support)
        gap = torch.as_tensor(gap, device=p.device, dtype=p.dtype).reshape(-1)
        if len(gap) != len(p) or torch.any(gap < 0):
            raise ValueError("One nonnegative real time gap per pair is required")
        gap = torch.log1p(gap)[:, None, None, None].expand(-1, 1, *p.shape[-2:])
        return self.decoder(torch.cat((p, c, c - p, previous_support.float(),
                                       current_support.float(), gap), 1))


def _masked_mean(values, mask):
    mask = mask.bool()
    if mask.ndim == values.ndim - 1:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return values.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)


def pair_auxiliary_loss(model, mode, features, support, *, stable_history=None,
                        changed_mask=None, noise_std=.05):
    """TRAIN-only supervision; ``stable_history`` is B,T-1,H,W, relative to current.

    The regularization control uses feature-level Gaussian perturbations, not RGB
    illumination/geometry augmentation. All three modes share exactly one model.
    Changed pixels are excluded from temporal equality and receive a separation
    margin. A reconstruction anchor penalizes collapsed representations.
    """
    if mode not in PAIR_MODES:
        raise ValueError("Unknown pair training control")
    support = _support(features, support)
    previous, current = features[:, -2], features[:, -1]
    p = model.encode(previous, support[:, -2])
    c = model.encode(current, support[:, -1])
    zero = c.sum() * 0
    if mode == "pair":
        return dict(total=zero, consistency=zero, reconstruction=zero, separation=zero)
    reconstruction = (_masked_mean((model.reconstruct(p) - previous.detach()).square(), support[:, -2])
                      + _masked_mean((model.reconstruct(c) - current.detach()).square(), support[:, -1])) / 2
    if mode == "regularized":
        perturbed = current + torch.randn_like(current) * noise_std
        perturbed_repr = model.encode(perturbed, support[:, -1])
        consistency = _masked_mean((perturbed_repr - c.detach()).square(), support[:, -1])
    else:
        if stable_history is None or stable_history.shape != (len(features), features.shape[1] - 1, *features.shape[-2:]):
            raise ValueError("History supervision needs explicit known stable TRAIN correspondences")
        numerator, denominator = zero, zero
        for index in range(features.shape[1] - 1):
            old = model.encode(features[:, index], support[:, index])
            valid = stable_history[:, index:index + 1].bool() & support[:, index] & support[:, -1]
            numerator = numerator + (old - c).square().masked_fill(~valid, 0).sum()
            denominator = denominator + valid.sum() * c.shape[1]
        consistency = numerator / denominator.clamp_min(1)
    separation = zero
    if changed_mask is not None:
        valid = changed_mask.bool()
        if valid.ndim == 3:
            valid = valid[:, None]
        valid = valid & support[:, -2] & support[:, -1]
        distance = ((c - p).square().mean(1, keepdim=True) + 1e-8).sqrt()
        separation = _masked_mean(F.relu(.25 - distance).square(), valid)
    return dict(total=consistency + .1 * reconstruction + .1 * separation,
                consistency=consistency, reconstruction=reconstruction, separation=separation)


def stable_correspondences(semantic_gt, known, support):
    """All endpoint classes must be known and equal; never infer stable from unknown."""
    if semantic_gt.ndim != 5:
        raise ValueError("Expected B,T,K,H,W semantic targets")
    if support.shape[-2:] != semantic_gt.shape[-2:]:
        b, t = support.shape[:2]
        support = F.interpolate(support.reshape(b * t, 1, *support.shape[-2:]).float(),
                                size=semantic_gt.shape[-2:], mode="nearest").reshape(b, t, 1, *semantic_gt.shape[-2:]).bool()
    support = _support(semantic_gt, support)
    if known.ndim == 4:
        known = known[:, :, None]
    known = known.bool().expand_as(semantic_gt)
    known = known & (semantic_gt >= 0) & (semantic_gt <= 1)
    result = []
    for index in range(semantic_gt.shape[1] - 1):
        valid = known[:, index].all(1) & known[:, -1].all(1)
        equal = (semantic_gt[:, index] == semantic_gt[:, -1]).all(1)
        result.append(valid & equal & support[:, index, 0] & support[:, -1, 0])
    return torch.stack(result, 1)


def _ontology_channels(class_names: Sequence[str], damage_classes=DAMAGE_CLASSES):
    names = [str(x).upper() for x in class_names]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous semantic channel names")
    missing = sorted(set(damage_classes) - set(names))
    if missing or "REPAIRS" not in names:
        raise ValueError("Progression ontology lacks required named channels: " + ", ".join(missing + ([] if "REPAIRS" in names else ["REPAIRS"])))
    return [names.index(name) for name in damage_classes], names.index("REPAIRS")


def progression_target(semantic_gt, known, support, class_names, times,
                       damage_classes=DAMAGE_CLASSES):
    """Build a conservative annotation-derived proxy on one common event cohort.

    Returns target -1/0/1 and states: -1 unknown, 0 stable, 1 worsening,
    2 recovery, 3 unconfirmed/transient. Every observation must be supported and
    semantically known. Repair resets the damage episode; a later reappearance
    needs two observations before it becomes a positive progression proxy.
    """
    gt = np.asarray(semantic_gt)
    if gt.ndim != 4 or gt.shape[1] != len(class_names):
        raise ValueError("Expected T,K,H,W semantic targets matching ontology")
    t, _, h, w = gt.shape
    times = np.asarray(times)
    if times.shape != (t,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Progression requires chronological known observation times")
    damage_channels, repair_channel = _ontology_channels(class_names, damage_classes)
    known = np.asarray(known, bool)
    if known.ndim == 3:
        known = known[:, None]
    known = np.broadcast_to(known, gt.shape) & (gt >= 0) & (gt <= 1)
    support = np.asarray(support, bool)
    if support.ndim == 4 and support.shape[1] == 1:
        support = support[:, 0]
    if support.shape != (t, h, w):
        raise ValueError("Progression support shape mismatch")
    selected = damage_channels + [repair_channel]
    complete = (known[:, selected].all(1) & support).all(0)
    target = np.full((h, w), -1, np.int8)
    state = np.full((h, w), -1, np.int8)
    damage = (gt[:, damage_channels] == 1).any(1)
    repair = gt[:, repair_channel] == 1
    if t >= 3:
        # Contradictory simultaneous repair/damage is not forced into a binary event.
        complete &= ~(damage & repair).any(0)
        unchanged = (damage == damage[0]).all(0) & ~repair.any(0)
        state[complete & unchanged] = 0
        target[complete & unchanged] = 0
        recovery = damage[:-1].any(0) & ~damage[-1]
        state[complete & recovery] = 2
        target[complete & recovery] = 0
        # Time of the latest episode start and its number of consecutive confirmations.
        run_length = np.zeros((h, w), np.int32)
        had_known_absence = np.zeros((h, w), bool)
        episode_started = np.zeros((h, w), bool)
        for index in range(t):
            absent = ~damage[index] | repair[index]
            had_known_absence |= absent
            started = damage[index] & (run_length == 0) & had_known_absence
            episode_started = np.where(absent, False, episode_started | started)
            run_length = np.where(absent, 0, run_length + 1)
        worsening = episode_started & (run_length >= 2)
        state[complete & worsening] = 1
        target[complete & worsening] = 1
        unresolved = complete & (state < 0)
        state[unresolved] = 3
    valid = target >= 0
    return dict(target=target, state=state, support=valid,
                rule=dict(PROGRESSION_RULE, damage_classes=list(damage_classes)),
                coverage=dict(observations=t, pixels=h * w, complete_pixels=int(complete.sum()),
                              valid_pixels=int(valid.sum()), positive_pixels=int((target == 1).sum()),
                              stable_pixels=int((state == 0).sum()), recovery_pixels=int((state == 2).sum()),
                              unconfirmed_pixels=int((state == 3).sum()),
                              status="ELIGIBLE" if valid.any() else "INSUFFICIENT_DATA",
                              reason=None if valid.any() else "Need >=3 supported known observations and a resolved progression proxy"))


def progression_review_manifest(rows, minimum_buildings=2, minimum_positive_buildings=1):
    """Coverage manifest for human review; no synthetic substitution for missing data."""
    events = []
    buildings, positives = set(), set()
    for row in rows:
        count = row["coverage"]
        entry = dict(event_id=row["event_id"], building_id=row["building_id"],
                     split=row["split"], **count)
        events.append(entry)
        if count["valid_pixels"]:
            buildings.add(row["building_id"])
        if count["positive_pixels"]:
            positives.add(row["building_id"])
    sufficient = len(buildings) >= minimum_buildings and len(positives) >= minimum_positive_buildings
    return dict(target_type=PROGRESSION_RULE["target_type"], rule=PROGRESSION_RULE,
                status="ELIGIBLE" if sufficient else "INSUFFICIENT_DATA",
                reason=None if sufficient else "Too few buildings with known progression proxy or positive events",
                buildings=len(buildings), positive_buildings=len(positives), events=events)


class ProgressionDetector(nn.Module):
    """Same-capacity latest/first/history detectors on the same progression target."""

    def __init__(self, channels, hidden=64, mode="history"):
        super().__init__()
        if mode not in PROGRESSION_MODES:
            raise ValueError("Unknown progression control")
        self.mode = mode
        self.encoder = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU())
        self.decoder = nn.Sequential(nn.Conv2d(4 * hidden + 3, hidden, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, features, support, times, padding=None):
        support = _support(features, support, padding)
        times = _times(times, features, padding)
        if padding is not None:
            outputs = []
            for index, length in enumerate(padding.bool().sum(1).tolist()):
                outputs.append(self.forward(features[index:index + 1, :length],
                    support[index:index + 1, :length], times[index:index + 1, :length]))
            return torch.cat(outputs, 0)
        if features.shape[1] < 2:
            raise ValueError("Progression detector needs at least a pair")
        b, t, c, h, w = features.shape
        values = self.encoder(features.masked_fill(~support, 0).reshape(b * t, c, h, w))
        values = values.reshape(b, t, -1, h, w) * support
        current = values[:, -1]
        if self.mode == "latest":
            reference = values[:, -2]
            trajectory = current - reference
            persistence = trajectory.abs()
            reference_support = support[:, -2]
            horizon = times[:, -1] - times[:, -2]
        elif self.mode == "first":
            first = torch.zeros(b, dtype=torch.long, device=features.device)
            reference = values[torch.arange(b, device=features.device), first]
            reference_support = support[torch.arange(b, device=features.device), first]
            trajectory = current - reference
            persistence = trajectory.abs()
            horizon = times[:, -1] - times[torch.arange(b, device=features.device), first]
        else:
            weights = support.to(values.dtype)
            reference = values[:, :-1].sum(1) / weights[:, :-1].sum(1).clamp_min(1)
            reference_support = support[:, :-1].any(1)
            age = (times - times[:, -1:]).to(values.dtype)[:, :, None, None, None]
            mean_time = (age * weights).sum(1) / weights.sum(1).clamp_min(1)
            mean_value = values.sum(1) / weights.sum(1).clamp_min(1)
            centered = (age - mean_time[:, None]) * weights
            trajectory = (centered * (values - mean_value[:, None])).sum(1) / centered.square().sum(1).clamp_min(1e-6)
            pairs = support[:, 1:] & support[:, :-1]
            delta = (values[:, 1:] - values[:, :-1]).abs().masked_fill(~pairs, 0)
            persistence = delta.sum(1) / pairs.sum(1).clamp_min(1)
            horizon = (times[:, -1:] - times).masked_fill(~support.flatten(2).any(2), 0).amax(1)
        time = torch.log1p(horizon.clamp_min(0))[:, None, None, None].expand(b, 1, h, w)
        inputs = torch.cat((reference, current, trajectory, persistence,
                            reference_support.float(), support[:, -1].float(), time), 1)
        return self.decoder(inputs)


def build_scenario_model(spec: Mapping, feature_channels, semantic_channels):
    scenario, mode = spec["scenario"], spec["mode"]
    hidden = int(spec.get("hidden", 64))
    if scenario == "A":
        return SemanticMemory(semantic_channels, mode, hidden)
    if scenario == "B":
        if mode not in PAIR_MODES:
            raise ValueError("Invalid B control")
        return PairAuxDetector(feature_channels, hidden)
    if scenario == "C":
        return ProgressionDetector(feature_channels, hidden, mode)
    raise ValueError("Unknown A/B/C scenario")


def scenario_forward(model, spec, features, support, times, *, semantic_probs=None, padding=None):
    scenario = spec["scenario"]
    values = semantic_probs if scenario == "A" else features
    if values is None:
        raise ValueError("A requires predicted semantic probabilities, not GT")
    valid = _support(values, support, padding)
    times = _times(times, values, padding)
    indices = torch.full((len(values),), values.shape[1] - 1, device=values.device, dtype=torch.long)
    if padding is not None:
        indices = padding.bool().sum(1) - 1
    if torch.any(indices < 1):
        raise ValueError("A/B/C event evaluation needs at least two observations")
    batch = torch.arange(len(values), device=values.device)
    if scenario == "A":
        semantic = model(semantic_probs, valid, times, padding)
        probabilities = semantic[batch, indices].clamp(1e-6, 1 - 1e-6)
        change = semantic_change(semantic[batch, indices - 1], semantic[batch, indices]).clamp(1e-6, 1 - 1e-6)
        return dict(semantic_sequence=semantic, semantic_logits=torch.logit(probabilities),
                    logits=torch.logit(change), last_indices=indices)
    if scenario == "B":
        logits = model(features[batch, indices - 1], features[batch, indices],
                       valid[batch, indices - 1], valid[batch, indices],
                       times[batch, indices] - times[batch, indices - 1])
    elif scenario == "C":
        logits = model(features, valid, times, padding)
    else:
        raise ValueError("Unknown A/B/C scenario")
    return dict(logits=logits, last_indices=indices)


def scenario_loss(model, spec, output, features, support, times, target, known, *,
                  semantic_probs=None, semantic_gt=None, semantic_known=None,
                  stable_history=None):
    """Loss helper; callers must supply TRAIN only for optimizer updates."""
    if spec["scenario"] == "A":
        if semantic_gt is None or semantic_known is None:
            raise ValueError("A needs explicit semantic TRAIN/VAL targets for loss only")
        indices = output.get("last_indices")
        batch = torch.arange(len(features), device=features.device)
        gt = semantic_gt[batch, indices] if semantic_gt.ndim == 5 else semantic_gt
        mask = semantic_known[batch, indices] if semantic_known.ndim >= 4 and semantic_known.shape[1] == features.shape[1] else semantic_known
        return masked_bce(output["semantic_logits"], gt, mask)
    logits = output["logits"]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    primary = masked_bce(logits, target, known)
    if spec["scenario"] == "B" and spec["mode"] != "pair":
        if stable_history is None and semantic_gt is not None and semantic_known is not None:
            stable_history = stable_correspondences(semantic_gt, semantic_known, support)
        if stable_history is not None and stable_history.shape[-2:] != features.shape[-2:]:
            b, t = stable_history.shape[:2]
            stable_history = F.adaptive_avg_pool2d(stable_history.reshape(b * t, 1, *stable_history.shape[-2:]).float(),
                features.shape[-2:]).reshape(b, t, *features.shape[-2:]) == 1
        changed = known.bool() & (target == 1)
        if changed.shape[-2:] != features.shape[-2:]:
            changed = F.adaptive_max_pool2d(changed[:, None].float(), features.shape[-2:])[:, 0].bool()
        auxiliary = pair_auxiliary_loss(model, spec["mode"], features, support,
            stable_history=stable_history, changed_mask=changed,
            noise_std=float(spec.get("feature_noise_std", .05)))
        return primary + float(spec.get("auxiliary_weight", .1)) * auxiliary["total"]
    return primary
