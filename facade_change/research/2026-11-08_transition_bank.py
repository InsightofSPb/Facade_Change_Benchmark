"""PatchCore-inspired banks of feature differences, not a PatchCore reproduction."""
from __future__ import annotations

from functools import lru_cache
import hashlib

import numpy as np
import torch


def patch_reduce(image, operation="mean"):
    blocks = np.asarray(image).reshape(16, 14, 16, 14).transpose(0, 2, 1, 3).reshape(256, 196)
    return getattr(blocks, operation)(axis=1)


def seed_for(seed, text):
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8], "little")


class Reservoir:
    """Bounded uniform sample with retained event/patch/year provenance."""

    def __init__(self, size, seed):
        self.size, self.rng = size, np.random.default_rng(seed)
        self.arrays = None

    def add(self, vectors, confidence, row, indices):
        if not len(indices):
            return
        n = len(indices)
        batch = dict(vectors=vectors[indices], confidence=confidence[indices],
                     year=np.full(n, row["current_year"]), event=np.full(n, row["event_id"]),
                     patch=indices, priority=self.rng.random(n))
        if self.arrays is not None:
            batch = {k: np.concatenate((self.arrays[k], value)) for k, value in batch.items()}
        selected = np.argsort(batch["priority"], kind="stable")[:self.size]
        self.arrays = {k: value[selected] for k, value in batch.items()}


class TransitionBank:
    def __init__(self, inputs, store, args, progress):
        self.inputs, self.store, self.args, self.progress = inputs, store, args, progress
        self.rows = dict(inputs.source.data.events)
        # Canonical scored events start at frame 2. The first transition has
        # cached endpoint features but no seven-detector maps or event target.
        for track in getattr(inputs.source.data, "tracks", {}).values():
            identifier = "prefix:" + track["track_id"]
            self.rows[identifier] = dict(event_id=identifier, frame=1,
                current_year=track["observations"][1]["year"],
                previous_year=track["observations"][0]["year"],
                **{k: track[k] for k in ("track_id", "building_id", "split")})
        self.events = {(r["track_id"], r["frame"]): r for r in self.rows.values()}
        self.global_banks, self.foreign_banks, self.coverage = {}, {}, {}

    @lru_cache(maxsize=128)
    def transition(self, event_id):
        """Prediction-only inputs: no annotations, future frames or RGB decoding."""
        row = self.rows[event_id]
        observations = self.inputs.source.data.tracks[row["track_id"]]["observations"]
        endpoints = observations[row["frame"] - 1:row["frame"] + 1]
        features, masks = zip(*(self.inputs.feature(r["observation_id"]) for r in endpoints))
        values = [f.reshape(384, 256).T.astype(np.float32) for f in features]
        values = [v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for v in values]
        vectors = values[1] - values[0]  # Signed; do not normalize away the magnitude.
        valid = masks[0].ravel() & masks[1].ravel()
        # The same DINO-only gate works for the otherwise unscored first pair.
        # This is a feature-distance score, not a calibrated change probability.
        confidence = np.linalg.norm(vectors, axis=1) / 2
        return vectors, valid, confidence

    def prepare(self):
        identifier = "2026-11-08_transition-banks"
        cached = self.store.read_unit(identifier)
        if cached:
            record, arrays = cached
            self.stable_threshold = record["stable_threshold"]
            self.buildings = record["buildings"]
            for i, building in enumerate(self.buildings):
                self.global_banks[building] = {k.split("-", 2)[2]: v for k, v in arrays.items()
                                               if k.startswith(f"g-{i}-")}
                self.foreign_banks[building] = {k.split("-", 2)[2]: v for k, v in arrays.items()
                                                if k.startswith(f"f-{i}-")}
            return record
        train = sorted((r for r in self.inputs.source.data.samples if r["split"] == "train"),
                       key=lambda r: r["event_id"])
        self.buildings = sorted({r["building_id"] for r in train})
        global_samples = {b: Reservoir(self.args.bank_per_building, seed_for(self.args.seed, "global:" + b))
                          for b in self.buildings}
        confidence_by_building = {b: [] for b in self.buildings}
        for row in self.progress(train, desc="TRAIN: annotated stable transitions", unit="crop", position=1):
            vectors, valid, confidence = self.transition(row["event_id"])
            target = self.inputs.source.data.target(row["event_id"])["target"]
            stable = (valid & (patch_reduce(target != 255) >= self.args.min_known_fraction)
                      & ~patch_reduce(target == 1, "any"))
            indices = np.flatnonzero(stable)
            global_samples[row["building_id"]].add(vectors, confidence, row, indices)
            confidence_by_building[row["building_id"]].extend(confidence[indices].tolist())
        if sum(s.arrays is not None for s in global_samples.values()) < 2:
            raise ValueError("Transition bank needs stable annotated patches in at least two TRAIN buildings; "
                             "inspect annotation/support coverage before changing --min-known-fraction")
        # A median of per-building quantiles avoids a large facade choosing the gate alone.
        self.stable_threshold = float(np.median([np.quantile(c, self.args.stable_quantile)
                                                for c in confidence_by_building.values() if c]))
        foreign_samples = {b: Reservoir(self.args.foreign_per_building, seed_for(self.args.seed, "foreign:" + b))
                           for b in self.buildings}
        donor_rows = sorted((r for r in self.rows.values() if r["split"] == "train"), key=lambda r: r["event_id"])
        for row in self.progress(donor_rows, desc="TRAIN: predicted stable donor transitions", unit="crop", position=1):
            vectors, valid, confidence = self.transition(row["event_id"])
            indices = np.flatnonzero(valid & (confidence <= self.stable_threshold))
            foreign_samples[row["building_id"]].add(vectors, confidence, row, indices)
        arrays = {}
        for i, building in enumerate(self.buildings):
            self.global_banks[building] = global_samples[building].arrays or {}
            self.foreign_banks[building] = foreign_samples[building].arrays or {}
            for prefix, sample in (("g", self.global_banks[building]), ("f", self.foreign_banks[building])):
                arrays.update({f"{prefix}-{i}-{k}": v for k, v in sample.items()})
        return self.store.save_unit(dict(unit_id=identifier, kind="transition_reference_banks",
            fit_split="train", buildings=self.buildings, stable_threshold=self.stable_threshold,
            normality="no known positive pixel; minimum known annotation fraction; unknown is not healthy",
            train_query_policy="exclude entire query building from global and foreign banks",
            admission_score="L2 distance between normalized frozen DINO tokens / 2; not a probability",
            sizes={b: len(v.get("vectors", [])) for b, v in self.global_banks.items()}), arrays)

    @lru_cache(maxsize=32)
    def reference(self, excluded_building):
        parts = [v["vectors"] for b, v in self.global_banks.items() if b != excluded_building and v]
        if not parts:
            raise ValueError(f"Empty leave-building-out transition bank: {excluded_building}")
        return torch.as_tensor(np.concatenate(parts), device=self.inputs.device)

    def personal(self, row):
        parts = []
        for frame in range(1, row["frame"]):
            old = self.events[(row["track_id"], frame)]
            if old["current_year"] >= row["current_year"]:
                raise ValueError("Noncausal personal transition prefix")
            vectors, valid, confidence = self.transition(old["event_id"])
            parts.append(vectors[valid & (confidence <= self.stable_threshold)])
        return np.concatenate(parts) if parts else np.empty((0, 384), np.float32)

    def matched_memories(self, row):
        own = self.personal(row)
        own_count = len(own)
        donors = []
        for building, bank in self.foreign_banks.items():
            if building == row["building_id"] or not bank:
                continue
            candidates = bank["vectors"][bank["year"] < row["current_year"]]
            if len(candidates):
                donors.append((building, candidates))
        if not len(own) or not donors:
            self.coverage[row["event_id"]] = dict(event_id=row["event_id"], split=row["split"],
                building_id=row["building_id"], frame=row["frame"], available_own=len(own), matched=0, donor="")
            return own[:0], own[:0]
        rng = np.random.default_rng(seed_for(self.args.seed, "match:" + row["event_id"]))
        # Pick one other TRAIN building, then match the exact number of added vectors.
        requested = min(len(own), self.args.personal_size)
        enough = [item for item in donors if len(item[1]) >= requested]
        donors = enough or [item for item in donors if len(item[1]) == max(len(v) for _, v in donors)]
        donor, foreign = donors[int(rng.integers(len(donors)))]
        size = min(len(own), len(foreign), self.args.personal_size)
        own = own[rng.choice(len(own), size, replace=False)]
        foreign = foreign[rng.choice(len(foreign), size, replace=False)]
        self.coverage[row["event_id"]] = dict(event_id=row["event_id"], split=row["split"],
            building_id=row["building_id"], frame=row["frame"], available_own=own_count,
            matched=size, donor=donor)
        return own, foreign

    @staticmethod
    def nearest(query, bank):
        if not len(bank):
            return torch.full((len(query),), 1., device=query.device)
        bank = torch.as_tensor(bank, device=query.device)
        # Both endpoints have norm <= 1, so distances between signed deltas are <= 4.
        distance = (query.square().sum(1)[:, None] + bank.square().sum(1)[None]
                    - 2 * query @ bank.T).clamp_min_(0)
        return (distance.min(1).values.sqrt() / 4).clamp_(0, 1)

    @lru_cache(maxsize=48)
    def maps(self, event_id):
        row = self.rows[event_id]
        vectors, valid, _ = self.transition(event_id)
        own, foreign = self.matched_memories(row)
        distances = np.zeros((3, 256), np.float32)
        if valid.any():
            with torch.inference_mode():
                query = torch.as_tensor(vectors[valid], device=self.inputs.device)
                excluded = row["building_id"] if row["split"] == "train" else None
                global_distance = self.nearest(query, self.reference(excluded))
                distances[:, valid] = torch.stack((global_distance,
                    torch.minimum(global_distance, self.nearest(query, own)),
                    torch.minimum(global_distance, self.nearest(query, foreign)))).cpu().numpy()
        maps = np.concatenate((distances, valid[None].astype(np.float32))).reshape(4, 16, 16)
        return maps.repeat(14, axis=1).repeat(14, axis=2)
