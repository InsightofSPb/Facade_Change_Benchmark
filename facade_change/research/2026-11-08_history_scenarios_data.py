"""Read-only adapters for the existing canonical history/cache contracts.

Forecast inputs and evaluation targets have separate APIs. The former reads only
an explicitly bounded prefix, even if the target observation is absent. No new
dataset split, detector run, or image preprocessing framework is introduced.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path

import numpy as np

from facade_change.annotation_targets import target_difference
from facade_change.history_features import digest
from facade_change.io import read_json, sha256

memory = importlib.import_module("data_analysis.2026-11-08_variability_memory")
saved_module = importlib.import_module("data_analysis.2026-11-08_memory_evaluate")
fusion, benchmark = memory.fusion, memory.benchmark
EXPECTED_NAMES = ("WATER_STAIN", "MISSING_ELEMENT", "DELAMINATION", "SPALLING",
                  "ORNAMENT_INTACT", "CRACK", "REPAIRS", "EFFLORESCENCE",
                  "TEXT_OR_IMAGES", "CORROSION", "ADVERTISEMENTS")
PREFIX = "2026-11-08_"


def validate_partitions(tracks):
    """Known buildings, views and source-image hashes cannot cross partitions."""
    seen = {name: {} for name in ("building_id", "view_id", "image_sha256")}
    for track in tracks:
        split = track["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        for name in ("building_id", "view_id"):
            value = track.get(name)
            if not value or seen[name].setdefault(value, split) != split:
                raise ValueError(f"Missing identity or split leakage: {name}={value}")
        years = [row["year"] for row in track["observations"]]
        if any(type(year) is not int for year in years) or any(b <= a for a, b in zip(years, years[1:])):
            raise ValueError("Need strictly increasing known years, without invented exact dates")
        for row in track["observations"]:
            value = row.get("image_sha256")
            if not value or seen["image_sha256"].setdefault(value, split) != split:
                raise ValueError("Missing source-image hash or cross-split image leakage")
    return seen["building_id"]


def ontology_contract(categories, semantic_contract=None):
    ids, names = [c["id"] for c in categories], [c["name"] for c in categories]
    if (not categories or len(set(ids)) != len(ids) or len(set(names)) != len(names)
            or len(categories) > 16 or any(type(i) is not int or i < 0 for i in ids)):
        raise ValueError("Ambiguous semantic ID/channel mapping")
    if semantic_contract is not None:
        mapping = semantic_contract.get("category_to_channel")
        if isinstance(mapping, dict):
            mapping = [{"category_id": int(k), "channel": v} for k, v in mapping.items()]
        if not mapping or {r["category_id"] for r in mapping} != set(ids):
            raise ValueError("Semantic checkpoint ontology differs from annotation ontology")
        if sorted(r["channel"] for r in mapping) != list(range(len(ids))):
            raise ValueError("Semantic checkpoint channels are not a bijection")
    return dict(sha256=digest(categories), categories=categories,
                category_to_channel=[dict(category_id=c["id"], name=c["name"], channel=i)
                                     for i, c in enumerate(categories)],
                source_head_category_to_channel=(semantic_contract or {}).get("category_to_channel"),
                probability_cache_order="canonical categories; existing semantic_context explicitly reorders source head",
                expected_names_missing=sorted(set(EXPECTED_NAMES) - set(names)),
                unexpected_names=sorted(set(names) - set(EXPECTED_NAMES)),
                unknown="packed mask 0 / temporal target 255; excluded, never unchanged",
                background="no independently annotated healthy-background channel")


def audit_causal_geometry(canonical):
    """Accept only saved independent direct-to-earliest-anchor rendering.

    A target's registration is allowed for comparison after prediction. It is
    never used to resample or mask the past input features. Cohort eligibility
    remains retrospective and is reported rather than confused with causality.
    """
    path = canonical.history_root / "manifest.json"
    document = read_json(path)
    expected = canonical.index["history_manifest_sha256"]
    if sha256(path) != expected:
        raise ValueError("Canonical history manifest changed")
    canonical.fp.add(path, expected)
    options = document.get("options", {})
    if (options.get("grid") != "native anchor (0,0); fixed integer multiples of stride, including negative coordinates"
            or options.get("selection") != "saved session selection; direct earliest-anchor homographies only"):
        raise ValueError("Forecast cache lacks a verified past-defined fixed grid and direct-anchor geometry; "
                         "re-export canonical histories using workflow prepare crop-histories")
    views = {(v["building_id"], v["view_id"]): v for v in document["views"]}
    for track in canonical.tracks.values():
        view = views.get((track["building_id"], track["view_id"]))
        if view is None or track.get("anchor_sha256") != view["anchor"]["sha256"]:
            raise ValueError("Track anchor does not match saved past-defined view")
        if any(r["year"] < view["anchor"]["year"] for r in track["observations"]):
            raise ValueError("A future observation defines the forecast coordinate frame")
        for row in track["observations"]:
            matrix = np.asarray(row["alignment"]["matrix"], np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError("Invalid source-to-earliest-anchor matrix")
    return dict(status="passed", history_manifest_sha256=expected,
                coordinates="native earliest-anchor grid; per-observation direct source_to_anchor",
                encoder="independent observation only; fixed ImageNet preprocessing",
                target_registration="comparison only; cannot update already-built forecast inputs",
                crop_selection="fixed past track/bbox for forecast; target coverage only evaluation support",
                cohort_caveat="original canonical >=3-observation cohort is retrospectively selected; "
                              "no claim of prospective deployment-cohort validation")


def _prefix_inputs(dataset, track_id, target_year, include_current=False):
    """Critical causal boundary: no target/future feature or GT accessor here."""
    rows = [r for r in dataset.tracks[track_id]["observations"]
            if r["year"] < target_year or (include_current and r["year"] == target_year)]
    if not rows:
        raise ValueError("No observations before forecast date")
    values = [dataset.feature(r["observation_id"]) for r in rows]
    return dict(features=np.stack([v[0] for v in values]),
                support=np.stack([v[1] for v in values]),
                years=np.asarray([r["year"] for r in rows], np.float32),
                target_year=float(target_year), padding=np.ones(len(rows), bool),
                observation_ids=[r["observation_id"] for r in rows],
                time_precision="year", coordinate_frame="native earliest-anchor grid",
                bbox_xywh=list(dataset.tracks[track_id]["bbox_xywh"]))


def pool_temporal_target(target, shape):
    """Conservative patch supervision: all pixels known; any positive is changed."""
    target = np.asarray(target)
    h, w = shape
    if target.ndim != 2 or target.shape[0] % h or target.shape[1] % w:
        raise ValueError("Target and feature grids are not integer aligned")
    blocks = target.reshape(h, target.shape[0] // h, w, target.shape[1] // w)
    known = np.all(blocks != 255, axis=(1, 3))
    result = np.full((h, w), 255, np.uint8)
    result[known] = np.any(blocks == 1, axis=(1, 3))[known]
    return result[None], known[None]


class EvaluationData:
    """The same feature-compatible geometry for every new method and its controls.

    Original canonical targets remain available separately for exact old-stack
    parity. This adapter changes only the declared common evaluation support.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.categories, self.groups = dataset.categories, dataset.groups
        self.samples = []
        for row in dataset.samples:
            target = self.target(row["event_id"])["target"]
            self.samples.append({**row, "known_pixels": int(np.count_nonzero(target != 255)),
                                 "positive_pixels": int(np.count_nonzero(target == 1)),
                                 "negative_pixels": int(np.count_nonzero(target == 0))})
        self.events = {r["event_id"]: r for r in self.samples}

    def target(self, event_id):
        row = self.dataset.events[event_id]
        result = {k: v.copy() for k, v in self.dataset.targets(row).items()}
        support = result["support"] & self.dataset.evaluation_support(row)
        result["support"] = support
        result["target"][~support] = 255
        result["class_targets"][:, ~support] = 255
        return result


class ScenarioData:
    """Bounded, read-only reuse of prepared targets, detector maps and DINO units."""
    def __init__(self, memory_run, benchmark_run=None, fusion_run=None):
        self.saved = saved_module.SavedMemory(memory_run)
        self.saved.__enter__()
        self.overlay = None
        try:
            config = self.saved.config
            self.source = fusion.Source(benchmark_run or config["source"])
            if config.get("source_sha256") != self.source.hashes:
                raise ValueError("Memory cache belongs to a different benchmark")
            self.frozen = memory.FrozenStack(fusion_run or config["fusion"], self.source)
            if config.get("fusion_sha256") != self.frozen.hashes:
                raise ValueError("Memory cache belongs to a different frozen stack")
            self.canonical = self.source.data
            self.tracks, self.samples = self.canonical.tracks, list(self.canonical.samples)
            self.events = self.canonical.events
            self.categories, self.groups = self.canonical.categories, self.canonical.groups
            self.assignments = validate_partitions(self.tracks.values())
            self.ontology = ontology_contract(self.categories, self.source.contract)
            self.geometry = audit_causal_geometry(self.canonical)
            self.encoder = read_json(self.saved.out / (PREFIX + "encoder.json"))
            self.source_memory_hashes = dict(self.saved.hashes)
            original = read_json(self.source.root / "method_contracts/dinov2.json")
            if (self.encoder.get("backbone") != "dinov2_vits14"
                    or self.encoder.get("dimensions") != 384
                    or self.encoder.get("save_patch_tokens") is not True
                    or self.encoder.get("weights_frozen") is not True
                    or self.encoder.get("resize") is not False
                    or self.encoder.get("augmentation") is not False
                    or self.encoder.get("checkpoint_sha256") != original["checkpoint"]["sha256"]
                    or self.encoder.get("python_tree_sha256") != original["sources"]["python_tree_sha256"]):
                raise ValueError("Existing DINO cache is incompatible with the shared frozen spatial target")
            self.channels, self.grid = 384, (16, 16)
            self.extra_train = self._warmup_rows()
            self.baseline_inputs = memory.Inputs(self.source, self.saved, self.frozen, "cpu")
            self.synthetic = False
            self.cache_index = self._cache_index()
        except BaseException:
            self.saved.__exit__(None, None, None)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.saved.__exit__(*exception)

    def close(self):
        self.saved.__exit__(None, None, None)

    def _warmup_rows(self):
        result = []
        for track in self.tracks.values():
            if track["split"] != "train" or len(track["observations"]) < 2:
                continue
            observations = track["observations"]
            result.append(dict(event_id=digest([track["track_id"], 1, observations[1]["observation_id"]])[:24],
                **{k: track[k] for k in ("track_id", "building_id", "view_id", "split", "bbox_xywh")},
                frame=1, previous_year=observations[0]["year"], current_year=observations[1]["year"],
                predictive_train_only=True))
        return result

    def split_samples(self, split, predictive=False):
        selected = [r for r in self.samples if r["split"] == split]
        return selected + (self.extra_train if split == "train" and predictive else [])

    def forecast_prefix(self, track_id, target_year):
        return _prefix_inputs(self, track_id, target_year)

    def inputs(self, row, predictive=False):
        return _prefix_inputs(self, row["track_id"], row["current_year"], not predictive)

    def pair_inputs(self, row):
        """Deployment input for scenario B: older observations need not exist."""
        track = self.tracks[row["track_id"]]
        observations = [r for r in track["observations"] if r["year"] <= row["current_year"]][-2:]
        if len(observations) != 2 or observations[-1]["year"] != row["current_year"]:
            raise ValueError("Pair inference needs exactly the previous and current observations")
        values = [self.feature(r["observation_id"]) for r in observations]
        return dict(features=np.stack([v[0] for v in values]), support=np.stack([v[1] for v in values]),
                    years=np.asarray([r["year"] for r in observations], np.float32),
                    target_year=float(row["current_year"]), padding=np.ones(2, bool),
                    observation_ids=[r["observation_id"] for r in observations], time_precision="year",
                    coordinate_frame="native earliest-anchor grid", bbox_xywh=list(track["bbox_xywh"]))

    def missing_features(self):
        return [identifier for identifier in sorted(self.canonical.observations)
                if not (self.saved.out / "units" / (PREFIX + "features-" + identifier + ".json")).is_file()
                and (self.overlay is None or self.overlay.read_unit(PREFIX + "causal-features-" + identifier) is None)]

    def _cache_index(self):
        """Stable input identity independent of lazy read/cache order, including absences."""
        records = {}
        for identifier in sorted(self.canonical.observations):
            for kind, root, unit in (("feature", self.saved.out, PREFIX + "features-" + identifier),
                                     ("semantic", self.frozen.root, fusion.PREFIX + "semantic-" + identifier)):
                path = root / "units" / (unit + ".json")
                if not path.exists():
                    records[kind + ":" + identifier] = None
                    continue
                self.saved.pin(path)
                record = read_json(path)
                artifact = fusion.data_module._inside(root, record["artifact_path"])
                self.saved.pin(artifact)
                if self.saved.hashes[str(artifact)] != record["artifact_sha256"]:
                    raise ValueError(f"Corrupt {kind} cache artifact before training: {artifact}")
                with np.load(artifact, allow_pickle=False) as arrays:
                    if kind == "feature":
                        if (arrays["patch_tokens"].shape != (256, 384)
                                or arrays["patch_support"].shape != (256,)
                                or arrays["patch_support"].dtype != bool
                                or not np.isfinite(arrays["patch_tokens"]).all()):
                            raise ValueError(f"Incompatible spatial feature cache: {artifact}")
                        if (record.get("encoder_sha256") != self.saved.encoder_sha256
                                or record.get("target_labels_used") is not False):
                            raise ValueError(f"Feature provenance mismatch: {artifact}")
                    else:
                        p, mask = arrays["probabilities"], arrays["support"]
                        if (p.shape != (len(self.categories), 224, 224) or mask.shape != (224, 224)
                                or mask.dtype != bool or not np.isfinite(p[:, mask]).all()
                                or np.any((p[:, mask] < 0) | (p[:, mask] > 1))
                                or record.get("kind") != "predicted_semantic_context"):
                            raise ValueError(f"Incompatible predicted semantic cache: {artifact}")
                records[kind + ":" + identifier] = dict(unit_sha256=sha256(path),
                    artifact_sha256=record["artifact_sha256"], artifact_path=record["artifact_path"])
        return records

    def prepare_missing(self, store, device="cpu"):
        """Only absent DINO observations; immutable old cache and detectors untouched."""
        self.overlay = store
        missing = self.missing_features()
        if not missing:
            return dict(newly_computed=0, reused=len(self.canonical.observations))
        settings = {**self.source.config["options"]["dinov2"],
                    "config_path": str(self.source.root / "run.json"),
                    "config_sha256": sha256(self.source.root / "run.json"),
                    "trust_checkpoint": self.source.config.get("trust_checkpoint", False)}
        contract = memory.history_features.encoder_contract(settings, device, True, self.canonical.fp)
        # Device metadata may differ; all mathematical/preprocessing/source fields must agree.
        exempt = {"device", "torch", "numpy", "cuda_runtime"}
        if {k: v for k, v in contract.items() if k not in exempt} != {
                k: v for k, v in self.encoder.items() if k not in exempt}:
            raise ValueError("Missing-cache preparation encoder contract differs; use original compatible environment")
        store.event("missing_causal_features_preparation", observations=missing,
                    reason="compatible per-observation units absent; no causal incompatibility",
                    encoder=contract, prior_encoder=self.encoder, source_cache_read_only=True)
        encoder = None
        try:
            for identifier in missing:
                rgb, support = self.canonical.frame(identifier)
                valid = memory.history_features.supported_patches(support)
                if valid.any():
                    if encoder is None:
                        encoder = memory.history_features.DinoEncoder(settings, device, contract)
                    arrays = encoder.encode([rgb], [support], save_patch_tokens=True)[0]
                else:
                    arrays = dict(patch_tokens=np.zeros((256, 384), np.float16), patch_support=valid)
                store.save_unit(dict(unit_id=PREFIX + "causal-features-" + identifier,
                    kind="frozen_observation_features", observation_id=identifier,
                    encoder_sha256=self.saved.encoder_sha256, execution_encoder=contract,
                    target_labels_used=False), arrays)
        finally:
            if encoder is not None:
                encoder.close()
        return dict(newly_computed=len(missing), reused=len(self.canonical.observations) - len(missing))

    @lru_cache(maxsize=24)
    def feature(self, identifier):
        unit = PREFIX + "features-" + identifier
        if (self.saved.out / "units" / (unit + ".json")).is_file():
            record, arrays = self.saved.read_unit(unit)
        else:
            result = None if self.overlay is None else self.overlay.read_unit(PREFIX + "causal-features-" + identifier)
            if result is None:
                raise ValueError(f"Missing DINO feature {identifier}; run prepare_missing in the new output first")
            record, arrays = result
            if record.get("encoder_sha256") != self.saved.encoder_sha256 or record.get("target_labels_used") is not False:
                raise ValueError("Incompatible feature overlay")
        features, support = arrays["patch_tokens"], arrays["patch_support"]
        if (features.shape != (256, 384) or support.shape != (256,)
                or support.dtype != bool or not np.isfinite(features).all()):
            raise ValueError("Invalid spatial frozen cache; expected 256x384 tokens and 256 bool support")
        x = features.T.reshape(384, 16, 16).astype(np.float32)
        m = support.reshape(1, 16, 16)
        return np.where(m, x, 0), m.copy()

    def feature_target(self, row):
        observation = self.tracks[row["track_id"]]["observations"][row["frame"]]
        x, support = self.feature(observation["observation_id"])
        return dict(features=x.copy(), support=support.copy())

    def feature_targets(self, row):
        target = self.targets(row)
        labels, known = pool_temporal_target(target["target"], self.grid)
        current = self.feature_target(row)
        past = self.forecast_prefix(row["track_id"], row["current_year"])
        geometric = current["support"] & past["support"][-1] & past["support"][0]
        return dict(change_target=labels, valid=known & geometric, stable=(labels == 0) & geometric,
                    target_features=current["features"], target_support=geometric)

    def evaluation_support(self, row):
        """Post-prediction only: shared first/previous/current fully supported tokens."""
        observations = self.tracks[row["track_id"]]["observations"]
        supported = np.logical_and.reduce([self.feature(observations[i]["observation_id"])[1][0]
                                            for i in (0, row["frame"] - 1, row["frame"])])
        width, height = row["bbox_xywh"][2:]
        if height % supported.shape[0] or width % supported.shape[1]:
            raise ValueError("Evaluation patch geometry is not aligned to native pixels")
        expanded = np.repeat(np.repeat(supported, height // supported.shape[0], 0),
                             width // supported.shape[1], 1)
        return expanded & self.stack_components(row)[1]

    @property
    def evaluation_data(self):
        if not hasattr(self, "_evaluation_data"):
            self._evaluation_data = EvaluationData(self)
        return self._evaluation_data

    @lru_cache(maxsize=24)
    def annotation(self, identifier):
        record, arrays = benchmark.read_unit(self.canonical.root, "observation-" + identifier)
        expected = self.canonical.index["observations"][identifier]
        if record["artifact_sha256"] != expected["artifact_sha256"]:
            raise ValueError("Annotation unit differs from canonical preparation")
        self.canonical.fp.add(self.canonical.root / record["artifact_path"], record["artifact_sha256"])
        return arrays

    def annotation_history(self, row):
        observations = self.tracks[row["track_id"]]["observations"][:row["frame"] + 1]
        values = [self.annotation(r["observation_id"]) for r in observations]
        packed = np.stack([v["packed"] for v in values])
        support = np.stack([v["support"] for v in values])
        labels = np.stack([(packed & (1 << channel)) != 0 for channel in range(len(self.categories))], axis=1)
        return dict(packed=packed, labels=labels, support=support[:, None],
                    known=((packed != 0) & support)[:, None], target_type="annotation-derived semantic state")

    def targets(self, row):
        if not row.get("predictive_train_only"):
            return self.canonical.target(row["event_id"])
        observations = self.tracks[row["track_id"]]["observations"]
        before, after = [self.annotation(r["observation_id"]) for r in observations[:2]]
        config = read_json(self.canonical.root / "run.json")["config"]
        target, layers, support, _ = target_difference(before, after, range(len(self.categories)),
                                                       config["boundary_tolerance"])
        return dict(target=target, class_targets=layers, support=support,
                    before=before["packed"], after=after["packed"])

    def scores(self, row):
        if row.get("predictive_train_only"):
            raise ValueError("First predictive TRAIN transition has no detector maps; never fabricate stack")
        raw, support = self.baseline_inputs.scores(row["event_id"])
        return self.stack_transform(raw), support

    def stack_components(self, row):
        """Evaluator must merge these before applying stack_transform, as before."""
        if row.get("predictive_train_only"):
            raise ValueError("First predictive TRAIN transition has no detector maps")
        return self.baseline_inputs.scores(row["event_id"])

    def stack_transform(self, raw):
        logits = self.frozen.logits(np.asarray(raw).transpose(1, 2, 0))
        return (1 / (1 + np.exp(-np.clip(logits, -80, 80)))).astype(np.float32)

    @property
    def stack_contract(self):
        return dict(kind="canonical_frozen_stack", method_names=list(fusion.METHODS),
            parameters=self.frozen.model["parameters"].tolist(),
            center=self.frozen.model["center"].tolist(), scale=self.frozen.model["scale"].tolist(),
            normalizer={k: v.tolist() for k, v in self.frozen.normalizer.items()},
            f1_threshold=self.frozen.f1_threshold, fpr_threshold=self.frozen.fpr_threshold,
            source_sha256=dict(self.frozen.hashes),
            overlap="merge raw detector channels (AnyChange OR), then normalize and frozen stack sigmoid")

    def semantic(self, row):
        observations = self.tracks[row["track_id"]]["observations"][:row["frame"] + 1]
        probabilities, support = [], []
        for observation in observations:
            result = benchmark.read_unit(self.frozen.root, fusion.PREFIX + "semantic-" + observation["observation_id"])
            if result is None:
                raise ValueError("Missing predicted semantic cache; source fusion must retain its semantic units")
            record, arrays = result
            if record.get("kind") != "predicted_semantic_context":
                raise ValueError("Semantic input is not predicted probabilities")
            p, m = arrays["probabilities"].astype(np.float32), arrays["support"]
            if (p.shape != (len(self.categories), 224, 224) or m.shape != (224, 224)
                    or m.dtype != bool or not np.isfinite(p[:, m]).all()
                    or np.any((p[:, m] < 0) | (p[:, m] > 1))):
                raise ValueError("Invalid predicted semantic probabilities")
            self.saved.pin(self.frozen.root / record["artifact_path"])
            probabilities.append(np.where(m[None], p, 0))
            support.append(m[None])
        return dict(probabilities=np.stack(probabilities), support=np.stack(support),
                    years=np.asarray([r["year"] for r in observations], np.float32),
                    target_labels_used=False)

    def iter_train_features(self):
        seen = set()
        for track in self.tracks.values():
            if track["split"] != "train":
                continue
            for observation in track["observations"]:
                identifier = observation["observation_id"]
                if identifier not in seen:
                    seen.add(identifier)
                    yield identifier, *self.feature(identifier)

    def summary(self, inspect_targets=True):
        counts = {}
        for split in ("train", "val", "test"):
            tracks = [t for t in self.tracks.values() if t["split"] == split]
            samples = self.split_samples(split)
            counts[split] = dict(buildings=len({t["building_id"] for t in tracks}), histories=len(tracks),
                transitions=len(samples), predictive_transitions=len(self.split_samples(split, True)),
                history_lengths=dict(Counter(str(len(t["observations"])) for t in tracks)),
                intervals_years=dict(Counter(str(b["year"] - a["year"]) for t in tracks
                    for a, b in zip(t["observations"], t["observations"][1:]))),
                valid_pixels=sum(int(np.count_nonzero(self.targets(r)["target"] != 255)) for r in samples)
                             if inspect_targets else None)
        return dict(kind="history_three_scenarios_data", synthetic=False, counts=counts,
                    channels=self.channels, grid=list(self.grid), ontology=self.ontology, causality=self.geometry,
                    source_sha256=dict(self.source.hashes), memory_sha256=dict(self.source_memory_hashes),
                    source_cache_index_sha256=digest(self.cache_index),
                    split=self.canonical.split, dataset_revision=self.canonical.index["history_manifest_sha256"],
                    missing_feature_observations=[k.split(":", 1)[1] for k, v in self.cache_index.items()
                                                  if k.startswith("feature:") and v is None],
                    target_type="annotation-derived class-membership differences, not independently reviewed events",
                    checkpoint_exposure=self.source.summary.get("checkpoint_exposure") or {
                        "status": "unknown", "note": "source run did not record verified VAL/TEST training independence"},
                    encoder=self.encoder, semantic_checkpoint=self.source.contract,
                    predicted_semantics_available=sum((self.frozen.root / "units" / (
                        fusion.PREFIX + "semantic-" + i + ".json")).exists() for i in self.canonical.observations),
                    original_semantics_available=len(self.canonical.index["observations"]),
                    all_equals_at_least_three_observations=all(r["frame"] >= 2 for r in self.samples),
                    valid_pixel_count_policy="original canonical known pixels summed over crops; overlap is merged by evaluator",
                    main_evaluation_support="all methods and stack share first/previous/current fully supported feature patches "
                                            "intersected with existing detector geometry; original-stack parity is separate",
                    geometry_is_occlusion_mask=False, test_interpretation="exploratory; previously inspected")

    def verify(self):
        self.source.verify()
        self.frozen.verify()
        self.saved.verify()


@dataclass
class FrozenNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    count: int
    observation_ids: tuple

    def apply(self, features, support):
        x = np.asarray(features, np.float32)
        shape = (1,) * (x.ndim - 3) + (len(self.mean), 1, 1)
        return np.where(support, (x - self.mean.reshape(shape)) / self.scale.reshape(shape), 0).astype(np.float32)

    def state_dict(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), count=self.count,
                    observation_ids=list(self.observation_ids), fit_split="train", minimum_scale=1e-4,
                    target_space="fixed standardized frozen DINO components; all modes share this transform")

    @classmethod
    def from_state_dict(cls, state):
        if state.get("fit_split") != "train":
            raise ValueError("Feature normalization must have been fitted on TRAIN")
        return cls(np.asarray(state["mean"], np.float32), np.asarray(state["scale"], np.float32),
                   int(state["count"]), tuple(state["observation_ids"]))


def fit_normalizer(dataset):
    """Stable streamed moments, unique TRAIN observations, bounded by one frame."""
    count, mean, m2, ids = 0, None, None, []
    for identifier, features, support in dataset.iter_train_features():
        values = features[:, support[0]].astype(np.float64)
        if not values.shape[1]:
            continue
        n = values.shape[1]
        local_mean = values.mean(1)
        local_m2 = np.square(values - local_mean[:, None]).sum(1)
        if mean is None:
            mean, m2 = local_mean, local_m2
        else:
            delta = local_mean - mean
            m2 += local_m2 + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
        count += n
        ids.append(identifier)
    if not count:
        raise ValueError("No supported TRAIN tokens for normalization")
    return FrozenNormalizer(mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-4).astype(np.float32),
                            count, tuple(ids))


class SyntheticData:
    """Deterministic CPU fixture using the exact spatial/event interface, never evidence."""
    inputs = ScenarioData.inputs
    pair_inputs = ScenarioData.pair_inputs
    forecast_prefix = ScenarioData.forecast_prefix
    feature_target = ScenarioData.feature_target
    feature_targets = ScenarioData.feature_targets
    split_samples = ScenarioData.split_samples
    iter_train_features = ScenarioData.iter_train_features
    annotation_history = ScenarioData.annotation_history
    evaluation_support = ScenarioData.evaluation_support
    evaluation_data = ScenarioData.evaluation_data

    def __init__(self, channels=8, grid=4, buildings_per_split=2, seed=13):
        self.channels, self.grid, self.synthetic = channels, (grid, grid), True
        self.categories = [dict(id=3 + 2 * i, name=n) for i, n in enumerate(EXPECTED_NAMES)]
        self.groups = [dict(name="structural_damage", categories=["CRACK", "SPALLING", "DELAMINATION"])]
        self.tracks, self.samples, self.extra_train, self._features, self._labels, self._rgb = {}, [], [], {}, {}, {}
        self.canonical = self
        rng = np.random.default_rng(seed)
        size, years = grid * 2, (2001, 2004, 2011, 2020)
        for split in ("train", "val", "test"):
            for number in range(buildings_per_split):
                building = f"synthetic-{split}-{number}"
                track = dict(track_id=building + "-track", building_id=building, view_id=building + "-view",
                             split=split, bbox_xywh=[0, 0, size, size], observations=[])
                base = rng.normal(size=(channels, grid, grid)).astype(np.float32)
                for index, year in enumerate(years):
                    identifier = f"{building}-{year}"
                    # Stable -> damage -> repair -> reappearance (one building)
                    # and stable -> damage -> sustained increase (the other).
                    damage = np.zeros((size, size), bool)
                    if index == 1 or (number % 2 == 0 and index == 3) or (number % 2 == 1 and index >= 2):
                        extent = size // 2 if number % 2 == 0 else min(size - 1, size // 2 + max(index - 1, 0))
                        damage[1:extent, 1:extent] = True
                    packed = np.full((size, size), 1 << EXPECTED_NAMES.index("ORNAMENT_INTACT"), np.uint16)
                    packed[damage] |= np.uint16(1 << EXPECTED_NAMES.index("CRACK"))
                    support = np.ones((size, size), bool)
                    support[0, 0] = False
                    packed[0, 0] = 0
                    down = damage.reshape(grid, 2, grid, 2).mean((1, 3))
                    feature = base + down[None] * 1.5 + rng.normal(0, .08, base.shape).astype(np.float32)
                    fs = support.reshape(grid, 2, grid, 2).all((1, 3))[None]
                    self._features[identifier] = (np.where(fs, feature, 0), fs)
                    self._labels[identifier] = dict(packed=packed, support=support)
                    self._rgb[identifier] = np.repeat((30 + 180 * damage)[..., None], 3, axis=-1).astype(np.uint8)
                    track["observations"].append(dict(observation_id=identifier, year=year,
                        image_sha256=digest([identifier, seed]), image_id=identifier))
                self.tracks[track["track_id"]] = track
                for frame in range(1, len(years)):
                    row = dict(event_id=f"{building}-event-{frame}", frame=frame,
                        **{k: track[k] for k in ("track_id", "building_id", "view_id", "split", "bbox_xywh")},
                        previous_year=years[frame - 1], current_year=years[frame])
                    if frame >= 2:
                        self.samples.append(row)
                    elif split == "train":
                        self.extra_train.append({**row, "predictive_train_only": True})
        self.events = {r["event_id"]: r for r in self.samples + self.extra_train}
        for row in self.events.values():
            target = self.targets(row)["target"]
            row.update(known_pixels=int(np.count_nonzero(target != 255)),
                       positive_pixels=int(np.count_nonzero(target == 1)),
                       negative_pixels=int(np.count_nonzero(target == 0)))
        self.assignments = validate_partitions(self.tracks.values())
        self.ontology = ontology_contract(self.categories)

    def feature(self, identifier):
        x, s = self._features[identifier]
        return x.copy(), s.copy()

    def annotation(self, identifier):
        return {k: v.copy() for k, v in self._labels[identifier].items()}

    def targets(self, row):
        observations = self.tracks[row["track_id"]]["observations"]
        before, after = [self.annotation(observations[i]["observation_id"]) for i in (row["frame"] - 1, row["frame"])]
        # Same target_difference used by canonical preparation.
        target, layers, support, _ = target_difference(before, after, range(len(self.categories)), 0)
        return dict(target=target, class_targets=layers, support=support, before=before["packed"], after=after["packed"])

    def target(self, event_id):
        return self.targets(self.events[event_id])

    def scores(self, row):
        # A procedural RGB/feature signal; never read annotations to make a score.
        inputs = self.inputs(row)
        residual = np.abs(inputs["features"][-1] - inputs["features"][-2]).mean(0)
        observations = self.tracks[row["track_id"]]["observations"]
        support = np.logical_and.reduce([self.frame(observations[i]["observation_id"])[1]
                                         for i in (0, row["frame"] - 1, row["frame"])])
        return np.repeat(np.repeat(np.clip(residual, 0, 1), 2, 0), 2, 1), support

    def stack_components(self, row):
        score, support = self.scores(row)
        return score[None], support

    def stack_transform(self, raw):
        return np.asarray(raw)[0]

    @property
    def stack_contract(self):
        return dict(kind="synthetic_identity", method_names=["synthetic"], channels=1)

    def semantic(self, row):
        inputs = self.inputs(row)
        probabilities = np.full((len(inputs["features"]), len(self.categories), *self.grid), .08, np.float32)
        probabilities[:, EXPECTED_NAMES.index("ORNAMENT_INTACT")] = .85
        # Feature-only noisy surrogate semantic head, not direct GT access.
        raw = inputs["features"].mean(1)
        probabilities[:, EXPECTED_NAMES.index("CRACK")] = 1 / (1 + np.exp(-raw))
        p = np.repeat(np.repeat(probabilities, 2, -2), 2, -1)
        m = np.repeat(np.repeat(inputs["support"], 2, -2), 2, -1)
        return dict(probabilities=p, support=m, years=inputs["years"], target_labels_used=False)

    def frame(self, identifier):
        return self._rgb[identifier].copy(), self._labels[identifier]["support"].copy()

    def summary(self, inspect_targets=True):
        return dict(kind="synthetic_history_scenarios_fixture", synthetic=True, seed=13,
            channels=self.channels, grid=list(self.grid), ontology=self.ontology,
            counts={split: dict(buildings=len({r["building_id"] for r in self.split_samples(split)}),
                transitions=len(self.split_samples(split)), predictive_transitions=len(self.split_samples(split, True)),
                valid_pixels=sum(int(np.count_nonzero(self.targets(r)["target"] != 255))
                                 for r in self.split_samples(split)) if inspect_targets else None)
                for split in ("train", "val", "test")},
            target_type="synthetic annotation-derived fixture; no real-facade evidence",
            all_equals_at_least_three_observations=True, test_interpretation="synthetic only")

    def verify(self):
        validate_partitions(self.tracks.values())

    def prepare_missing(self, store, device="cpu"):
        return dict(newly_computed=0, synthetic=True)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        pass
