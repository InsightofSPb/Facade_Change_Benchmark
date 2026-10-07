"""Reuse verified completed benchmark maps; never loads models or checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .benchmark_subset import replay_selection, validate_input_hashes
from .io import read_json, sha256


CASE_IDENTITY_KEYS = (
    "case_id", "base_id", "building_id", "view_id", "split", "state",
    "scenario_id", "nuisance_kind", "hypothesis", "sham_self_paste",
    "full_edit_pixel_count", "visible_edit_pixel_count", "comparable_fraction",
    "retained_visible_edit_fraction", "exclude_from_visible_recall",
    "reference_file_name", "source_file_name", "reference_year", "source_year",
    "source_image_sha256",
)


class ReuseResults:
    """A completed run whose selection, score artifacts and timings are frozen.

    Preflight checks all referenced score/raw files before any new inference.
    A metadata-only archive cannot be used as a cache: the original local run
    must still contain its saved score maps. Individual maps are checked again
    when loaded so a mutation after preflight cannot pass silently.
    """

    def __init__(self, reuse_run_path, exclude_methods=()):
        self.path = Path(reuse_run_path).expanduser().resolve()
        self.record = read_json(self.path / "run.json")
        if self.record.get("kind") != "hypothesis_benchmark" or self.record.get("status") != "completed_exploratory":
            raise ValueError("Reuse requires a completed hypothesis benchmark run")
        self._run_sha256 = sha256(self.path / "run.json")
        self._hashes = self.record.get("artifact_sha256")
        if not isinstance(self._hashes, dict):
            raise ValueError("Completed reuse run has no artifact SHA256 inventory")
        config = self.record.get("config", {})
        serialized = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(serialized).hexdigest() != self.record.get("config_sha256"):
            raise ValueError("Reuse run configuration SHA256 disagrees")
        self.summary = read_json(self._checked("summary.json"))
        self.codec_stats = read_json(self._checked("codec_stats.json")) if "codec_stats.json" in self._hashes else {}
        self.selection = read_json(self._checked("selection.json"))
        metrics = read_json(self._checked("metrics.json"))
        threshold_selection = read_json(self._checked("threshold_selection.json"))
        if self.summary.get("status") != "completed_exploratory":
            raise ValueError("Reuse summary is not completed")
        self.input_sha256 = copy.deepcopy(config.get("input_sha256"))
        validate_input_hashes(self.input_sha256, self.selection.get("input_sha256"))
        self.methods = tuple(config.get("methods", []))
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("Reuse run requires distinct method names")
        self._metadata = copy.deepcopy(config.get("scorers", {}))
        if set(self._metadata) != set(self.methods):
            raise ValueError("Reuse run method/scorer metadata disagree")
        self.thresholds = copy.deepcopy(threshold_selection.get("methods", {}))
        if set(self.thresholds) != set(self.methods):
            raise ValueError("Reuse run has missing or extra frozen thresholds")
        for method, choice in self.thresholds.items():
            if not isinstance(choice, dict):
                raise ValueError("Reuse threshold selection must be a mapping")
            native_mask = self._metadata[method].get("output_kind") == "native_mask"
            original_author_mask = (native_mask and choice.get("selection_split") is None
                                    and choice.get("threshold") == .5
                                    and choice.get("prediction_rule") == "author native binary mask"
                                    and choice.get("criterion") == "author inference; no threshold calibration")
            validation_choice = not native_mask and choice.get("selection_split") == "val"
            if (not (original_author_mask or validation_choice)
                    or choice.get("threshold") != self.summary.get("thresholds", {}).get(method)):
                raise ValueError("Reuse thresholds must be the original validation-only choices")
        self.rows = {}
        cases = metrics.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Reuse metrics has no per-case rows")
        case_ids = self.selection.get("case_ids", [])
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("Reuse selection has invalid case IDs")
        expected = {(method, identifier) for method in self.methods for identifier in case_ids}
        for row in cases:
            key = row.get("method"), row.get("case_id")
            if key in self.rows or key not in expected:
                raise ValueError("Reuse metrics duplicate or change method/case identity")
            if row.get("threshold") != self.thresholds[key[0]]["threshold"]:
                raise ValueError("Reuse case row changes its frozen threshold")
            timing = row.get("scoring_seconds")
            if not isinstance(timing, (int, float)) or isinstance(timing, bool) or not math.isfinite(timing) or timing < 0:
                raise ValueError("Reuse scoring time must be the original finite nonnegative duration")
            self._checked(row.get("score_path"), row.get("score_sha256"))
            if row.get("prediction_path"):
                self._checked(row["prediction_path"], row.get("prediction_sha256"))
            raw_path = self._raw_path(row)
            if raw_path:
                self._checked(raw_path, row.get("raw_score_sha256") or row.get("native_bpb_sha256"))
            if row.get("native_prediction_path"):
                self._checked(row["native_prediction_path"], row.get("native_prediction_sha256"))
            elif self._metadata[key[0]].get("output_kind") == "native_mask":
                raise ValueError("Reused author-mask method is missing its native prediction artifact")
            self.rows[key] = copy.deepcopy(row)
        if set(self.rows) != expected:
            raise ValueError("Reuse run is missing selected case/method results")
        if (self.summary.get("selected_case_count") != len(case_ids)
                or self.summary.get("selected_base_count") != len(self.selection.get("bases", []))
                or self.summary.get("scored_case_method_count") != len(expected)):
            raise ValueError("Reuse run summary/selection result counts disagree")
        # These parent snapshots exist in complete runs. Older metadata exports
        # may omit them; the five input hashes still bind the same parent index.
        for name, key in (("parent_summary.json", "parent_summary"), ("split.json", "split")):
            if (self.path / name).exists():
                self._checked(name, self.input_sha256[key])
        self.exclude(exclude_methods)

    def _checked(self, relative, expected=None):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Reused artifacts must have safe relative paths")
        path = (self.path / relative).resolve()
        if self.path not in path.parents:
            raise ValueError("Reused artifact path escapes its run directory")
        if not path.is_file():
            raise ValueError(f"Reuse requires the complete original run with saved maps; missing artifact: {relative}")
        digest = sha256(path)
        if self._hashes.get(relative) != digest or (expected is not None and expected != digest):
            raise ValueError(f"Reused artifact changed or SHA256 disagrees: {relative}")
        return path

    @property
    def provenance(self):
        return {"run_path": str(self.path), "run_sha256": self._run_sha256,
                "config_sha256": self.record["config_sha256"],
                "selection_sha256": self._hashes["selection.json"],
                "recomputed_methods": list(self.excluded_methods),
                "source_environment": copy.deepcopy(self.record.get("environment", {})),
                "scoring_time_scope": "original inference duration; no inference is performed for reused maps"}

    def exclude(self, methods):
        """Filter only after full source-run validation; preserve its files."""
        methods = tuple(methods)
        if len(set(methods)) != len(methods) or set(methods) - set(self.methods):
            raise ValueError("Recomputed methods must be distinct methods in the reuse run")
        self.excluded_methods = methods
        self.methods = tuple(name for name in self.methods if name not in methods)

    def metadata(self, method):
        return copy.deepcopy(self._metadata[method])

    @staticmethod
    def _raw_path(row):
        generic, bpb = row.get("raw_score_path"), row.get("native_bpb_path")
        if generic and bpb and generic != bpb:
            raise ValueError("Reused raw score and native bits-per-byte paths disagree")
        return generic or bpb

    def validate_selection(self, input_sha256, bases, cases):
        """Reject even accidental subset reuse or a changed case identity."""
        if sha256(self.path / "run.json") != self._run_sha256:
            raise ValueError("Reuse run provenance changed after preflight")
        validate_input_hashes(input_sha256, self.input_sha256)
        bases, cases = list(bases), list(cases)
        saved_bases, saved_cases, _ = replay_selection(
            bases, cases, input_sha256, self._checked("selection.json"))
        if ([row["base_id"] for row in saved_bases] != [row["base_id"] for row in bases]
                or [row["case_id"] for row in saved_cases] != [row["case_id"] for row in cases]):
            raise ValueError("Reuse requires exactly the original ordered bases and cases")
        for case in cases:
            for method in self.methods:
                row = self.rows[method, case["case_id"]]
                if any(row.get(key) != case.get(key) for key in CASE_IDENTITY_KEYS):
                    raise ValueError(f"Reused case metadata differs from its parent: {case['case_id']}")

    def load(self, method, case_id):
        row = self.rows[method, case_id]
        scores = np.load(self._checked(row["score_path"], row.get("score_sha256")), allow_pickle=False)
        raw_path = self._raw_path(row)
        raw = (np.load(self._checked(raw_path, row.get("raw_score_sha256") or row.get("native_bpb_sha256")), allow_pickle=False)
               if raw_path else None)
        if scores.ndim != 2 or scores.dtype != np.float32 or np.isinf(scores).any():
            raise ValueError("Reused scores must be native-grid float32 maps without infinities")
        finite = scores[np.isfinite(scores)]
        if np.any((finite < 0) | (finite > 1)):
            raise ValueError("Reused normalized scores fall outside [0,1]")
        total = row.get("evaluated_pixel_count", 0) + row.get("ignored_pixel_count", 0)
        if total != scores.size:
            raise ValueError("Reused map grid differs from its recorded evaluated/ignored pixels")
        if raw is not None and (raw.shape != scores.shape or raw.dtype != np.float32 or np.isinf(raw).any()
                                or np.any(raw[np.isfinite(raw)] < 0)
                                or not np.array_equal(np.isnan(raw), np.isnan(scores))):
            raise ValueError("Reused native score map disagrees with its normalized grid/support")
        result = {"scores": scores, "raw_scores": raw,
                  "raw_score_units": row.get("raw_score_units", self._metadata[method].get("raw_units")),
                  "scoring_seconds": row["scoring_seconds"], "row": copy.deepcopy(row)}
        if row.get("native_prediction_path"):
            with Image.open(self._checked(row["native_prediction_path"], row.get("native_prediction_sha256"))) as image:
                prediction = np.asarray(image)
            if prediction.shape != scores.shape or prediction.dtype != np.uint8 or not np.isin(prediction, [0, 255]).all():
                raise ValueError("Reused native prediction must be a binary native-grid mask")
            result["native_prediction"] = prediction == 255
            if self._metadata[method].get("output_kind") == "native_mask":
                support = np.isfinite(scores)
                if (not np.isin(scores[support], [0, 1]).all()
                        or not np.array_equal(result["native_prediction"][support], scores[support] > .5)):
                    raise ValueError("Reused author mask disagrees with its binary scores")
        return result
