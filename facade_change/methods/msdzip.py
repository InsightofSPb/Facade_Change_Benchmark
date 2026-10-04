"""Adapter for the existing frozen H0-trained, original MSDZip predictor.

This wrapper does not alter the model, byte ordering, contexts, or score scale.
"""
from __future__ import annotations

import importlib


class MSDZipScorer:
    def __init__(self, representation, **options):
        module = importlib.import_module("..2026-10-04_msdzip_h0", __package__)
        self._scorer = module.MSDZipScorer(representation=representation, **options)
        if self._scorer.metadata.get("representation") != representation:
            raise ValueError(f"Checkpoint representation does not match method msdzip_{representation}")
        self.metadata = self._scorer.metadata
        self.metadata["output_kind"] = "score"
        self._raw_scores = None

    @property
    def raw_scores(self):
        return self._raw_scores

    def __getattr__(self, name):
        # Preserve the legacy scorer's public attributes, including its latest
        # raw_scores, metadata, model, and checkpoint metadata.
        scorer = self.__dict__.get("_scorer")
        if scorer is None:
            raise AttributeError(name)
        return getattr(scorer, name)

    def __call__(self, reference_rgb, source_rgb, geometric_support):
        if self._scorer is None:
            raise RuntimeError("MSDZip scorer has already been closed")
        scores = self._scorer(reference_rgb, source_rgb, geometric_support)
        self._raw_scores = self._scorer.raw_scores
        return scores

    def close(self):
        # The worker owns this predictor; releasing it never touches other GPU
        # processes or any checkpoint files.
        self._scorer = None
