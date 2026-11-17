"""Inspection admission policy over the shared physical-identity validator."""
from importlib import import_module


_shared = import_module(".2026-11-16_www_identity", __package__)
template = _shared.template


class Identities(_shared.Identities):
    """Keep known aliases pending until reviewed; never infer a physical merge."""

    def __init__(self, path=None, fingerprints=None, expected=None):
        super().__init__(path, fingerprints, expected, pending_candidates=True)
