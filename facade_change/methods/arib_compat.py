"""Source-preserving numeric compatibility for original ArIB bits-back decoding."""
from __future__ import annotations

from copy import deepcopy


_GUARD = "_facade_arib_posterior_rgb_compatibility"


def canonical_normalized_rgb(image):
    """Recover the original uint8/255 context from accumulated decoded bitplanes.

    The author encoder computes uint8 significant planes divided by 255, while
    the decoder sums independently divided bitplanes. One float32 ULP can alter
    posterior CDF entries and prevent restoration of the bits-back state.
    """
    import torch

    if image.dtype != torch.float32 or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("ArIB posterior compatibility expects float32 BCHW normalized RGB")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("ArIB posterior context must contain finite normalized RGB in [0,1]")
    return torch.round(image * 255).to(torch.uint8).float() / 255.0


class PosteriorRGBCompatibility:
    """Reversible wrapper around only the author's posterior restoration method."""

    def __init__(self, model):
        self._posterior = model.sig.lvae
        self._original = self._posterior._compress_qz
        self._closed = False
        self._metadata = {
            "name": "canonical uint8 posterior RGB context",
            "implementation_version": 1,
            "scope": "decoder-only sig.lvae._compress_qz input; original entropy coder and architecture unchanged",
            "formula": "round(decoded_normalized_RGB * 255).to(uint8).float() / 255.0",
            "reason": "match encoder uint8/255 exactly before posterior CDF restoration; avoid float32 bitplane accumulation drift",
            "encoder": "unchanged; no wrapper is called during compress_to_file",
            "source_files_modified": False,
            "verification": "every actual bitstream must still reconstruct all original RGB bytes exactly",
        }

        def restore_posterior(image, z_ids, mups, inputs, coder):
            return self._original(canonical_normalized_rgb(image), z_ids, mups, inputs, coder)

        self._wrapper = restore_posterior
        self._posterior._compress_qz = self._wrapper
        setattr(self._posterior, _GUARD, self)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def close(self):
        """Restore the original method once, without overwriting a later wrapper."""
        if self._closed:
            return
        if self._posterior._compress_qz is not self._wrapper:
            raise RuntimeError("ArIB posterior method changed after compatibility installation")
        self._posterior._compress_qz = self._original
        delattr(self._posterior, _GUARD)
        self._closed = True


def install_posterior_rgb_compatibility(model):
    """Install once per model; repeated calls return the existing cleanup handle."""
    posterior = model.sig.lvae
    existing = getattr(posterior, _GUARD, None)
    if existing is not None:
        if not isinstance(existing, PosteriorRGBCompatibility):
            raise RuntimeError("Unexpected ArIB posterior compatibility guard")
        if posterior._compress_qz is not existing._wrapper:
            raise RuntimeError("ArIB posterior method changed after compatibility installation")
        return existing
    return PosteriorRGBCompatibility(model)
