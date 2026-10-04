"""Official spatial LPIPS 0.1.4 with local pretrained weights and fixed scaling.

No model downloads occur during a benchmark. The official ImageNet AlexNet trunk
and bundled v0.1 calibration state are loaded explicitly and strictly before any
inference, so missing weights cannot silently produce an untrained baseline.
"""
from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path

import numpy as np

from .base import file_provenance, mask_scores, validate_rgb_inputs


LPIPS_PACKAGE_VERSION = "0.1.4"
ALEXNET_FILENAME = "alexnet-owt-7be5be79.pth"


def _safe_state(path, trust_checkpoint=False):
    import torch

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        if not trust_checkpoint:
            raise RuntimeError("This PyTorch lacks weights_only loading; use explicit trust_checkpoint for a trusted checkpoint") from exc
        state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise RuntimeError("LPIPS weights must contain only a nonempty tensor state_dict")
    return state


def _local_weights(package, backbone_checkpoint, calibration_checkpoint):
    import torch

    backbone = (Path(backbone_checkpoint).expanduser() if backbone_checkpoint else
                Path(torch.hub.get_dir()) / "checkpoints" / ALEXNET_FILENAME).resolve()
    calibration = (Path(calibration_checkpoint).expanduser() if calibration_checkpoint else
                   Path(package.__file__).resolve().parent / "weights" / "v0.1" / "alex.pth").resolve()
    for label, path in (("ImageNet AlexNet backbone", backbone), ("official LPIPS v0.1 calibration", calibration)):
        if not path.is_file():
            raise FileNotFoundError(f"Local {label} missing: {path}; prepare weights before benchmark (automatic downloads disabled)")
    return backbone, calibration


def _load_pretrained_states(model, backbone_state, calibration_state):
    """Populate every learned tensor using the official released state layouts.

    LPIPS wraps original AlexNet feature modules in five slices while retaining
    their global torchvision layer indices. Its calibration linears are also
    aliased in ``lins``; both state_dict names are populated with the same tensor.
    Scaling-layer buffers are fixed official constants, not trained parameters.
    """
    import torch

    trunk = {}
    required_features = set()
    for key, target in model.net.state_dict().items():
        parts = key.split(".")
        if len(parts) != 3 or not parts[0].startswith("slice"):
            raise RuntimeError(f"Unexpected LPIPS AlexNet trunk parameter: {key}")
        source_key = "features." + ".".join(parts[1:])
        required_features.add(source_key)
        value = backbone_state.get(source_key)
        if (not torch.is_tensor(value) or value.shape != target.shape
                or not torch.isfinite(value).all()):
            raise RuntimeError(f"Missing or invalid pretrained AlexNet tensor: {source_key}")
        trunk[key] = value
    features = {key for key in backbone_state if key.startswith("features.")}
    if features != required_features:
        raise RuntimeError("AlexNet checkpoint feature layout does not match official LPIPS trunk")
    if not trunk:
        raise RuntimeError("LPIPS AlexNet trunk has no learned tensors")
    model.net.load_state_dict(trunk, strict=True)

    required_calibration = {f"lin{index}.model.1.weight" for index in range(5)}
    if set(calibration_state) != required_calibration:
        raise RuntimeError("LPIPS calibration must contain the five official v0.1 AlexNet linear weights")
    full_state = model.state_dict()
    for index in range(5):
        key = f"lin{index}.model.1.weight"
        value = calibration_state[key]
        if (not torch.is_tensor(value) or value.shape != full_state[key].shape
                or not torch.isfinite(value).all() or (value < 0).any()):
            raise RuntimeError(f"Invalid official LPIPS calibration tensor: {key}")
        full_state[key] = value
        alias = f"lins.{index}.model.1.weight"
        if alias not in full_state:
            raise RuntimeError("LPIPS package calibration alias layout differs from version 0.1.4")
        full_state[alias] = value
    parameter_names = {name for name, _ in model.named_parameters()}
    learned_names = {"net." + key for key in trunk} | required_calibration
    if parameter_names != learned_names:
        raise RuntimeError("LPIPS contains unaccounted learned parameters; refusing random-weight inference")
    model.load_state_dict(full_state, strict=True)


def _construct_model(package, backbone, calibration, device, trust_checkpoint):
    # These two flags disable both torchvision downloads and LPIPS's implicit
    # pickle loader. Every random constructor parameter is then replaced by the
    # released pretrained tensors before the model can perform inference.
    model = package.LPIPS(net="alex", version="0.1", lpips=True, spatial=True,
                          pretrained=False, pnet_rand=True, pnet_tune=False,
                          use_dropout=True, eval_mode=True, verbose=False)
    _load_pretrained_states(model, _safe_state(backbone, trust_checkpoint),
                            _safe_state(calibration, trust_checkpoint))
    model.pnet_rand = False
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _input_tensor(image, device):
    import torch

    return (torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
            .unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0)


def distance_to_scores(distance):
    """Fixed monotone mapping, without pair-wise min/max or quantile scaling."""
    distance = np.asarray(distance)
    if distance.ndim != 2 or not np.isfinite(distance).all() or np.any(distance < 0):
        raise RuntimeError("LPIPS returned invalid native-grid perceptual distances")
    return (distance.astype(np.float64) / (1.0 + distance)).astype(np.float32)


class LPIPSScorer:
    """Frozen official AlexNet spatial LPIPS; raw distances remain available."""

    def __init__(self, **options):
        allowed = {"network", "backbone_checkpoint", "calibration_checkpoint", "device", "trust_checkpoint"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown LPIPS options: {', '.join(sorted(unknown))}")
        if options.get("network", "alex") != "alex":
            raise ValueError("This baseline fixes official LPIPS network='alex'")
        try:
            version = metadata.version("lpips")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("LPIPS requires the official package: pip install lpips==0.1.4") from exc
        if version != LPIPS_PACKAGE_VERSION:
            raise RuntimeError(f"This baseline requires lpips=={LPIPS_PACKAGE_VERSION}; found {version}")
        package = importlib.import_module("lpips")
        backbone, calibration = _local_weights(
            package, options.get("backbone_checkpoint"), options.get("calibration_checkpoint"))
        weight_records = {"backbone": file_provenance(backbone), "calibration": file_provenance(calibration)}
        package_root = Path(package.__file__).resolve().parent
        source_records = [file_provenance(package_root / name) for name in
                          ("__init__.py", "lpips.py", "pretrained_networks.py")]
        device = str(options.get("device", "cpu"))
        self.model = _construct_model(package, backbone, calibration, device,
                                      bool(options.get("trust_checkpoint", False)))
        self.device = device
        self.native_prediction = None
        self.raw_scores = None
        self.metadata = {
            "method": "lpips", "implementation_version": 1, "output_kind": "score",
            "package": {"name": "lpips", "version": version, "source_files": source_records},
            "architecture": "official LPIPS v0.1; ImageNet pretrained AlexNet; spatial=True",
            "weights": weight_records, "device": device,
            "input": "native uint8 RGB -> float32 [-1,1]; no external resize or normalization",
            "settings": {"net": "alex", "version": "0.1", "spatial": True,
                         "lpips": True, "normalize": False, "frozen": True, "eval": True},
            "raw_units": "LPIPS perceptual distance", "raw_score_formula": "official spatial LPIPS distance d",
            "score_formula": "d / (1 + d); fixed monotone conversion, not a probability",
            "output": "native float32 [0,1]; NaN outside base geometric support",
            "support": "base geometric support only; no labels, nuisance, or edit visibility",
            "downloads": "disabled; explicit local backbone and calibration state loading",
            "primary_source": "https://github.com/richzhang/PerceptualSimilarity",
        }

    def __call__(self, reference, source, support):
        import torch

        reference, source, support = validate_rgb_inputs(reference, source, support)
        self.raw_scores = None
        self.native_prediction = None
        with torch.inference_mode():
            output = self.model(_input_tensor(reference, self.device),
                                _input_tensor(source, self.device), normalize=False)
            if tuple(output.shape) != (1, 1, *reference.shape[:2]):
                raise RuntimeError("Official spatial LPIPS must return the unchanged native HxW grid")
            raw = output[0, 0].detach().cpu().numpy().astype(np.float32)
        scores = mask_scores(distance_to_scores(raw), support)
        self.raw_scores = raw.copy()
        self.raw_scores[~support] = np.nan
        return scores
