"""Merge common-runner JSON configuration with explicit command-line overrides."""
from __future__ import annotations

import inspect
from pathlib import Path

from .io import read_json


def benchmark_arguments(arguments):
    from .hypothesis_benchmark import run_hypothesis_benchmark
    arguments = dict(arguments)
    config_path = arguments.pop("benchmark_config", None)
    config_file = Path(config_path).expanduser().resolve() if config_path else None
    config = read_json(config_file) if config_file else {}
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a JSON object")
    config = dict(config)
    if config_file:
        for key in ("dataset_run", "out", "selection_path", "reuse_run", "methods_config",
                    "msdzip_abs_checkpoint", "msdzip_mod256_checkpoint"):
            if config.get(key):
                config[key] = _path(config[key], config_file.parent)
    options = {**config, **arguments}
    methods_config = options.pop("methods_config", None)
    allowed = set(inspect.signature(run_hypothesis_benchmark).parameters)
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark configuration keys: {', '.join(sorted(unknown))}")
    if not options.get("dataset_run") or not options.get("out"):
        raise ValueError("h0h1-benchmark requires dataset_run and out via flags or --config")
    if methods_config:
        methods_file = Path(methods_config).expanduser().resolve()
        method_data = read_json(methods_file)
        if not isinstance(method_data, dict) or set(method_data) != {"methods"} or not isinstance(method_data["methods"], dict):
            raise ValueError("Methods configuration must contain one 'methods' object")
        local_methods = {}
        for method, values in method_data["methods"].items():
            if not isinstance(values, dict):
                raise ValueError(f"Method options must be an object: {method}")
            local_methods[method] = {
                key: _path(value, methods_file.parent) if key in {
                    "worker_python", "source_root", "checkpoint_path", "dino_root", "dino_checkpoint",
                    "py_utils_root", "sam_checkpoint", "backbone_checkpoint", "calibration_checkpoint",
                    "training_run", "model_config"
                } and value else value for key, value in values.items()}
        options["method_options"] = {**local_methods, **options.get("method_options", {})}
    return options


def _path(value, directory):
    if not isinstance(value, str):
        raise ValueError("Configured paths must be strings")
    value = Path(value).expanduser()
    return str((directory / value).resolve() if not value.is_absolute() else value.resolve())
