"""CPU-only configuration of the existing semantic/change inference assets."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from .night_common import _current_manifest, load_prepared
from .robustness_common import atomic_json, sha256


CONTEXT_VERSION = "facade_context_v1_2026-10-18"
CONTEXT_CONCEPTS = (
    ("WALL_SURFACE", "plastered exterior wall surface", "structure"),
    ("WINDOW", "window", "structure"),
    ("DOOR", "door", "structure"),
    ("BALCONY", "balcony", "structure"),
    ("DRAINPIPE", "drainpipe", "structure"),
    ("AIR_CONDITIONER", "air conditioning unit", "structure"),
    ("BRICKWORK", "brick wall", "material"),
    ("STONEWORK", "stone wall", "material"),
    ("TREE", "tree", "context"),
    ("VEHICLE", "vehicle", "context"),
    ("SCAFFOLDING", "scaffolding", "context"),
)
DOMAIN_PROMPTS = {
    "CRACK": "crack in a wall",
    "SPALLING": "spalled masonry",
    "DELAMINATION": "peeling plaster",
    "MISSING_ELEMENT": "missing architectural element",
    "WATER_STAIN": "water stain on a wall",
    "EFFLORESCENCE": "white salt deposits on masonry",
    "CORROSION": "rust on metal",
    "ORNAMENT_INTACT": "architectural ornament",
    "ORNAMENT_REGION": "architectural ornament",
    "REPAIRS": "repaired patch on a wall",
    "TEXT_OR_IMAGES": "text or pictures on a wall",
    "ADVERTISEMENTS": "advertisement sign",
}
LEGACY_CHECKPOINTS = {
    "maskclip_head": (
        "outputs/facade_segmentation_group_split_v1_head_allclean_plus1aug/"
        "finetune_tiled_20260530-195724_lr2e-05_depth0_bs128/"
        "epoch_0004_neg_mIoU_no_background_-0.1092.pth"
    ),
    "maskclip_depth1": (
        "outputs/facade_segmentation_group_split_v1_depth1_damage/"
        "finetune_tiled_20260530-212408_lr2e-05_depth1_bs64/"
        "epoch_0003_neg_DAMAGE_MACRO_MIOU_-0.0822.pth"
    ),
}



# Known sibling checkouts only; no recursive home-directory search or cloning.
STOCK_SIBLING_NAMES = ("LPOSS_pr58_cuda_20260820", "Compress_to_prevent")
STOCK_REQUIRED_FILES = (
    "ovs_heritage/__init__.py",
    "ovs_heritage/stock_features.py",
    "ovs_heritage/stock_lposs.py",
    "ovs_heritage/vocabulary.py",
    "ovs_heritage/ontology.py",
    "ovs_heritage/projection.py",
    "ovs_heritage/scoring.py",
    "segmentation/evaluation/lposs_eval.py",
)


def inspect_stock_root(root, model_config="configs/stock_lposs.yaml"):
    """Check the local stock configuration and source files, without importing them."""
    root = Path(root).expanduser().resolve()
    config = Path(model_config).expanduser()
    config = (config if config.is_absolute() else root / config).resolve()
    expected = [("model_config", config)] + [("stock_source", root / name) for name in STOCK_REQUIRED_FILES]
    missing = [{"field": field, "path": str(path)} for field, path in expected if not path.is_file()]
    return {"root": str(root), "model_config": str(config), "missing": missing,
            "status": "files_present_unverified" if not missing else "missing_stock_files",
            "runtime_and_weights_validated": False}


def resolve_stock_root(lposs_root, explicit=None, model_config="configs/stock_lposs.yaml"):
    """Resolve only an existing checkout; an explicit root always takes precedence.

    A missing/ambiguous automatic lookup retains the declared root with precise
    diagnostics. It never claims that a historical sibling currently exists.
    """
    declared = Path(lposs_root).expanduser().resolve()
    if explicit is not None:
        selected = Path(explicit).expanduser().resolve()
        inspected = inspect_stock_root(selected, model_config)
        return selected, {"explicit": True, "status": "explicit", "selected_root": str(selected),
                          "candidates": [inspected]}
    inspected = inspect_stock_root(declared, model_config)
    if not inspected["missing"]:
        return declared, {"explicit": False, "status": "declared_root_present", "selected_root": str(declared),
                          "candidates": [inspected]}
    candidates = [inspected]
    seen = {str(declared)}
    for name in STOCK_SIBLING_NAMES:
        candidate = (declared.parent / name).resolve()
        if str(candidate) not in seen:
            seen.add(str(candidate))
            candidates.append(inspect_stock_root(candidate, model_config))
    usable = [candidate for candidate in candidates if not candidate["missing"]]
    if len(usable) == 1:
        selected = Path(usable[0]["root"])
        return selected, {"explicit": False, "status": "existing_sibling_discovered", "selected_root": str(selected),
                          "candidates": candidates}
    return declared, {"explicit": False, "status": "ambiguous" if usable else "missing",
                      "selected_root": None, "retained_root": str(declared), "candidates": candidates,
                      "note": "Укажите --stock-root: подходящий checkout не выбран однозначно."}


def _asset_diagnostics(segmenters):
    result = {}
    for name, spec in segmenters.items():
        missing = []
        for key in ("worker_python", "source_root", "checkpoint", "training_run", "current_manifest"):
            if key in spec and (not spec[key] or not Path(spec[key]).exists()):
                missing.append({"field": key, "path": spec[key]})
        if spec.get("kind") in {"stock_maskclip", "stock_lposs"} and spec.get("source_root"):
            missing.extend(inspect_stock_root(spec["source_root"], spec.get("model_config", "configs/stock_lposs.yaml"))["missing"])
        result[name] = {"status": "missing_declared_assets" if missing else "paths_present_unverified",
                        "missing": missing, "runtime_and_weights_validated": False}
    return result


def repair_config(config_path, out, stock_root=None, *, trust_lposs_checkpoint=False):
    """Copy a configuration, repairing stock paths or explicitly trusting local DINO weights.

    Custom settings, trained checkpoint roots, vocabularies and cache roots are
    retained verbatim. The source config is never overwritten. Explicit roots
    recorded by configure remain explicit; a new stock_root argument is an
    intentional override. Legacy configs are auto-repaired only when the stock
    root equals the trained LPOSS root, the earlier configure default. A trust-
    only repair preserves every root and other setting; root discovery runs
    only when requested by stock_root or when no trust-only action was supplied.
    """
    source = Path(config_path).expanduser().resolve(strict=True)
    destination = Path(out).expanduser().resolve()
    if destination == source or destination.exists():
        raise FileExistsError(f"Выберите новое имя конфигурации: {destination}")
    config = json.loads(source.read_text(encoding="utf-8"))
    updated = copy.deepcopy(config)
    segmenters = updated.get("segmenters", {})
    if not isinstance(segmenters, dict):
        raise ValueError("Конфигурация не содержит словарь segmenters")
    trust_repairs = []
    if trust_lposs_checkpoint:
        for name, spec in segmenters.items():
            if spec.get("kind") == "stock_lposs":
                trust_repairs.append({"segmenter": name, "previous": spec.get("trust_checkpoint"),
                                      "trust_checkpoint": True})
                spec["trust_checkpoint"] = True
        if not trust_repairs:
            raise ValueError("В конфигурации нет stock_lposs для --trust-lposs-checkpoint")
    repair_roots = stock_root is not None or not trust_lposs_checkpoint
    legacy_roots = {str(Path(spec["source_root"]).expanduser().resolve()) for spec in segmenters.values()
                    if spec.get("kind") == "finetuned_maskclip" and spec.get("source_root")}
    prior = updated.get("source", {}).get("stock_root_resolution", {})
    repairs = []
    for name, spec in (segmenters.items() if repair_roots else ()):
        if spec.get("kind") not in {"stock_maskclip", "stock_lposs"}:
            continue
        original = spec.get("source_root")
        if not original:
            repairs.append({"segmenter": name, "status": "missing_source_root"})
            continue
        model_config = spec.get("model_config", "configs/stock_lposs.yaml")
        current = Path(original).expanduser().resolve()
        inspection = inspect_stock_root(current, model_config)
        if stock_root is not None:
            selected, resolution = resolve_stock_root(current, stock_root, model_config)
        elif prior.get("explicit"):
            selected, resolution = current, {"status": "explicit_root_preserved", "explicit": True, "candidates": [inspection]}
        elif not inspection["missing"]:
            selected, resolution = current, {"status": "valid_root_preserved", "candidates": [inspection]}
        elif str(current) not in legacy_roots:
            selected, resolution = current, {"status": "custom_root_preserved", "candidates": [inspection]}
        else:
            selected, resolution = resolve_stock_root(current, model_config=model_config)
        if selected != current:
            spec["source_root"] = str(selected)
        repairs.append({"segmenter": name, "old_root": original, "new_root": spec["source_root"], **resolution})
    if repair_roots:
        updated["configure_diagnostics"] = _asset_diagnostics(segmenters)
        updated.setdefault("source", {})["stock_root_repair"] = {
            "source_config": str(source), "source_config_sha256": sha256(source), "models": repairs,
        }
    if trust_repairs:
        updated.setdefault("source", {})["checkpoint_trust_repair"] = {
            "source_config": str(source), "source_config_sha256": sha256(source), "models": trust_repairs,
            "scope": "explicit legacy loading of existing local stock LPOSS DINO weights; no download",
        }
    if stock_root is not None:
        updated["source"]["stock_root_resolution"] = {"explicit": True, "status": "explicit_repair",
                                                       "selected_root": str(Path(stock_root).expanduser().resolve())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, updated)
    print(f"Конфигурация: {destination}")
    for row in repairs:
        print(f"{row['segmenter']}: {row['status']}; {row.get('new_root', '')}")
    for row in trust_repairs:
        print(f"{row['segmenter']}: trust_checkpoint=true для существующих локальных весов DINO")
    return updated

def add_configure_parser(subparsers):
    parser = subparsers.add_parser("configure", help="Собрать конфигурацию без загрузки моделей")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-run", action="append", default=[])
    parser.add_argument("--lposs-root", default="/home/sasha/LPOSS")
    parser.add_argument("--lposs-python")
    parser.add_argument("--stock-root", help="Отдельный checkout с ovs_heritage, если нужен")
    parser.add_argument("--trust-lposs-checkpoint", action="store_true",
                        help="Разрешить legacy-загрузку существующих локальных весов DINO для stock LPOSS")
    parser.add_argument("--sam3-python", default="/home/sasha/miniconda3/envs/sam3_facades_20260907/bin/python")
    parser.add_argument("--sam3-checkpoint", default="models/sam3.pt")
    parser.add_argument("--expanded-vocabulary", help="Точный пользовательский словарь JSON/YAML")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _absolute(value, base=None):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else (base or Path.cwd()) / path).resolve())


def _vocabularies(categories, expanded_path):
    names = [category["name"] for category in categories]
    if (any(not isinstance(name, str) or not name.strip() for name in names)
            or len({name.casefold() for name in names}) != len(names)):
        raise ValueError("Названия категорий должны быть непустыми и уникальными")
    domain = [{"name": name, "prompts": [DOMAIN_PROMPTS.get(name.upper(), name.lower().replace("_", " "))],
               "group": "domain"} for name in names]
    if expanded_path is None:
        existing = {name.casefold() for name in names}
        extra = [{"name": name, "prompts": [prompt], "group": group}
                 for name, prompt, group in CONTEXT_CONCEPTS if name.casefold() not in existing]
        return domain, domain + extra, {
            "name": CONTEXT_VERSION, "origin": "new_explicit_context_vocabulary",
            "historical_context60_reproduced": False,
            "note": "Новый словарь; это не восстановленный heritage_context60.",
        }
    path = Path(expanded_path).expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload
    if isinstance(payload, dict):
        entries = payload.get("concepts", payload.get("classes"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Словарь должен содержать непустой список concepts/classes")
    normalized, seen = [], set()
    for item in entries:
        if isinstance(item, str):
            item = {"name": item, "prompts": [item.lower().replace("_", " ")]}
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError("Каждому понятию требуется непустое name")
        name = item["name"].strip()
        if name.casefold() in seen:
            raise ValueError(f"Повтор понятия в расширенном словаре: {name}")
        seen.add(name.casefold())
        prompts = item.get("prompts", [name.lower().replace("_", " ")])
        if (not isinstance(prompts, list) or not prompts
                or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)):
            raise ValueError(f"Понятие {name}: prompts должен быть непустым списком строк")
        group = item.get("group", "context")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Понятие {name}: требуется непустая group")
        normalized.append({"name": name, "prompts": [prompt.strip() for prompt in prompts], "group": group})
    supplied = {item["name"].casefold(): item for item in normalized}
    # The same base prompts are used in both conditions; extension alone changes.
    domain = [{**item, "prompts": supplied.get(item["name"].casefold(), item)["prompts"]} for item in domain]
    base_names = {item["name"].casefold() for item in domain}
    extra = [item for item in normalized if item["name"].casefold() not in base_names]
    if not extra:
        raise ValueError("Расширенный словарь не содержит дополнительных понятий")
    return domain, domain + extra, {
        "name": payload.get("name", path.stem) if isinstance(payload, dict) else path.stem,
        "origin": "user_file", "path": str(path), "sha256": sha256(path),
        "historical_context60_reproduced": None,
        "note": "Использован указанный файл; происхождение исторического словаря не предполагается.",
    }


def _semantic_options(manifest):
    config = manifest.get("source_config", {})
    options = dict(config.get("method_options", {}).get("semantic_dino", {}))
    provenance_path = Path(manifest["source_run"]) / "method_provenance" / "full_semantic_dino.json"
    rows = manifest.get("source_records", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    saved = [row.get("method_metadata", {}) for row in rows
             if row.get("method") == "semantic_dino" and row.get("status") == "completed"
             and row.get("variant", "frozen") == "frozen"]
    if provenance_path.is_file():
        saved.append(json.loads(provenance_path.read_text(encoding="utf-8")))
    for provenance in saved:
        configured = provenance.get("configured_options", {})
        for key in ("training_run", "worker_python"):
            value = configured.get(key)
            if value is not None:
                if options.get(key) and options[key] != value:
                    raise ValueError(f"semantic_dino: конфликт сохранённых настроек {key}")
                options[key] = value
    for provenance in saved:
        for key in ("training_run", "worker_python"):
            if not options.get(key) and provenance.get(key):
                options[key] = provenance[key]
        executable = provenance.get("worker_environment", {}).get("executable")
        if executable and not options.get("worker_python"):
            options["worker_python"] = executable
    if not options.get("training_run") and config.get("semantic_run"):
        options["training_run"] = config["semantic_run"]
    if not options.get("worker_python"):
        executable = config.get("method_options", {}).get("dinov2", {}).get("worker_python")
        if executable:
            options["worker_python"] = executable
    return options


def _lposs_python(value, root):
    if value:
        return _absolute(value)
    current = Path(sys.executable).resolve()
    candidates = []
    if "lposs" in current.parent.parent.name.lower():
        candidates.append(current)
    candidates.extend((current.parent.parent.parent / "lposs" / "bin" / "python",
                       root.parent / "miniconda3" / "envs" / "lposs" / "bin" / "python"))
    return str(next((path for path in candidates if path.is_file()), candidates[-1]).resolve())


def configure(args):
    """Write a reviewable config; retain missing declared models as explicit entries."""
    prepared, manifest = load_prepared(args.prepared)
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Конфигурация уже существует, выберите новое имя: {out}")
    repo = Path(__file__).resolve().parents[2]
    lposs_root = Path(args.lposs_root).expanduser().resolve()
    stock_root, stock_resolution = resolve_stock_root(lposs_root, args.stock_root)
    lposs_python = _lposs_python(args.lposs_python, lposs_root)
    device = args.device
    semantic_options = _semantic_options(manifest)
    semantic = {"kind": "dino_semantic", "device": device, "batch_size": 1, "variants": ["fixed"],
                "confidence": .5,
                "worker_python": _absolute(semantic_options.get("worker_python") or sys.executable),
                "training_run": _absolute(semantic_options["training_run"], Path(manifest["source_run"]))
                if semantic_options.get("training_run") else None}
    current_manifest = semantic_options.get("current_manifest") or _current_manifest(manifest)
    if current_manifest:
        semantic["current_manifest"] = _absolute(current_manifest, Path(manifest["source_run"]))
    semantic["expected_categories"] = manifest["categories"]
    segmenters = {"dino_semantic": semantic}
    for name, relative in LEGACY_CHECKPOINTS.items():
        segmenters[name] = {"kind": "finetuned_maskclip", "source_root": str(lposs_root),
                            "checkpoint": str(lposs_root / relative), "worker_python": lposs_python,
                            "device": device, "mix_strategy": "add", "use_embedding_mixer": False,
                            "input_color_order": "bgr", "trust_checkpoint": True, "variants": ["fixed"]}
    variants = ["domain", "expanded_flat", "expanded_layered"]
    for name, kind in (("maskclip_stock", "stock_maskclip"), ("lposs_stock", "stock_lposs")):
        segmenters[name] = {"kind": kind, "source_root": str(stock_root), "worker_python": lposs_python,
                            "model_config": "configs/stock_lposs.yaml", "device": device,
                            "clip_checkpoint": "auto", "softmax_scale": 100., "variants": variants.copy()}
        if kind == "stock_lposs":
            segmenters[name].update(dino_source_root="auto", dino_checkpoint="auto")
            if getattr(args, "trust_lposs_checkpoint", False):
                segmenters[name]["trust_checkpoint"] = True
    segmenters["sam3"] = {"kind": "sam3", "worker_python": _absolute(args.sam3_python),
                           "checkpoint": _absolute(args.sam3_checkpoint, repo), "device": device,
                           "confidence_threshold": .25, "mask_threshold": .5, "precision": "bfloat16",
                           "variants": variants.copy()}
    domain, expanded, vocabulary_origin = _vocabularies(manifest["categories"], args.expanded_vocabulary)
    diagnostics = _asset_diagnostics(segmenters)
    reuse = args.reuse_run or []
    if isinstance(reuse, str):
        reuse = [reuse]
    config = {
        "schema_version": 1, "prepared": str(prepared), "device": device,
        "reuse_runs": [_absolute(path) for path in reuse],
        "splits": ["val", "test", "train"], "modes": ["previous"],
        "change_methods": {name: {} for name in ("anychange", "dinov2", "semantic_dino")},
        "segmenters": segmenters, "vocabularies": {"domain": domain, "expanded": expanded},
        "vocabulary_provenance": vocabulary_origin,
        "routes": ["change_first", "semantic_first_gate", "semantic_first_rerun"],
        "roi": {"min_area": 16, "max_regions": 8, "padding": 16, "min_size": 56},
        "semantic": {"confidence": .25}, "smoke": {"pairs_per_split": 2},
        "cache_root": str(prepared.parent / "semantic-cache-v1"),
        "configure_diagnostics": diagnostics,
        "source": {"prepared_manifest_sha256": sha256(prepared / "manifest.json"),
                   "source_run": manifest["source_run"], "model_selection": "existing frozen checkpoints; no training",
                   "stock_root_resolution": stock_resolution},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, config)
    unavailable = [name for name, value in diagnostics.items() if value["missing"]]
    print(f"Конфигурация: {out}")
    print(f"Сегментаторы: {', '.join(segmenters)}; словарь: {vocabulary_origin['name']}")
    print(f"Stock checkout: {stock_root} ({stock_resolution['status']})")
    if unavailable:
        print(f"Нужна проверка указанных путей: {', '.join(unavailable)}. Модели сохранены в конфигурации.")
    return config
