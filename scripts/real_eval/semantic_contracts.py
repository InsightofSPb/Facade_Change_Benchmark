"""Recover frozen method provenance from explicitly declared wide-run caches.

Only metadata and recorded model files are read. RGB/map compatibility remains
the responsibility of the existing cache loader, immediately before reuse.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .robustness_common import digest, read_json, sha256
from .robustness_run import _file_hashes, _method_contract, _source_rows, _verify_metadata, model_identity


_FORBIDDEN = ('download', 'download_weights', 'download-weights', 'train', 'auto_train')


def _options_agree(expected, actual, label):
    """Device is the sole explicitly supported runtime override here."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError(f'{label}: options must be dictionaries')
    for key, value in expected.items():
        if key != 'device' and key in actual and actual[key] != value:
            raise ValueError(f'{label}: conflicting recorded option {key}')


def _has_source_provenance(method, manifest):
    path = Path(manifest['source_run']).expanduser().resolve() / 'method_provenance' / f'full_{method}.json'
    if path.exists():
        return True
    variant = 'current_train' if method.startswith('msdzip_') else 'frozen'
    return any(row.get('method') == method and row.get('status') == 'completed'
               and row.get('variant', 'frozen') == variant and row.get('method_metadata')
               for row in _source_rows(manifest))


def _candidate(method, manifest, device, root, prepared):
    root = Path(root).expanduser().resolve()
    run_path = root / 'run.json'
    run = read_json(run_path)
    contract = run.get('contract')
    if run.get('kind') != 'real_temporal_run' or not isinstance(contract, dict):
        raise ValueError(f'{root}: expected a real_temporal_run with a saved contract')
    if not contract.get('source_run') or (Path(contract['source_run']).expanduser().resolve()
                                        != Path(manifest['source_run']).expanduser().resolve()):
        raise ValueError(f'{root}: reuse run belongs to a different source_run')
    if run.get('config_hash') is not None and run['config_hash'] != digest(contract):
        raise ValueError(f'{root}: saved run contract checksum changed')
    prepared_path = Path(prepared or contract.get('prepared_path', '')).expanduser().resolve()
    if prepared_path.is_dir():
        prepared_path = prepared_path / 'manifest.json'
    if (not prepared_path.is_file() or sha256(prepared_path) != contract.get('manifest_sha256')
            or read_json(prepared_path) != manifest):
        raise ValueError(f'{root}: reuse run belongs to a different or changed prepared manifest')
    if method not in contract.get('methods', []):
        raise ValueError(f'{root}: method {method} was not declared in the saved run')

    variant = 'current_train' if method.startswith('msdzip_') else 'frozen'
    rows, evidence = [], []
    for path in sorted((root / 'units').glob('*.json')):
        row = read_json(path)
        if (row.get('method') == method and row.get('status') == 'completed'
                and row.get('variant', 'frozen') == variant):
            metadata = row.get('method_metadata')
            if not isinstance(metadata, dict) or not metadata:
                raise ValueError(f'{path}: completed unit has no method metadata')
            rows.append(row)
            evidence.append({'path': str(path), 'sha256': sha256(path)})
    provenance_path = root / 'method_provenance' / f'{method}.json'
    if not provenance_path.is_file():
        provenance_path = root / 'method_provenance' / f'full_{method}.json'
    provenance = read_json(provenance_path) if provenance_path.is_file() else None
    if provenance is not None and (not isinstance(provenance, dict) or not provenance):
        raise ValueError(f'{provenance_path}: invalid saved method provenance')
    original_methods = manifest['source_config'].get('method_options', {})
    original = original_methods.get(method, {})
    model = contract.get('models', {}).get(method)
    if model is not None:
        if not isinstance(model, dict) or not isinstance(model.get('options'), dict):
            raise ValueError(f'{root}: invalid saved model contract for {method}')
        expected = copy.deepcopy(model.get('expected_metadata'))
        options = copy.deepcopy(model['options'])
    else:
        if rows and isinstance(rows[0]['method_metadata'].get('configured_options'), dict):
            expected = copy.deepcopy(rows[0]['method_metadata'])
            options = copy.deepcopy(expected['configured_options'])
        elif provenance is not None and (isinstance(provenance.get('configured_options'), dict)
                                         or method in original_methods):
            expected = copy.deepcopy(provenance)
            options = copy.deepcopy(provenance.get('configured_options', original))
        else:
            raise ValueError(f'{root}: no saved model contract or completed configured model metadata for {method}')
        model = {}
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f'{root}: empty expected model metadata for {method}')
    if any(options.get(key) for key in _FORBIDDEN):
        raise ValueError(f'{root}: training/downloads are forbidden')

    _options_agree(original, options, str(root))
    # Retain explicit source settings omitted by an older runtime's metadata.
    options = {**copy.deepcopy(original), **options}
    sources = [expected, *(row['method_metadata'] for row in rows)]
    if provenance is not None:
        _verify_metadata(expected, provenance, method)
        sources.append(provenance)
    configured = [item['configured_options'] for item in sources if item.get('configured_options') is not None]
    for settings in configured:
        _options_agree(options, settings, str(root))
    for row in rows:
        _verify_metadata(expected, row['method_metadata'], method)
    if rows:
        identity = model_identity(rows[0]['method_metadata'])
        if any(model_identity(row['method_metadata']) != identity for row in rows[1:]):
            raise ValueError(f'{root}: completed units disagree on model provenance')

    files = copy.deepcopy(model.get('model_files_sha256', {}))
    if not isinstance(files, dict):
        raise ValueError(f'{root}: invalid recorded model file checksums')
    for metadata in sources:
        for path, checksum in _file_hashes(metadata).items():
            if path in files and files[path] != checksum:
                raise ValueError(f'{root}: conflicting recorded model checksum: {path}')
            files[path] = checksum
    for path, checksum in files.items():
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file() or sha256(candidate) != checksum:
            raise ValueError(f'{root}: recorded model file missing or changed: {candidate}')

    # Reuse the established validation path, including MSDZip's checkpoint
    # contract. Supply recovered metadata as a source row without editing any
    # historical source run. The explicit device override is applied last.
    copied = copy.deepcopy(manifest)
    copied['source_config'].setdefault('method_options', {})[method] = copy.deepcopy(options)
    copied['source_config']['method_options'][method].pop('device', None)
    copied['source_records'] = [{'method': method, 'status': 'completed', 'variant': variant,
                                 'method_metadata': expected}]
    resolved = _method_contract(method, copied, Path(manifest['source_run']).expanduser().resolve(),
                                device if device is not None else options.get('device'))
    resolved['model_files_sha256'] = files
    resolved['provenance_file'] = str(run_path)
    resolved['provenance_sha256'] = sha256(run_path)
    resolved['provenance_origin'] = {
        'kind': 'declared_reuse_run', 'run': str(root), 'run_json_sha256': sha256(run_path),
        'source_run': contract['source_run'], 'prepared_manifest_sha256': contract['manifest_sha256'],
        'contract_model_present': method in contract.get('models', {}),
        'completed_unit_metadata': evidence,
        'method_provenance_file': str(provenance_path) if provenance is not None else None,
        'method_provenance_sha256': sha256(provenance_path) if provenance is not None else None,
    }
    return resolved


def resolve_change_contract(method, manifest, device=None, *, reuse_runs=(), prepared=None):
    """Resolve the source contract or safely recover it from declared wide runs.

    A bad existing source contract is never bypassed. Recovery only fills absent
    source provenance; every candidate remains bound to the same source and
    prepared manifest. Different valid frozen models are an ambiguity, not an
    instruction to choose the first. No model initialization occurs here.
    """
    try:
        return _method_contract(method, manifest, Path(manifest['source_run']).expanduser().resolve(), device)
    except (ValueError, OSError, KeyError) as error:
        source_error = error
    if _has_source_provenance(method, manifest):
        raise source_error
    config = manifest.get('source_config')
    if not isinstance(config, dict):
        raise source_error
    options = config.get('method_options', {}).get(method, {})
    if not isinstance(options, dict) or any(options.get(key) for key in _FORBIDDEN):
        raise source_error
    if isinstance(reuse_runs, (str, Path)):
        reuse_runs = [reuse_runs]
    candidates, rejected = [], []
    for root in dict.fromkeys(str(Path(value).expanduser().resolve()) for value in reuse_runs):
        try:
            candidates.append(_candidate(method, manifest, device, root, prepared))
        except (ValueError, OSError, KeyError) as error:
            rejected.append({'run': root, 'error': str(error)})
    if not candidates:
        details = '; '.join(f'{item["run"]}: {item["error"]}' for item in rejected)
        raise ValueError(f'{source_error}; no compatible declared reuse-run provenance'
                         + (f': {details}' if details else '')) from source_error
    selected = candidates[0]
    for candidate in candidates[1:]:
        if (candidate['options'] != selected['options']
                or model_identity(candidate['expected_metadata']) != model_identity(selected['expected_metadata'])
                or candidate['model_files_sha256'] != selected['model_files_sha256']):
            raise ValueError(f'{method}: declared reuse runs contain different frozen model contracts')
    selected['provenance_origin']['rejected_runs'] = rejected
    selected['provenance_origin']['source_error'] = str(source_error)
    return selected
