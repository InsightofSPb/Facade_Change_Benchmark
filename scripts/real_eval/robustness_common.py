"""Small CPU-only artifact contract for reviewed real-image robustness runs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_METHODS = ('rgb_diff', 'ssim', 'lpips', 'dinov2', 'semantic_dino',
                   'anychange', 'geoscd', 'msdzip_abs')
CONDITIONS = ('exposure', 'shadow')
NATIVE = frozenset(('anychange', 'geoscd'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return sha256(path)


def safe_path(root, relative):
    root = Path(root).expanduser().resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Artifact path must be relative')
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError('Artifact path escapes its run')
    return result


def load_prepared(path):
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != 'manifest.json':
            raise ValueError('Expected prepared directory or manifest.json')
        root = root.parent
    manifest = read_json(root / 'manifest.json')
    if (manifest.get('schema_version') != SCHEMA_VERSION
            or manifest.get('kind') != 'real_robustness_prepared'
            or manifest.get('conditions') != list(CONDITIONS)):
        raise ValueError('Unsupported robustness preparation contract')
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('Prepared manifest contains no pairs')
    ids = [row.get('pair_id') for row in pairs]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Prepared pair IDs must be distinct nonempty strings')
    return root, manifest


def load_pair(prepared_root, row):
    path = safe_path(prepared_root, row['artifact_path'])
    if sha256(path) != row['artifact_sha256']:
        raise ValueError(f'Prepared artifact changed: {row["pair_id"]}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {'reference', 'source', 'support', 'labels', 'reference_bits', 'source_bits', *CONDITIONS}
    if not required <= arrays.keys():
        raise ValueError('Prepared pair is missing required arrays')
    support = arrays['support']
    if support.dtype != bool or support.ndim != 2 or min(support.shape) < 1:
        raise ValueError('Invalid geometric support')
    for name in ('reference', 'source', *CONDITIONS):
        if arrays[name].dtype != np.uint8 or arrays[name].shape != (*support.shape, 3):
            raise ValueError(f'Invalid prepared uint8 RGB: {name}')
    labels = arrays['labels']
    if (labels.dtype != np.uint8 or labels.shape != support.shape
            or not np.isin(labels, [0, 1, 255]).all() or np.any(labels[~support] != 255)):
        raise ValueError('Invalid prepared change labels')
    for name in ('reference_bits', 'source_bits'):
        if arrays[name].dtype != np.uint16 or arrays[name].shape != support.shape:
            raise ValueError(f'Invalid prepared category bitsets: {name}')
    return arrays


def load_selection(path, prepared_root, manifest):
    selection = read_json(path)
    if (selection.get('schema_version') != SCHEMA_VERSION or selection.get('reviewed') is not True
            or selection.get('manifest_sha256') != sha256(Path(prepared_root) / 'manifest.json')):
        raise ValueError('Selection must be reviewed and bound to this exact manifest')
    ids = selection.get('pair_ids')
    available = {row['pair_id']: row for row in manifest['pairs']}
    if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids)
            or len(ids) != len(set(ids)) or any(x not in available for x in ids)
            or any(available[x].get('status', 'ready') != 'ready' for x in ids)):
        raise ValueError('Select distinct ready pairs from the gallery')
    return selection, [available[x] for x in ids]
