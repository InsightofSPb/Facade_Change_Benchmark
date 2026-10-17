"""Shared, CPU-only contracts for the selected real chronology experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .robustness_common import atomic_json, atomic_npz, digest, read_json, safe_path, sha256


DEFAULT_METHODS = ('rgb_diff', 'ssim', 'lpips', 'dinov2', 'semantic_dino',
                   'anychange', 'geoscd', 'msdzip_abs', 'msdzip_mod256')
HISTORY_METHODS = ('dinov2', 'semantic_dino', 'msdzip_abs', 'msdzip_mod256')
MODES = ('previous', 'earliest', 'history')


def unit_id(method, mode, sequence_id, frame):
    return digest([method, mode, sequence_id, int(frame)])[:32]


def load_prepared(path):
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != 'manifest.json':
            raise ValueError('Expected a prepared directory or manifest.json')
        root = root.parent
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'real_temporal_prepared':
        raise ValueError('Unsupported real temporal preparation')
    sequences = manifest.get('sequences')
    if not isinstance(sequences, list) or not sequences:
        raise ValueError('No prepared real histories')
    identifiers, buildings = set(), {}
    for sequence in sequences:
        identifier = sequence.get('sequence_id')
        building, split = sequence.get('building_id'), sequence.get('split')
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or not building or split not in ('train', 'val', 'test')):
            raise ValueError('Invalid or duplicate history identity')
        identifiers.add(identifier)
        if building in buildings and buildings[building] != split:
            raise ValueError('One physical facade appears in multiple splits')
        buildings[building] = split
        frames = sequence.get('frames', [])
        if len(frames) < 2 or [frame.get('frame') for frame in frames] != list(range(len(frames))):
            raise ValueError('History requires chronological, zero-based frame indices')
        years = [int(frame['year']) for frame in frames]
        if years != sorted(set(years)):
            raise ValueError('History years must increase strictly')
        for frame in frames:
            safe_path(root, frame['artifact_path'])
            if not isinstance(frame.get('artifact_sha256'), str) or len(frame['artifact_sha256']) != 64:
                raise ValueError('Missing prepared frame checksum')
    categories = manifest.get('categories')
    if not isinstance(categories, list) or not 1 <= len(categories) <= 16:
        raise ValueError('Packed annotations require 1–16 explicit categories')
    if 'category_to_channel' in manifest:
        from .robustness_prepare import _validate_category_mapping
        _validate_category_mapping(categories, manifest['category_to_channel'])
    return root, manifest


def load_frame(root, frame):
    path = safe_path(root, frame['artifact_path'])
    if sha256(path) != frame['artifact_sha256']:
        raise ValueError(f'Prepared frame changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('rgb', 'support', 'packed')}
    support = arrays['support']
    if support.dtype != bool or support.ndim != 2 or min(support.shape) < 1:
        raise ValueError('Invalid geometric support')
    if arrays['rgb'].dtype != np.uint8 or arrays['rgb'].shape != (*support.shape, 3):
        raise ValueError('Raw RGB must be uint8 on the prepared native grid')
    if arrays['packed'].dtype != np.uint16 or arrays['packed'].shape != support.shape:
        raise ValueError('Invalid packed annotation channels')
    return arrays


def read_model_contracts(manifest, methods, device=None):
    from .robustness_run import _method_contract
    return {method: _method_contract(method, manifest, Path(manifest['source_run']), device)
            for method in methods}


def _current_manifest(manifest):
    """Find the actual evaluation identity manifest, without decoding pixels."""
    dataset = manifest.get('dataset') or manifest.get('source_config', {}).get('dataset')
    if not dataset:
        return None
    path = Path(dataset).expanduser().resolve()
    if path.is_dir():
        path = path / 'dataset.json'
    if not path.is_file():
        return None
    data = read_json(path)
    value = data.get('manifest')
    if not value:
        return None
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()


def training_exposure(manifest, contracts):
    """Checkpoint training exposure, not a claim about web-scale pretraining.

    Lack of an observed overlap is not enough to label a facade unseen. Unknown
    identities in semantic training keep non-overlapping buildings unverified.
    """
    output = {}
    current = _current_manifest(manifest)
    current_hash = sha256(current) if current is not None and current.is_file() else None
    for method, contract in contracts.items():
        metadata = contract.get('expected_metadata', {})
        row = dict(status='unknown', building_ids=[], complete=False,
                   scope='project checkpoint training identities; external pretraining not audited')
        if method.startswith('msdzip_'):
            train = metadata.get('buildings', {}).get('train')
            split = metadata.get('reviewed_split', {})
            assignments = split.get('building_assignments', {})
            if (isinstance(train, list) and train and all(isinstance(x, str) and x for x in train)
                    and metadata.get('checkpoint_sha256')):
                row['building_ids'] = sorted(set(train))
                complete = (split.get('mode') == 'reviewed'
                            and all(assignments.get(x) == 'train' for x in train))
                row.update(status='verified' if complete else 'unknown', complete=complete)
        elif method == 'semantic_dino':
            origin = metadata.get('checkpoint_origin_audit', {})
            overlaps = origin.get('train_overlap_by_current_split', {})
            buildings = {str(item['building_id']) for values in overlaps.values()
                         for item in values if item.get('building_id')}
            row['building_ids'] = sorted(buildings)
            bound = origin.get('current_manifest') or {}
            same_manifest = bool(current_hash and bound.get('sha256') == current_hash)
            complete = bool(origin.get('training_manifest_verified') and same_manifest
                            and not origin.get('unverified_training_image_ids')
                            and origin.get('training_images', 0) > 0)
            row.update(status='verified' if complete else 'unknown', complete=complete)
            if not same_manifest:
                # The overlap list is only meaningful for the identity manifest
                # against which it was originally audited.
                row['building_ids'] = []
                row['reason'] = 'current identity manifest differs or cannot be verified'
        elif method in DEFAULT_METHODS:
            row.update(status='not_project_trained', complete=True)
        output[method] = row
    return output
