"""Negative-only LOFO calibration on already matched night-run score maps.

Rows contain ``sequence_id``, ``frame``, ``building_id``, ``pixels``, ``labels``
(known GT only, uint8 0/1), and ``left`` plus optional ``right``. Components are
float32 scores or fixed boolean native masks. The caller intersects support
before constructing rows. No model inference or file I/O occurs here.

For two scalar components their weighted strict-left negative ECDFs put scores
on a common scale. AND thresholds their minimum, OR their maximum. The final
threshold is fitted separately for each operation; branch FPR budgets alone
do not constrain the FPR of the union. Positive pixels never select a threshold.
"""
from collections import defaultdict

import numpy as np


_TOL = 1e-12


def _key(row):
    return row['sequence_id'], row['frame']


def _kind(value):
    value = np.asarray(value)
    if value.dtype == bool:
        return 'native'
    if value.dtype == np.float32:
        return 'scalar'
    raise ValueError('Components must be boolean native masks or float32 scores')


def _validate_row(row, paired):
    labels, pixels = np.asarray(row['labels']), np.asarray(row['pixels'])
    if (labels.ndim != 1 or labels.dtype != np.uint8
            or not np.isin(labels, (0, 1)).all() or pixels.shape != labels.shape
            or not np.issubdtype(pixels.dtype, np.integer) or np.any(pixels < 0)
            or (not np.all(pixels[1:] > pixels[:-1]) and len(np.unique(pixels)) != len(pixels))):
        raise ValueError('Expected distinct pixels and one-dimensional known uint8 GT labels 0/1')
    if ('right' in row) != paired:
        raise ValueError('Single and paired rows cannot be mixed')
    kinds = []
    for name in ('left', 'right') if paired else ('left',):
        value = np.asarray(row[name])
        kinds.append(_kind(value))
        if value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError('Component arrays must be finite and match the known-GT support')
    return tuple(kinds)


def _ecdf(values, weights):
    """Weighted P(S < s), with tied observations kept as one knot."""
    order = np.argsort(values, kind='stable')
    sorted_values, sorted_weights = values[order], weights[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    masses = np.add.reduceat(sorted_weights, starts)
    cumulative = np.cumsum(masses, dtype=np.float64)
    cumulative /= cumulative[-1]
    # Prefix length K+1 makes values above the largest knot map to exactly 1.
    return dict(knots=sorted_values[starts], prefix=np.r_[0., cumulative])


def _rank(values, ecdf):
    return ecdf['prefix'][np.searchsorted(ecdf['knots'], values, side='left')]


def _inverse_rank_gate(ecdf, threshold):
    """Encode r(s) > threshold as one raw-score gate, without storing knots."""
    if threshold < 0:
        return dict(mode='all', threshold=None)
    index = int(np.searchsorted(ecdf['prefix'], threshold, side='right'))
    if index == len(ecdf['prefix']):
        return dict(mode='empty', threshold=None)
    return dict(mode='threshold', threshold=float(ecdf['knots'][index - 1]))


def _tail_profile(values, weights, native=None):
    """All strict-threshold final FPRs, in increasing threshold order."""
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    total = float(weights.sum())
    tail = np.clip((total - np.cumsum(weights)[ends]) / total, 0., 1.)
    result = dict(values=values[ends], single=tail)
    if native is not None:
        native_weights = weights * native[order]
        native_total = float(native_weights.sum())
        native_fpr = float(np.clip(native_total / total, 0., 1.))
        both = np.clip((native_total - np.cumsum(native_weights)[ends]) / total,
                       0., native_fpr)
        result.update(native_fpr=native_fpr, and_=both,
                      or_=np.clip(native_fpr + tail - both, native_fpr, 1.))
    return result


def prepare(rows, heldout):
    """Fit reusable negative distributions, excluding heldout before array access.

    Call once per common-support family and heldout facade, then ``choose`` for
    several budgets/operations. Internal arrays are deliberately separate from
    the JSON-safe audit. A facade/frame with no known negatives gets no weight.
    """
    buildings = defaultdict(list)
    kinds = None
    paired = None
    seen = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            raise ValueError('Duplicate pair identity')
        seen.add(key)
        if row['building_id'] == heldout:
            continue  # Do not even read heldout scores, labels or support.
        if not isinstance(row['building_id'], str) or not row['building_id']:
            raise ValueError('A confirmed building_id is required')
        if paired is None:
            paired = 'right' in row
        row_kinds = _validate_row(row, paired)
        if kinds is None:
            kinds = row_kinds
        elif kinds != row_kinds:
            raise ValueError('Component kinds must be consistent across rows')
        if np.any(np.asarray(row['labels']) == 0):
            buildings[row['building_id']].append(row)
    # Stable ordering makes audit and floating-point sums invariant to row order.
    buildings = {building: sorted(members, key=_key)
                 for building, members in sorted(buildings.items())}
    training = [row for members in buildings.values() for row in members]
    audit = dict(
        heldout_building=heldout, train_buildings=list(buildings),
        train_pair_keys=[[str(row['sequence_id']), int(row['frame'])] for row in training],
        train_negative_pixels=sum(int(np.count_nonzero(np.asarray(row['labels']) == 0)) for row in training),
        train_support_sha256={f'{row["sequence_id"]}:{row["frame"]}': row.get('support_sha256')
                              for row in training},
        weighting='equal facade, then equal frames containing known GT negatives, then equal negative pixels',
        threshold_operator='>', calibration_split='val',
    )
    result = dict(audit=audit, kinds=kinds, profiles={}, rank_models=None)
    if not training:
        return result
    weights, components = [], [[] for _ in kinds]
    for members in buildings.values():
        for row in members:
            negative = np.asarray(row['labels']) == 0
            count = int(np.count_nonzero(negative))
            weights.append(np.full(count, 1 / (len(buildings) * len(members) * count), np.float64))
            for index, name in enumerate(('left', 'right')[:len(kinds)]):
                components[index].append(np.asarray(row[name])[negative])
    weights = np.concatenate(weights)
    components = [np.concatenate(values) for values in components]
    weights /= weights.sum()
    if kinds == ('scalar', 'scalar'):
        rank_models = [_ecdf(value, weights) for value in components]
        ranks = [_rank(value, model) for value, model in zip(components, rank_models)]
        result['rank_models'] = rank_models
        result['profiles'] = {
            operation: _tail_profile(value, weights)
            for operation, value in (('and', np.minimum(*ranks)), ('or', np.maximum(*ranks)))
        }
    elif all(kind == 'native' for kind in kinds):
        if len(kinds) == 1:
            predictions = dict(single=components[0])
        else:
            predictions = dict(and_=components[0] & components[1], or_=components[0] | components[1])
        result['fixed_fpr'] = {operation.rstrip('_'): float(np.clip(np.dot(value, weights), 0., 1.))
                               for operation, value in predictions.items()}
    elif len(kinds) == 1:
        result['profiles']['single'] = _tail_profile(components[0], weights)
    else:
        scalar_index = kinds.index('scalar')
        result['profiles']['mixed'] = _tail_profile(components[scalar_index], weights,
                                                   components[1 - scalar_index])
    return result


def choose(profile, budget, operation='single'):
    """Choose the least restrictive whole-tie threshold meeting final FPR.

    A fixed native prediction is returned unchanged even when outside budget.
    An infeasible native OR scalar model cannot create a prediction. No-negative
    training folds are unsupported, including fixed-mask folds.
    """
    if operation not in ('single', 'and', 'or') or not np.isfinite(budget) or not 0 <= budget < 1:
        raise ValueError('Expected single/and/or and finite final FPR budget in [0,1)')
    kinds = profile['kinds']
    if kinds is not None and ((len(kinds) == 1) != (operation == 'single')):
        raise ValueError('single requires one component; and/or require two')
    choice = dict(profile['audit'], goal=float(budget), operation=operation,
                  component_kinds=list(kinds) if kinds is not None else None,
                  threshold=None, gate_mode=None, achieved_train_fpr=None,
                  minimum_train_fpr=None, maximum_train_fpr=None, feasible=None,
                  native_train_fpr=None, rank_normalization=None)
    model = dict(choice=choice, kinds=kinds, rank_models=profile['rank_models'])
    if not profile['audit']['train_negative_pixels']:
        choice.update(status='unsupported', reason='no_other_facade_training_negatives')
        return model
    if 'fixed_fpr' in profile:
        actual = profile['fixed_fpr'][operation]
        feasible = actual <= budget + _TOL
        choice.update(status='fixed_reference_within_budget' if feasible else 'fixed_reference_outside_budget',
                      reason='native_binary_masks_unchanged', feasible=feasible,
                      achieved_train_fpr=actual, minimum_train_fpr=actual,
                      maximum_train_fpr=actual, gate_mode='fixed')
        return model
    minimum, maximum = 0., 1.
    if 'mixed' in profile['profiles']:
        curve = profile['profiles']['mixed']
        native = curve['native_fpr']
        choice['native_train_fpr'] = native
        minimum, maximum = (native, 1.) if operation == 'or' else (0., native)
        fprs = curve[operation + '_']
    else:
        curve = profile['profiles'][operation]
        fprs = curve['single']
        if kinds == ('scalar', 'scalar'):
            choice['rank_normalization'] = 'weighted_training_negative_strict_left_ecdf'
            choice['fusion_score'] = 'min(left_rank,right_rank)' if operation == 'and' else 'max(left_rank,right_rank)'
    choice.update(minimum_train_fpr=minimum, maximum_train_fpr=maximum,
                  empty_gate_train_fpr=minimum, all_gate_train_fpr=maximum)
    if minimum > budget + _TOL:
        choice.update(status='budget_unattainable', reason='native_OR_floor_exceeds_budget', feasible=False)
    elif maximum <= budget + _TOL:
        choice.update(status='calibrated', gate_mode='all', achieved_train_fpr=maximum, feasible=True)
    else:
        index = int(np.flatnonzero(fprs <= budget + _TOL)[0])
        choice.update(status='calibrated', gate_mode='threshold', threshold=float(curve['values'][index]),
                      achieved_train_fpr=float(fprs[index]), feasible=True)
    if choice['status'] == 'calibrated':
        if kinds == ('scalar', 'scalar'):
            choice['raw_gates'] = {
                name: _inverse_rank_gate(ecdf, choice['threshold'])
                for name, ecdf in zip(('left', 'right'), profile['rank_models'])
            }
        else:
            name = ('left', 'right')[kinds.index('scalar')]
            choice['raw_gates'] = {name: dict(mode=choice['gate_mode'], threshold=choice['threshold'])}
    return model


def fit(rows, heldout, budget, operation='single'):
    """Convenience wrapper; reuse ``prepare`` when fitting multiple budgets."""
    return choose(prepare(rows, heldout), budget, operation)


def predict(row, model):
    """Apply a frozen model; GT labels and pixel indices are not consulted.

    Exact inverse-ECDF gates avoid repeated rank searches during evaluation.
    The JSON-safe choice alone suffices to reproduce every prediction.
    """
    return predict_choice(row, model['choice'])


def predict_choice(row, choice):
    """Apply a serialized audit recipe, with no calibration rows or ECDFs."""
    if choice['status'] in ('unsupported', 'budget_unattainable'):
        raise ValueError('An unsupported/infeasible calibration cannot create a prediction')
    kinds = tuple(choice['component_kinds'])
    arrays = [np.asarray(row[name]) for name in ('left', 'right')[:len(kinds)]]
    if any(value.ndim != 1 or _kind(value) != kind or not np.isfinite(value).all()
           for value, kind in zip(arrays, kinds)) or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError('Prediction components do not match the fitted kinds/shape')
    operation = choice['operation']
    if choice['gate_mode'] == 'fixed':
        if operation == 'single':
            return arrays[0].copy()
        return arrays[0] & arrays[1] if operation == 'and' else arrays[0] | arrays[1]
    gates = []
    for name, scores, kind in zip(('left', 'right'), arrays, kinds):
        if kind == 'native':
            gates.append(scores)
            continue
        recipe = choice['raw_gates'][name]
        if recipe['mode'] == 'all':
            gate = np.ones(scores.shape, bool)
        elif recipe['mode'] == 'empty':
            gate = np.zeros(scores.shape, bool)
        elif recipe['mode'] == 'threshold':
            gate = scores > recipe['threshold']
        else:
            raise ValueError('Unknown gate mode')
        gates.append(gate)
    if operation == 'single':
        return gates[0]
    return gates[0] & gates[1] if operation == 'and' else gates[0] | gates[1]
