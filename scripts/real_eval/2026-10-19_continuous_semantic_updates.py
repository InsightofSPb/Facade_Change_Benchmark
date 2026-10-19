"""CPU-only event fusion from saved DINO probabilities, LPIPS and MSDZip maps."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import importlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

pilot = importlib.import_module('scripts.real_eval.2026-10-19_semantic_updates')
calibration = pilot.real_gt_calibration
metrics = pilot.real_gt_metrics
atomic_json, atomic_npz = pilot.atomic_json, pilot.atomic_npz

SCHEMA = '2026-10-19-continuous-events-v1'
FEATURES = ('p_from', 'p_to', 'lpips', 'msdzip_abs')
FEATURE_COUNTS = {'semantic': 2, 'semantic_lpips': 3, 'semantic_lpips_msdzip': 4}
DELTA_FEATURES = ('p_to_minus_p_from', 'lpips', 'msdzip_abs')
DELTA_FEATURE_COUNTS = {'semantic': 1, 'semantic_lpips': 2, 'semantic_lpips_msdzip': 3}
FEATURE_DESIGNS = ('raw_pair', 'directed_delta')
DELTA_SCHEMA = '2026-10-19-directed-delta-events-v1'
VARIANTS = ('semantic_delta', 'lpips_only', *FEATURE_COUNTS)
L2 = .01
SEED = 20261019
MAX_EVENTS_PER_PAIR = 8192
FPR_GOAL = .05
NOTES = (
    'TRAIN subset counts describe cached temporal sequences, not the full DINO training dataset.',
    'train_fit: fusion fitted and threshold calibrated on these same TRAIN facades; diagnostic only.',
    'train_fusion_oof: each facade excluded from fusion fitting, scaling and threshold calibration. '
    'Frozen DINO/MSDZip may have seen it; this is NOT end-to-end held-out evaluation.',
    'VAL and previously examined TEST use one model fitted only on all temporal TRAIN facades. '
    'TEST remains exploratory; no search over recipes or regularization.',
    'GT is a change in annotation membership, not independent physical-event annotation. '
    'Pixels without annotations at either date are unknown.',
    'Event scores are not calibrated physical-event probabilities. A predicted loss does not imply repair.',
    'This predicts gain/loss events; it does not claim exact C,V reconstruction of thresholded semantic maps.',
    'One final threshold per variant is calibrated to 5% binary FPR on fitting facades. '
    'Actual held-out FPR is measured, not assumed.',
    'F1: all_frames, mean pairs within facade then equal facades. AP: positive pairs only, '
    'then equal facades; undefined AP remains N/A. Directed-event metrics are micro over '
    'class/direction within each pair, not class-macro.',
)
DELTA_NOTES = (
    'Semantic features are directed p_to - p_from; no independent state-confidence feature.',
    'Each fitted semantic slope must be strictly positive; an unsupported/nonpositive fit blocks the run.',
    'The semantic-only model must match raw semantic_delta masks and AP in every evaluated pair. '
    'Learned directed-delta scores retain float64 to avoid sigmoid float32 quantization.',
    'This revision follows inspection of earlier VAL/TEST results and remains exploratory.',
)


def feature_spec(design):
    if design not in FEATURE_DESIGNS:
        raise ValueError(f'Unknown feature design: {design}')
    return (DELTA_FEATURES, DELTA_FEATURE_COUNTS) if design == 'directed_delta' else (FEATURES, FEATURE_COUNTS)


def fit_event_model(x, y, weight, features, design):
    model = dict(fit_logistic(x, y, weight), features=list(features), feature_design=design)
    if design == 'directed_delta' and model['coefficients'][0] <= 0:
        raise ValueError('Directed-delta fit has no positive semantic slope; no sign flip or fallback applied')
    return model


def event_truth(pair):
    count = len(pair['reference']['probabilities'])
    before = pilot.unpack(pair['reference']['packed'], count)
    after = pilot.unpack(pair['source']['packed'], count)
    return np.stack((~before & after, before & ~after))


def training_sample(pairs, max_events=MAX_EVENTS_PER_PAIR):
    """Stratify events, preserving prevalence and equal facade/pair loss weights."""
    if not pairs or any(p['context']['split'] != 'train' for p in pairs):
        raise ValueError('Fusion fitting accepts TRAIN pairs only')
    if max_events < 2:
        raise ValueError('Need at least two sampled events per pair')
    counts = Counter(p['context']['building_id'] for p in pairs)
    xs, ys, ws, audit = [], [], [], []
    for pair in pairs:
        known = pair['labels'] != 255
        old = pair['reference']['probabilities'][:, known]
        new = pair['source']['probabilities'][:, known]
        truth = event_truth(pair)[:, :, known].ravel()
        n, classes = int(known.sum()), len(old)
        if not n:
            raise ValueError('No known pixels in a training pair')
        key = (pair['context']['sequence_id'], pair['context']['frame'])
        rng = np.random.default_rng(SEED ^ int(pilot.digest(key)[:16], 16))
        lpips = pair['maps']['lpips']['scores'][known]
        msdzip = pair['maps']['msdzip_abs']['scores'][known]
        pair_weight = 1 / (len(counts) * counts[pair['context']['building_id']])
        sampled = []
        for label in (False, True):
            indices = np.flatnonzero(truth == label)
            population = len(indices)
            if not population:
                continue
            if population > max_events // 2:
                indices = np.sort(rng.choice(indices, max_events // 2, replace=False))
            direction, channel, pixel = indices // (classes * n), (indices // n) % classes, indices % n
            first, second = old[channel, pixel], new[channel, pixel]
            xs.append(np.column_stack((np.where(direction == 0, first, second),
                                       np.where(direction == 0, second, first), lpips[pixel], msdzip[pixel])))
            ys.append(np.full(len(indices), label, np.float64))
            ws.append(np.full(len(indices), pair_weight * population / (len(truth) * len(indices))))
            sampled.append(dict(label=int(label), population=population, sampled=len(indices)))
        audit.append(dict(pair['context'], pair_weight=pair_weight, strata=sampled))
    x, y, weight = np.concatenate(xs).astype(np.float64), np.concatenate(ys), np.concatenate(ws)
    if not np.isfinite(x).all() or not 0 < np.dot(weight, y) < weight.sum():
        raise ValueError('Need finite features and both event/non-event TRAIN examples')
    return x, y, weight / weight.sum(), audit


def sigmoid(value):
    return np.exp(-np.logaddexp(0., -value))


def fit_logistic(x, y, weight, l2=L2):
    """Weighted Newton fit; fixed L2 on standardized slopes, unpenalized intercept."""
    x, y, weight = np.asarray(x, float), np.asarray(y, float), np.asarray(weight, float)
    if (x.ndim != 2 or y.shape != (len(x),) or weight.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(weight).all()
            or np.any(weight <= 0) or not np.isin(y, [0, 1]).all() or l2 <= 0):
        raise ValueError('Invalid logistic fitting arrays/regularization')
    weight = weight / weight.sum()
    prevalence = float(weight @ y)
    if not 0 < prevalence < 1:
        raise ValueError('Both labels are required for logistic fitting')
    mean = weight @ x
    scale = np.sqrt(weight @ ((x - mean) ** 2))
    scale[scale < 1e-8] = 1.
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(prevalence / (1 - prevalence))
    penalty = np.diag(np.r_[0., np.full(x.shape[1], l2)])

    def loss(parameters):
        logits = design @ parameters
        return float(weight @ (np.logaddexp(0., logits) - y * logits) + .5 * parameters @ penalty @ parameters)

    for iteration in range(60):
        probability = sigmoid(design @ beta)
        gradient = design.T @ (weight * (probability - y)) + penalty @ beta
        if np.max(np.abs(gradient)) < 1e-7:
            break
        curvature = weight * probability * (1 - probability)
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        value, fraction = loss(beta), 1.
        for _ in range(25):
            candidate = beta - fraction * step
            if loss(candidate) <= value - 1e-4 * fraction * float(gradient @ step):
                beta = candidate
                break
            fraction *= .5
        else:
            raise ValueError('Logistic line search failed')
    else:
        raise ValueError('Logistic fit did not converge; no scores published')
    coefficients = beta[1:] / scale
    return dict(coefficients=coefficients.tolist(), intercept=float(beta[0] - coefficients @ mean),
                standardized_coefficients=beta[1:].tolist(), feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                prevalence=prevalence, loss=loss(beta), iterations=iteration, l2=l2, converged=True,
                gradient_max=float(np.max(np.abs(gradient))))


def score_pair(pair, name, model=None):
    """Inference uses probabilities/maps/support only, never GT or known-label masks."""
    support = pair['support']
    if name == 'lpips_only':
        return None, np.where(support, pair['maps']['lpips']['scores'], 0).astype(np.float32)
    old = pair['reference']['probabilities']
    new = pair['source']['probabilities']
    if name == 'semantic_delta':
        delta = new - old
        events = np.stack((np.maximum(delta, 0), np.maximum(-delta, 0)))
    elif model.get('feature_design') == 'directed_delta':
        weights = model['coefficients']
        if weights[0] <= 0:
            raise ValueError('Directed-delta inference requires a positive semantic slope')
        # Subtract in the original float32 coordinate, exactly as semantic_delta does.
        delta = (new - old).astype(np.float64)
        shared = np.full(support.shape, model['intercept'], np.float64)
        for i, method in enumerate(pilot.METHODS, 1):
            if i < len(weights):
                shared += weights[i] * np.where(support, pair['maps'][method]['scores'], 0).astype(np.float64)
        events = np.stack((np.where(delta > 0, sigmoid(shared + weights[0] * delta), 0),
                           np.where(delta < 0, sigmoid(shared - weights[0] * delta), 0)))
    else:
        weights = model['coefficients']
        shared = np.full(support.shape, model['intercept'], np.float64)
        for i, method in enumerate(pilot.METHODS, 2):
            if i < len(weights):
                shared += weights[i] * np.where(support, pair['maps'][method]['scores'], 0)
        gain = shared + weights[0] * old + weights[1] * new
        loss = shared + weights[0] * new + weights[1] * old
        # Resolve direction before thresholding; exact ties abstain. Classes may overlap.
        events = np.stack((np.where(gain > loss, sigmoid(gain), 0),
                           np.where(loss > gain, sigmoid(loss), 0)))
    dtype = np.float64 if model and model.get('feature_design') == 'directed_delta' else np.float32
    events = np.where(support[None, None], events, 0).astype(dtype)
    if not np.isfinite(events).all():
        raise ValueError('Nonfinite event scores')
    return events, events.max(axis=(0, 1))


def fit_bundle(pairs, excluded=None, feature_design='raw_pair'):
    features, feature_counts = feature_spec(feature_design)
    training = [p for p in pairs if p['context']['split'] == 'train'
                and p['context']['building_id'] != excluded]
    x, y, weight, sample_audit = training_sample(training)
    if feature_design == 'directed_delta':
        delta = x[:, 1].astype(np.float32) - x[:, 0].astype(np.float32)
        x = np.column_stack((delta, x[:, 2:]))
    bundle = dict(excluded_building=excluded, feature_design=feature_design,
                  training_buildings=sorted({p['context']['building_id'] for p in training}),
                  samples=len(y), sample_audit=sample_audit, models={}, thresholds={})
    for name in VARIANTS:
        print(f'Fit/calibrate: {name}; excluded={excluded or "none"}', flush=True)
        model = None
        if name in feature_counts:
            size = feature_counts[name]
            model = fit_event_model(x[:, :size], y, weight, features[:size], feature_design)
            bundle['models'][name] = model
        rows = []
        for pair in training:
            _, scores = score_pair(pair, name, model)
            known = pair['labels'] != 255
            rows.append(dict(pair['context'], scores=scores[known], labels=pair['labels'][known],
                             support_sha256=pair['support_sha256']))
        choice = calibration.calibrate_threshold(rows, None, FPR_GOAL)
        if choice['status'] != 'calibrated':
            raise ValueError('No negative TRAIN pixels for final-score calibration')
        bundle['thresholds'][name] = dict(choice, fitting_excluded_building=excluded,
                                          calibration_split='train', target='binary_change')
    return bundle


def check_parity(reference, fitted):
    """Check predictions over full support and AP over known labels, without refitting."""
    before_rows, before_mask, before_events = reference
    after_rows, after_mask, after_events = fitted
    result = dict(binary_disagreements=int(np.count_nonzero(before_mask != after_mask)),
                  event_disagreements=int(np.count_nonzero(before_events != after_events)))
    for target, before, after in zip(('binary', 'directed'), before_rows[:2], after_rows[:2]):
        a, b = before['average_precision'], after['average_precision']
        result[target + '_ap_difference'] = None if a is None or b is None else float(b - a)
        result[target + '_ap_matches'] = (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-12)
    result['passed'] = (result['binary_disagreements'] == result['event_disagreements'] == 0
                        and result['binary_ap_matches'] and result['directed_ap_matches'])
    return result


def evaluate(pair, event_scores, scores, threshold, categories):
    labels = pair['labels']
    prediction = (scores > threshold) & pair['support']
    known = labels != 255
    binary = dict(target='binary_change', class_name='__all__', direction='change',
                  **metrics.evaluate_mask(prediction, labels),
                  average_precision=calibration.average_precision(scores[known], labels[known]))
    rows = [binary]
    predicted_events = None
    if event_scores is not None:
        predicted_events = (event_scores > threshold) & pair['support'][None, None]
        truth = event_truth(pair)
        event_labels = np.where(known[None, None], truth, 255).astype(np.uint8)
        if np.any(predicted_events[0] & predicted_events[1]):
            raise ValueError('Conflicting gain/loss predictions for one class')
        if not np.array_equal(predicted_events.any(axis=(0, 1)), prediction):
            raise ValueError('Binary and event predictions disagree')
        rows.append(dict(target='directed_events', class_name='__all__', direction='gain_loss',
                         **metrics.evaluate_mask(predicted_events, event_labels),
                         average_precision=calibration.average_precision(
                             event_scores[:, :, known].ravel(), event_labels[:, :, known].ravel())))
        for d, direction in enumerate(('gain', 'loss')):
            for c, category in enumerate(categories):
                rows.append(dict(target='directed_class', class_name=category['name'], direction=direction,
                                 **metrics.evaluate_mask(predicted_events[d, c], event_labels[d, c])))
    return rows, prediction, predicted_events


def aggregate(rows):
    summaries, facades = pilot.aggregate(rows)
    keys = ('split', 'variant', 'target', 'class_name', 'direction')
    grouped = pilot.groups_from(rows, (*keys, 'building_id'))
    for row in facades:
        members = grouped[tuple(row[k] for k in (*keys, 'building_id'))]
        values = [r['average_precision'] for r in members if r.get('average_precision') is not None]
        row.update(average_precision=float(np.mean(values)) if values else None, ap_pairs=len(values))
    grouped = pilot.groups_from(facades, (*keys, 'protocol'))
    for row in summaries:
        members = grouped[tuple(row[k] for k in (*keys, 'protocol'))]
        values = [r['average_precision'] for r in members if r['average_precision'] is not None]
        row.update(average_precision=float(np.mean(values)) if values else None,
                   ap_facades=len(values), ap_pairs=sum(r['ap_pairs'] for r in members))
    return summaries, facades


def audit_inputs(source, out):
    pairs, coverage, splits = [], [], {}
    for sequence in source.manifest['sequences']:
        building, split = sequence['building_id'], sequence['split']
        if splits.setdefault(building, split) != split:
            raise ValueError('A facade occurs in more than one split')
        print(f'Cache audit: {split} {building}', flush=True)
        for frame in range(1, len(sequence['frames'])):
            header = dict(split=split, building_id=building, sequence_id=sequence['sequence_id'], frame=frame)
            try:
                pair = source.pair(sequence, frame)
                if not pair['valid_pixels']:
                    raise ValueError('Pair has no common known pixels')
                pairs.append(pair)
                coverage.append(dict(header, status='ready', **{k: pair[k] for k in
                    ('geometry_pixels', 'common_support_pixels', 'geometry_known_pixels',
                     'dropped_known_pixels', 'valid_pixels', 'support_sha256')}))
            except (OSError, ValueError, KeyError, TypeError) as error:
                coverage.append(dict(header, status='blocked', error=f'{type(error).__name__}: {error}'))
    pilot.semantic_report._write_csv(out / 'coverage.csv', coverage)
    if not pairs or any(r['status'] != 'ready' for r in coverage):
        raise ValueError('Incomplete/invalid cache; no partial-cohort evaluation')
    if {p['context']['split'] for p in pairs} != {'train', 'val', 'test'}:
        raise ValueError('Expected separate TRAIN/VAL/TEST temporal subsets')
    if len({p['context']['building_id'] for p in pairs if p['context']['split'] == 'train'}) < 2:
        raise ValueError('At least two TRAIN facades needed for fusion-only facade holdout')
    return pairs


def report(out, summaries, articles, notes=NOTES):
    columns = ('split', 'variant', 'target', 'average_precision', 'f1', 'precision', 'recall', 'fpr', 'facades', 'pairs')
    primary = [r for r in summaries if r['class_name'] == '__all__' and r['protocol'] == 'all_frames']
    def cell(value):
        return 'N/A' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    table = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for _ in columns) + ' |\n'
    table += '\n'.join('| ' + ' | '.join(cell(r.get(c)) for c in columns) + ' |' for r in primary)
    (out / '2026-10-19_tables.md').write_text(table + '\n\n' + '\n'.join('- ' + x for x in notes), encoding='utf-8')
    header = '<tr>' + ''.join('<th>' + c + '</th>' for c in columns) + '</tr>'
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(cell(r.get(c))) + '</td>' for c in columns) + '</tr>' for r in primary)
    content = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Continuous semantic events</title>'
               '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}'
               'td,th{padding:6px;border:1px solid #ccc}.grid{display:flex;flex-wrap:wrap}'
               'figure{width:300px;margin:8px}img{max-width:100%}figcaption{font-size:13px}</style>'
               '<h1>Continuous semantic event experiment</h1><ul>'
               + ''.join('<li>' + html.escape(n) + '</li>' for n in notes) + '</ul><table>' + header + body
               + '</table><h2>All TRAIN fusion-OOF / VAL / TEST pairs</h2>'
               '<p>S panels show the original 0.5-threshold states for context; they do not gate event predictions. '
               'Binary overlays: green TP, red FP, blue FN, evaluated region only. These overlays do not verify event class/direction.</p>'
               + ''.join(articles) + '</html>')
    (out / '2026-10-19_report.html').write_text(content, encoding='utf-8')


def run(semantic_run, change_run, out_path, audit_only=False, feature_design='raw_pair'):
    features, feature_counts = feature_spec(feature_design)
    schema = DELTA_SCHEMA if feature_design == 'directed_delta' else SCHEMA
    notes = NOTES + DELTA_NOTES if feature_design == 'directed_delta' else NOTES
    out = Path(out_path).expanduser().resolve()
    protected = [Path(p).expanduser().resolve() for p in (semantic_run, change_run)]
    initial = pilot.read_json(protected[0] / 'run.json')
    protected += [Path(initial['config']['prepared']).expanduser().resolve(),
                  Path(initial['cache_root']).expanduser().resolve()]
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError('Output must be separate from all input/cache directories')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(kind='continuous_semantic_events', schema=schema, feature_design=feature_design, status='running',
                 started_utc=datetime.now(timezone.utc).isoformat(), semantic_run=str(protected[0]),
                 change_run=str(protected[1]), audit_only=audit_only, limitations=list(notes))
    atomic_json(out / 'run.json', state)
    source = None
    try:
        source = pilot.Inputs(semantic_run, change_run, retain_probabilities=True)
        pairs = audit_inputs(source, out)
        buildings = sorted({p['context']['building_id'] for p in pairs if p['context']['split'] == 'train'})
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, check=False)
        config = dict(schema=schema, feature_design=feature_design, variants=list(VARIANTS), feature_counts=feature_counts,
                      software=dict(python=platform.python_version(), numpy=np.__version__),
                      features=list(features), l2=L2, seed=SEED, max_events_per_pair=MAX_EVENTS_PER_PAIR,
                      positive_semantic_slope_required=feature_design == 'directed_delta',
                      target_binary_fpr=FPR_GOAL, mode='previous', threshold_operator='>',
                      event_directions=['gain', 'loss'], direction_policy='higher logit wins; equal logits abstain',
                      sampling='event-stratified; population-corrected weights; equal facades then pairs',
                      categories=source.categories, category_to_channel=source.manifest['category_to_channel'],
                      temporal_subset={s: dict(facades=sorted({p['context']['building_id'] for p in pairs if p['context']['split'] == s}),
                                              pairs=sum(p['context']['split'] == s for p in pairs)) for s in ('train', 'val', 'test')},
                      semantic_model=source.model, method_contracts=source.contract['models'],
                      saved_training_exposure=source.contract.get('training_exposure', {}),
                      git_commit=git.stdout.strip() if git.returncode == 0 else None,
                      code_sha256={str(Path(p).resolve().relative_to(Path(__file__).resolve().parents[2])): pilot.sha256(p)
                                   for p in (Path(__file__), Path(pilot.__file__), Path(calibration.__file__),
                                             Path(metrics.__file__), Path(pilot.semantic_geometry.__file__),
                                             Path(pilot.night_common.__file__), Path(pilot.night_report.__file__),
                                             Path(pilot.semantic_report.__file__))})
        atomic_json(out / 'resolved_config.json', config)
        if audit_only:
            state.update(status='audit_passed', pairs=len(pairs))
        else:
            jobs = [(None, pairs)] + [(b, [p for p in pairs if p['context']['split'] == 'train'
                                           and p['context']['building_id'] == b]) for b in buildings]
            rows, articles, artifacts, model_index, parity_rows = [], [], [], [], []
            for excluded, evaluation in jobs:
                fold = 'all_train' if excluded is None else 'holdout-' + pilot.digest(excluded)[:12]
                bundle = fit_bundle(pairs, excluded, feature_design)
                relative_model = f'models/2026-10-19-{fold}.json'
                (out / 'models').mkdir(exist_ok=True)
                atomic_json(out / relative_model, bundle)
                model_index.append(dict(fold=fold, path=relative_model, sha256=pilot.sha256(out / relative_model)))
                for i, pair in enumerate(evaluation, 1):
                    original = pair['context']
                    split = ('train_fusion_oof' if excluded else 'train_fit') if original['split'] == 'train' else original['split']
                    context = dict(original, split=split, source_split=original['split'], fold=fold,
                                   feature_design=feature_design)
                    print(f'Evaluate: {split} {i}/{len(evaluation)} {original["building_id"]}', flush=True)
                    arrays = dict(labels=pair['labels'], support=pair['support'])
                    panels = {}
                    parity_reference = None
                    for name in VARIANTS:
                        events, scores = score_pair(pair, name, bundle['models'].get(name))
                        measured, mask, predicted = evaluate(pair, events, scores, bundle['thresholds'][name]['threshold'], source.categories)
                        if feature_design == 'directed_delta':
                            if name == 'semantic_delta':
                                parity_reference = measured, mask, predicted
                            elif name == 'semantic':
                                check = check_parity(parity_reference, (measured, mask, predicted))
                                parity_rows.append(dict(context, **check))
                                if not check['passed']:
                                    pilot.semantic_report._write_csv(out / '2026-10-19_parity.csv', parity_rows)
                                    raise ValueError('Semantic-only delta parity failed; inspect 2026-10-19_parity.csv')
                        rows.extend(dict(context, variant=name, **r) for r in measured)
                        panels[name] = dict(C=mask)
                        arrays[name + '__change_score'] = scores
                        arrays[name + '__prediction'] = mask
                        if predicted is not None:
                            arrays[name + '__gain'] = pilot.pack(predicted[0])
                            arrays[name + '__loss'] = pilot.pack(predicted[1])
                    if split != 'train_fit':
                        path = 'maps/2026-10-19-' + pilot.digest(context)[:24] + '.npz'
                        checksum = atomic_npz(out / path, arrays)
                        artifacts.append(dict(context, artifact_path=path, artifact_sha256=checksum, model_path=relative_model))
                        articles.append(pilot.gallery(out, dict(pair, context=context), panels, source.names))
            summaries, facades = aggregate(rows)
            for name, values in (('per_pair', rows), ('per_facade', facades), ('summary', summaries)):
                pilot.semantic_report._write_csv(out / (name + '.csv'), values)
            atomic_json(out / '2026-10-19_artifacts.json', dict(models=model_index, maps=artifacts))
            if parity_rows:
                pilot.semantic_report._write_csv(out / '2026-10-19_parity.csv', parity_rows)
                state.update(parity_pairs=len(parity_rows), parity_passed=all(r['passed'] for r in parity_rows))
            report(out, summaries, articles, notes)
            state.update(status='completed', pairs=len(pairs), temporal_train_facades=len(buildings), result_rows=len(rows))
    except KeyboardInterrupt:
        state.update(status='interrupted', error='KeyboardInterrupt')
    except Exception as error:
        state.update(status='blocked', error=f'{type(error).__name__}: {error}')
    finally:
        if source is not None:
            atomic_json(out / '2026-10-19_input_hashes.json', source.evidence)
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(out / 'run.json', state)
    print(json.dumps({k: state[k] for k in ('status', 'pairs', 'error') if k in state}, ensure_ascii=False), flush=True)
    print(f'Results: {out}', flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-run', required=True)
    parser.add_argument('--change-run', required=True)
    parser.add_argument('--out', required=True, help='New output directory, never overwritten')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--feature-design', choices=FEATURE_DESIGNS, default='raw_pair',
                        help='directed_delta replaces the two state probabilities with their signed difference')
    args = parser.parse_args(argv)
    state = run(args.semantic_run, args.change_run, args.out, args.audit_only, args.feature_design)
    return 0 if state['status'] in ('completed', 'audit_passed') else 130 if state['status'] == 'interrupted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
