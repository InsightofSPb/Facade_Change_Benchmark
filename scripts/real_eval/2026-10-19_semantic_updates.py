"""Cache-only pilot: directed semantic updates, LPIPS and MSDZip confirmation.

Run with python -m scripts.real_eval.2026-10-19_semantic_updates --help.
No worker, checkpoint loader, training, alignment or model inference is called.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import night_common, night_report, real_gt_calibration, real_gt_metrics
from . import semantic_geometry, semantic_report
from .robustness_common import atomic_json, atomic_npz, digest, read_json, safe_path, sha256
from .robustness_run import _verify_metadata, payload_sha256
from .semantic_cache import hash_array


SCHEMA = '2026-10-19-directed-semantic-updates-v1'
VARIANTS = ('semantic', 'semantic_lpips', 'semantic_lpips_msdzip')
METHODS = ('lpips', 'msdzip_abs')
GATE_FPR = .05
LIMITATIONS = (
    'Разведочный опыт на ранее выбранных кропах. TEST уже использовался в анализе.',
    'GT — изменение состава аннотаций; это не независимая разметка физических событий.',
    'Нулевой битсет GT означает неизвестную область, а не здоровый фасад.',
    'Исчезновение CRACK не означает REPAIRS. Переходы многоклассовые, с пересечениями.',
    'Пороги LPIPS и MSDZip ABS отдельно калибруются на TRAIN при FPR 5%; '
    'это не обещание FPR 5% на VAL/TEST или у итоговой композиции.',
    'Правило AND только подтверждает предсказанные семантические переходы: '
    'оно не восстанавливает пропуски сегментатора.',
    'Проверка кэша подтверждает происхождение вычислений, но сама по себе '
    'не доказывает независимость VAL/TEST от обучения checkpoint.',
)


def pack(masks):
    if masks.dtype != bool or masks.ndim != 3 or not 1 <= len(masks) <= 16:
        raise ValueError('Expected 1–16 boolean semantic channels')
    bits = np.zeros(masks.shape[1:], np.uint16)
    for channel, mask in enumerate(masks):
        bits |= mask.astype(np.uint16) << channel
    return bits


def unpack(bits, count):
    return np.stack([(bits & (1 << i)) != 0 for i in range(count)])


def updates(before, after, lpips, msdzip, thresholds, support):
    """C and V reconstruct the accepted next state from the predicted old state."""
    changed = (before != after) & support
    gates = (support, lpips > thresholds['lpips'],
             (lpips > thresholds['lpips']) & (msdzip > thresholds['msdzip_abs']))
    result = {}
    for name, gate in zip(VARIANTS, gates):
        c = changed & gate
        v = np.where(c, after, 0).astype(np.uint16)
        reconstructed = np.where(c, v, before).astype(np.uint16)
        if not np.array_equal((before != reconstructed), c):
            raise ValueError('C,V reconstruction invariant failed')
        result[name] = dict(C=c, V=v, reconstructed=reconstructed)
    return result


def evaluate(before, update, old_gt, new_gt, support, categories):
    count = len(categories)
    labels = real_gt_metrics.derive_labels(old_gt, new_gt, support, count)
    known = labels != 255
    old, new = unpack(before, count), unpack(update['reconstructed'], count)
    gt_old, gt_new = unpack(old_gt, count), unpack(new_gt, count)
    predicted = np.stack((~old & new, old & ~new))
    truth = np.stack((~gt_old & gt_new, gt_old & ~gt_new))
    event_labels = np.where(known[None, None], truth, 255).astype(np.uint8)
    rows = [dict(target='binary_change', class_name='__all__', direction='change',
                 **real_gt_metrics.evaluate_mask(update['C'], labels)),
            dict(target='directed_events', class_name='__all__', direction='gain_loss',
                 **real_gt_metrics.evaluate_mask(predicted, event_labels))]
    for d, direction in enumerate(('gain', 'loss')):
        for channel, category in enumerate(categories):
            rows.append(dict(target='directed_class', class_name=category['name'], direction=direction,
                             **real_gt_metrics.evaluate_mask(predicted[d, channel], event_labels[d, channel])))
    # Conditional V accuracy is accompanied by an end-to-end recall that counts misses.
    hits = update['C'] & (labels == 1)
    exact = hits & (update['V'] == new_gt)
    rows[0].update(exact_v_tp=int(exact.sum()), detected_gt_change_pixels=int(hits.sum()),
                   exact_v_precision=(float(exact.sum() / hits.sum()) if hits.any() else None),
                   exact_v_recall=(float(exact.sum() / (labels == 1).sum()) if (labels == 1).any() else None))
    return rows


class Inputs:
    """Read-only readers; even corrupt source caches are never quarantined/edited."""

    def __init__(self, semantic_run, change_run):
        self.semantic_root = Path(semantic_run).expanduser().resolve()
        self.change_root = Path(change_run).expanduser().resolve()
        self.evidence = {}
        self.sem_run = self.json(self.semantic_root / 'run.json')
        if self.sem_run.get('kind') != 'semantic_routes' or self.sem_run.get('smoke'):
            raise ValueError('Need a full semantic_routes run, not smoke')
        self.config = self.sem_run['config']
        self.prepared, self.manifest = night_common.load_prepared(self.config['prepared'])
        manifest_hash = self.note(self.prepared / 'manifest.json')
        revision = digest(dict(config=self.config, prepared=manifest_hash, schema=self.sem_run['schema']))
        if revision != self.sem_run['config_revision']:
            raise ValueError('Semantic config/prepared manifest changed')
        self.categories = semantic_report._ontology(self.manifest['categories'],
                                                    self.manifest['category_to_channel'])
        self.names = [c['name'] for c in self.categories]
        self.cache_root = Path(self.sem_run['cache_root']).expanduser().resolve()
        self.preflight = self.json(self.semantic_root / 'preflight.json')
        self.model = self.preflight['segmenters']['dino_semantic']
        if not self.model.get('ok'):
            raise ValueError('Saved DINO semantic preflight was not successful')
        cfg = self.config['segmenters']['dino_semantic']
        if cfg.get('kind') != 'dino_semantic':
            raise ValueError('This fixed pilot requires the DINO semantic head')
        self.confidence = float(cfg.get('confidence', self.config.get('semantic', {}).get('confidence', .25)))
        if self.confidence != .5:
            raise ValueError('Pilot precommits to the existing DINO confidence 0.5')
        self.wide_run = self.json(self.change_root / 'run.json')
        self.contract = self.wide_run['contract']
        if self.contract['manifest_sha256'] != manifest_hash:
            raise ValueError('Wide run and semantic run use different prepared manifests')
        if ('config_hash' in self.wide_run and self.wide_run['config_hash'] != digest(self.contract)):
            raise ValueError('Wide run contract checksum mismatch')
        for method in METHODS:
            if not self.contract.get('models', {}).get(method, {}).get('expected_metadata'):
                raise ValueError(f'Missing saved model contract: {method}')
        self.semantic_units, self.change_units = {}, {}
        for path in sorted((self.semantic_root / 'units').glob('*.json')):
            row = read_json(path)
            if (row.get('config_revision') == revision and row.get('kind') == 'segmentation'
                    and row.get('segmenter') == 'dino_semantic' and row.get('variant') == 'fixed'
                    and row.get('route') == 'baseline'):
                self.note(path)
                self.insert(self.semantic_units, (row['sequence_id'], row['frame']), row)
        for path in sorted((self.change_root / 'units').glob('*.json')):
            row = read_json(path)
            if row.get('method') in METHODS and row.get('mode') == 'previous':
                self.note(path)
                self.insert(self.change_units, (row['method'], row['sequence_id'], row['frame']), row)
        self.frame_cache = {}

    @staticmethod
    def insert(index, key, row):
        if key in index:
            raise ValueError(f'Ambiguous duplicate unit: {key}')
        index[key] = row

    def note(self, path):
        checksum = sha256(path)
        self.evidence[str(Path(path).resolve())] = checksum
        return checksum

    def json(self, path):
        self.note(path)
        return read_json(path)

    def cache(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid semantic cache key')
        marker = self.cache_root / 'records' / (key + '.json')
        record = self.json(marker)
        if (record.get('schema_version') != 1 or record.get('cache_key') != key
                or digest(dict(schema_version=1, spec=record['spec'])) != key
                or record.get('artifact_path') != f'objects/{key}.npz'):
            raise ValueError('Semantic cache identity mismatch')
        path = safe_path(self.cache_root, record['artifact_path'])
        if self.note(path) != record['artifact_sha256']:
            raise ValueError(f'Semantic cache checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as z:
            if len(z.files) != len(set(z.files)):
                raise ValueError('Duplicate NPZ members')
            arrays = {k: z[k].copy() for k in z.files}
        description = {k: dict(dtype=v.dtype.str, shape=list(v.shape), sha256=hash_array(v))
                       for k, v in arrays.items()}
        if description != record['arrays']:
            raise ValueError('Semantic array checksums/shape mismatch')
        return arrays, record

    def frame(self, sequence, frame):
        key = sequence['sequence_id'], frame
        if key not in self.frame_cache:
            row = self.semantic_units[key]
            if (row.get('status') != 'completed' or row.get('split') != sequence['split']
                    or row.get('building_id') != sequence['building_id'] or row.get('confidence') != .5):
                raise ValueError(f'Invalid semantic unit: {key}')
            data = night_common.load_frame(self.prepared, sequence['frames'][frame])
            self.note(safe_path(self.prepared, sequence['frames'][frame]['artifact_path']))
            arrays, record = self.cache(row['native_cache_key'])
            spec = record['spec']
            if (spec.get('stage') != 'segmentation_native' or spec.get('model') != self.model
                    or spec.get('vocabulary') != self.config['vocabularies']['domain']
                    or spec.get('rgb') != hash_array(data['rgb'])):
                raise ValueError('Semantic native cache does not match RGB/model/vocabulary')
            metadata = record['metadata']
            if metadata['score_kind'] != 'independent_sigmoid':
                raise ValueError('Expected the saved independent sigmoid DINO head')
            native = dict(scores=arrays['scores'], **{k: metadata[k] for k in ('class_names', 'score_kind')})
            _, predicted, names = semantic_geometry.semantic_prediction(
                native, self.config['vocabularies']['domain'], 'fixed', .5)
            if set(names) != set(self.names) or predicted.shape[1:] != data['packed'].shape:
                raise ValueError('DINO channels/grid differ from the explicit current ontology')
            data['semantic'] = pack(predicted[[names.index(name) for name in self.names]])
            self.frame_cache[key] = data
        return self.frame_cache[key]

    def pair(self, sequence, frame):
        reference, source = self.frame(sequence, frame - 1), self.frame(sequence, frame)
        geometry = reference['support'] & source['support']
        maps, support = {}, geometry.copy()
        payload = dict(reference=reference['rgb'], source=source['rgb'], support=geometry)
        for method in METHODS:
            key = method, sequence['sequence_id'], frame
            row = self.change_units[key]
            if (row.get('status') != 'completed' or row.get('split') != sequence['split']
                    or row.get('building_id') != sequence['building_id'] or row.get('reference_frame') != frame - 1
                    or row.get('input_sha256') != payload_sha256(payload)
                    or row.get('variant') != ('current_train' if method == 'msdzip_abs' else 'frozen')):
                raise ValueError(f'Invalid change unit/input identity: {key}')
            _verify_metadata(self.contract['models'][method]['expected_metadata'], row['method_metadata'], method)
            maps[method] = night_report._load_map(self.change_root, row, reference, source, geometry)
            self.note(safe_path(self.change_root, row['artifact_path']))
            support &= maps[method]['support']
        labels = real_gt_metrics.derive_labels(reference['packed'], source['packed'], support, len(self.names))
        geometry_known = geometry & (reference['packed'] != 0) & (source['packed'] != 0)
        context = dict(split=sequence['split'], building_id=sequence['building_id'],
                       sequence_id=sequence['sequence_id'], frame=frame, reference_frame=frame - 1,
                       reference_year=sequence['frames'][frame - 1]['year'], source_year=sequence['frames'][frame]['year'])
        return dict(context=context, reference=reference, source=source, support=support, labels=labels,
                    maps=maps, geometry_pixels=int(geometry.sum()), common_support_pixels=int(support.sum()),
                    geometry_known_pixels=int(geometry_known.sum()),
                    dropped_known_pixels=int((geometry_known & ~support).sum()),
                    lpips_support_pixels=int(maps['lpips']['support'].sum()),
                    msdzip_support_pixels=int(maps['msdzip_abs']['support'].sum()),
                    valid_pixels=int((labels != 255).sum()), support_sha256=hash_array(support))


def fit_thresholds(pairs):
    choices = {}
    for method in METHODS:
        rows = []
        for pair in pairs:
            if pair['context']['split'] != 'train':
                continue
            known = pair['labels'] != 255
            rows.append(dict(pair['context'], scores=pair['maps'][method]['scores'][known],
                             labels=pair['labels'][known], support_sha256=pair['support_sha256']))
        choice = real_gt_calibration.calibrate_threshold(rows, None, GATE_FPR)
        if choice['status'] != 'calibrated':
            raise ValueError(f'Cannot calibrate {method}: no known unchanged TRAIN pixels')
        choices[method] = dict(choice, calibration_split='train', evaluation_splits=['val', 'test'])
    return choices


def aggregate(rows):
    """Do not pool splits or facades. Publish both explicitly named averaging rules."""
    keys = ('split', 'variant', 'target', 'class_name', 'direction')
    metric_names = ('precision', 'recall', 'f1', 'iou', 'fpr', 'exact_v_precision', 'exact_v_recall')
    per_facade, summaries = [], []
    mean = lambda values: float(np.mean(values)) if values else None
    for key, members in sorted(groups_from(rows, keys).items()):
        for protocol in ('all_frames', 'positive_frames'):
            current_facades = []
            for building, observations in sorted(groups_from(members, ('building_id',)).items()):
                observed = [r for r in observations if r['valid_pixels']]
                result = dict(zip(keys, key), building_id=building[0], protocol=protocol,
                              pairs=len(observations), positive_pairs=sum(r['positive_pixels'] > 0 for r in observed),
                              **{c: sum(r[c] for r in observed) for c in semantic_report.COUNTS})
                for metric in metric_names:
                    result[metric] = mean([r[metric] for r in observed if r.get(metric) is not None
                                           and (protocol == 'all_frames' or metric == 'fpr' or r['positive_pixels'] > 0)])
                current_facades.append(result)
                per_facade.append(result)
            summaries.append(dict(zip(keys, key), protocol=protocol, facades=len(current_facades),
                                  pairs=len(members), positive_pairs=sum(r['positive_pixels'] > 0 for r in members),
                                  **{c: sum(r[c] for r in current_facades) for c in semantic_report.COUNTS},
                                  **{m: mean([r[m] for r in current_facades if r[m] is not None]) for m in metric_names},
                                  **{m + '_facades': sum(r[m] is not None for r in current_facades) for m in metric_names}))
    return summaries, per_facade


def groups_from(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def gallery(out, pair, results, names):
    context, source, reference = pair['context'], pair['source'], pair['reference']
    stem = '2026-10-19-' + digest(context)[:16]
    valid = pair['labels'] != 255
    palette = {name: np.array(semantic_report._palette([{'name': name, 'id': i + 1}])[name])
               for i, name in enumerate(names)}
    panels = [('RGB раньше', reference['rgb']), ('RGB позже', source['rgb'])]
    for label, rgb, bits in [('S раньше', reference['rgb'], reference['semantic']),
                             ('S позже', source['rgb'], source['semantic'])]:
        panels.append((label, semantic_report._overlay(rgb, unpack(bits, len(names)) & valid, names, palette)))
    gt = pair['labels'] == 1
    panels.append(('GT: изменение аннотаций', semantic_report._overlay(source['rgb'], gt[None],
                                                                     ['GT'], {'GT': np.array([255, 150, 0])})))
    for name, result in results.items():
        predicted = result['C'] & valid
        error = source['rgb'].copy()
        for mask, color in ((predicted & gt, [0, 220, 0]), (predicted & ~gt, [230, 0, 0]),
                            (~predicted & gt, [0, 120, 255])):
            error[mask] = (.35 * error[mask] + .65 * np.asarray(color)).astype(np.uint8)
        panels.append((name + ': TP зелёный / FP красный / FN синий', error))
    cards = []
    for i, (label, rgb) in enumerate(panels):
        relative = f'gallery/{stem}-{i}.png'
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(rgb)
        im.thumbnail((384, 384))
        im.save(path)
        cards.append(f'<figure><img src="{relative}"><figcaption>{html.escape(label)}</figcaption></figure>')
    caption = f"{context['split']} · {context['building_id']} · {context['reference_year']} → {context['source_year']}"
    return '<article><h3>' + html.escape(caption) + '</h3><div class="grid">' + ''.join(cards) + '</div></article>'


def report(out, summaries, articles, names):
    columns = ('split', 'variant', 'target', 'f1', 'precision', 'recall', 'fpr', 'exact_v_recall', 'facades', 'pairs')
    primary = [r for r in summaries if r['class_name'] == '__all__' and r['protocol'] == 'all_frames']
    def cell(value):
        return 'N/A' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    table = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join('---' for _ in columns) + ' |\n'
    table += '\n'.join('| ' + ' | '.join(cell(r.get(c)) for c in columns) + ' |' for r in primary)
    notes = '\n'.join('- ' + item for item in LIMITATIONS)
    description = ('Основной протокол: all_frames, сначала среднее по парам внутри фасада, затем по фасадам. '
                   'В summary.csv также positive_frames: P/R/F1/IoU только на парах с GT-позитивами, FPR на всех. '
                   'N/A не заменяется единицей. directed_events считает пары (пиксель, класс, направление): '
                   'ошибка класса/направления даёт FP и FN. exact_v_recall — доля GT-изменений с найденным C '
                   'и полностью верным новым набором классов V. TRAIN используется для калибровки; '
                   'его метрики не являются проверкой обобщения.')
    (out / '2026-10-19_tables.md').write_text(description + '\n\n' + table + '\n\n' + notes + '\n', encoding='utf-8')
    th = ''.join('<th>' + c + '</th>' for c in columns)
    tr = ''.join('<tr>' + ''.join('<td>' + html.escape(cell(r.get(c))) + '</td>' for c in columns) + '</tr>' for r in primary)
    palette = semantic_report._palette([{'name': name} for name in names])
    legend = ' · '.join('<span style="border-left:16px solid rgb(' + ','.join(str(int(x)) for x in palette[name])
                        + ');padding-left:5px">' + html.escape(name) + '</span>' for name in names)
    content = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Semantic updates</title>'
               '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #bbb}'
               '.grid{display:flex;flex-wrap:wrap}figure{margin:8px;width:300px}img{max-width:100%}figcaption{font-size:13px}</style>'
               '<h1>Направленные семантические обновления</h1><p>' + html.escape(description) + '</p><ul>'
               + ''.join('<li>' + html.escape(x) + '</li>' for x in LIMITATIONS) + '</ul><table><tr>' + th + '</tr>' + tr
               + '</table><h2>Все VAL/TEST пары, без отбора лучших</h2><p>S — предсказанное состояние. '
               + 'Цвета классов: ' + legend + '</p>' + ''.join(articles) + '</html>')
    (out / '2026-10-19_report.html').write_text(content, encoding='utf-8')


def run(semantic_run, change_run, out_path, audit_only=False):
    out = Path(out_path).expanduser().resolve()
    protected = [Path(p).expanduser().resolve() for p in (semantic_run, change_run)]
    # Discover and protect source/cache directories before creating output.
    initial = read_json(protected[0] / 'run.json')
    protected += [Path(initial['config']['prepared']).expanduser().resolve(),
                  Path(initial['cache_root']).expanduser().resolve()]
    if any(out == p or out in p.parents or p in out.parents for p in protected):
        raise ValueError('Output must be separate from all input/cache directories')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(schema_version=1, kind='directed_semantic_updates', schema=SCHEMA, status='running',
                 started_utc=datetime.now(timezone.utc).isoformat(), audit_only=audit_only,
                 semantic_run=str(protected[0]), change_run=str(protected[1]), limitations=list(LIMITATIONS))
    atomic_json(out / 'run.json', state)
    source, coverage = None, []
    try:
        source = Inputs(semantic_run, change_run)
        atomic_json(out / 'resolved_config.json', dict(schema=SCHEMA, prepared=str(source.prepared),
                    category_to_channel=source.manifest['category_to_channel'], categories=source.categories,
                    variants=list(VARIANTS), methods=list(METHODS), fusion='AND', semantic_confidence=.5,
                    gate_fpr=GATE_FPR, calibration_split='train', mode='previous',
                    method_contracts=source.contract['models'], semantic_model=source.model,
                    saved_training_exposure=source.contract.get('training_exposure', {}),
                    code_sha256={p.name: sha256(p) for p in [Path(__file__), Path(night_common.__file__),
                        Path(night_report.__file__), Path(real_gt_metrics.__file__), Path(real_gt_calibration.__file__),
                        Path(semantic_geometry.__file__), Path(semantic_report.__file__)]}))
        pairs = []
        for sequence in source.manifest['sequences']:
            print(f"Проверка кэша: {sequence['split']} {sequence['building_id']}", flush=True)
            for frame in range(1, len(sequence['frames'])):
                header = dict(split=sequence['split'], building_id=sequence['building_id'],
                              sequence_id=sequence['sequence_id'], frame=frame)
                try:
                    pair = source.pair(sequence, frame)
                    pairs.append(pair)
                    coverage.append(dict(header, status='ready', **{k: pair[k] for k in
                        ('geometry_pixels', 'common_support_pixels', 'geometry_known_pixels', 'dropped_known_pixels',
                         'lpips_support_pixels', 'msdzip_support_pixels', 'valid_pixels', 'support_sha256')}))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    coverage.append(dict(header, status='blocked', error=f'{type(error).__name__}: {error}'))
        semantic_report._write_csv(out / 'coverage.csv', coverage)
        if any(r['status'] != 'ready' for r in coverage) or not pairs:
            raise ValueError('Incomplete/invalid cache; no partial-cohort scores. See coverage.csv. No inference started.')
        if {p['context']['split'] for p in pairs} != {'train', 'val', 'test'}:
            raise ValueError('This pilot requires separate TRAIN, VAL and TEST facades')
        if any(not p['valid_pixels'] for p in pairs):
            raise ValueError('Pair without common known support; see coverage.csv')
        if audit_only:
            state.update(status='audit_passed', pairs=len(pairs))
        else:
            choices = fit_thresholds(pairs)
            atomic_json(out / '2026-10-19_thresholds.json', choices)
            thresholds = {name: value['threshold'] for name, value in choices.items()}
            rows, articles, artifacts = [], [], []
            for pair in pairs:
                before, after = pair['reference']['semantic'], pair['source']['semantic']
                results = updates(before, after, pair['maps']['lpips']['scores'],
                                  pair['maps']['msdzip_abs']['scores'], thresholds, pair['support'])
                arrays = dict(before=before, after=after, support=pair['support'], labels=pair['labels'])
                for name, result in results.items():
                    for row in evaluate(before, result, pair['reference']['packed'], pair['source']['packed'],
                                        pair['support'], source.categories):
                        rows.append(dict(pair['context'], variant=name, **row))
                    arrays.update({name + '__' + key: value for key, value in result.items()})
                relative = 'maps/2026-10-19-' + digest(pair['context'])[:24] + '.npz'
                checksum = atomic_npz(out / relative, arrays)
                artifacts.append(dict(pair['context'], artifact_path=relative, artifact_sha256=checksum))
                if pair['context']['split'] != 'train':
                    articles.append(gallery(out, pair, results, source.names))
            summaries, facades = aggregate(rows)
            for name, values in (('per_pair', rows), ('per_facade', facades), ('summary', summaries)):
                semantic_report._write_csv(out / (name + '.csv'), values)
            atomic_json(out / '2026-10-19_artifacts.json', artifacts)
            report(out, summaries, articles, source.names)
            state.update(status='completed', pairs=len(pairs), result_rows=len(rows))
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
    print(f'Результат: {out}', flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-run', required=True)
    parser.add_argument('--change-run', required=True)
    parser.add_argument('--out', required=True, help='New output directory; no overwrite/resume')
    parser.add_argument('--audit-only', action='store_true', help='Verify every required cache; do not evaluate')
    args = parser.parse_args(argv)
    state = run(args.semantic_run, args.change_run, args.out, args.audit_only)
    return 0 if state['status'] in ('completed', 'audit_passed') else 130 if state['status'] == 'interrupted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
