#!/usr/bin/env python3
"""Compose pair comparisons and chronological RGB sheets from saved facade reports.

Inputs: a report HTML with adjacent gallery/ images, or its .tar.gz review archive.
Dependencies: Python >= 3.9 and Pillow. No inference, threshold selection or metrics.
Supports the 2026-10-19 semantic-updates and directed-delta report formats.
"""
import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.articles = []
        self.article = None
        self.figure = None
        self.capture = None
        self.title = ''

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article':
            self.article = {'title': '', 'figures': []}
        elif tag in ('title', 'h1') and self.article is None:
            self.capture = 'document_title'
        elif tag == 'h3' and self.article is not None:
            self.capture = 'article_title'
        elif tag == 'figure' and self.article is not None:
            self.figure = {'src': '', 'caption': ''}
        elif tag == 'img' and self.figure is not None:
            self.figure['src'] = attrs.get('src', '')
        elif tag == 'figcaption' and self.figure is not None:
            self.capture = 'caption'

    def handle_data(self, data):
        if self.capture == 'document_title':
            self.title += data
        elif self.capture == 'article_title':
            self.article['title'] += data
        elif self.capture == 'caption':
            self.figure['caption'] += data

    def handle_endtag(self, tag):
        if tag in ('h3', 'figcaption', 'title', 'h1'):
            self.capture = None
        elif tag == 'figure' and self.figure is not None:
            self.article['figures'].append(self.figure)
            self.figure = None
        elif tag == 'article' and self.article is not None:
            self.articles.append(self.article)
            self.article = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    if '://' in relative or relative.startswith('data:'):
        raise ValueError('Only local, saved image files are supported: ' + relative[:80])
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Path leaves the report directory: ' + relative)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def extract_archive(path, directory):
    # Extract regular files only, reject links and traversal, never execute contents.
    with tarfile.open(path, 'r:*') as t:
        for member in t.getmembers():
            target = (directory / member.name).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError('Unsafe archive member: ' + member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(member) as src, target.open('xb') as dst:
                    while True:
                        data = src.read(1024 * 1024)
                        if not data:
                            break
                        dst.write(data)
            else:
                raise ValueError('Archive contains a non-regular member: ' + member.name)
    reports = sorted(directory.rglob('*report.html'))
    if len(reports) != 1:
        raise ValueError(f'Expected one *report.html; found {len(reports)}. Use --report explicitly.')
    return reports[0]


def load_report(path):
    parser = ReportParser()
    parser.feed(path.read_text(encoding='utf-8'))
    if not parser.articles:
        raise ValueError('No supported <article><h3>...<figure> records in this report.')
    root = path.parent
    artifact_files = sorted(root.glob('*artifacts.json'))
    metadata = []
    for p in artifact_files:
        obj = json.loads(p.read_text(encoding='utf-8'))
        metadata.extend(obj.get('maps', []))
    pairs = []
    for article in parser.articles:
        title = article['title'].strip()
        match = re.fullmatch(r'(.+?)\s*·\s*(.+?)\s*·\s*(\d{4})\s*→\s*(\d{4})', title)
        if not match:
            raise ValueError('Unrecognized pair title: ' + title)
        split, building, before, after = match.groups()
        if int(before) >= int(after):
            raise ValueError('Non-chronological pair: ' + title)
        figures = article['figures']
        for fig in figures:
            fig['path'] = safe_path(root, fig['src'])
            fig['caption'] = fig['caption'].strip()
        by_caption = {fig['caption'].split(':', 1)[0]: fig for fig in figures}
        if len(by_caption) != len(figures):
            raise ValueError('Duplicate figure labels: ' + title)
        for required in ('RGB раньше', 'RGB позже', 'GT'):
            if required not in by_caption:
                raise ValueError(f'Missing {required}: {title}')
        prefix = Path(figures[0]['src']).stem.rsplit('-', 1)[0]
        found = [m for m in metadata if Path(m.get('artifact_path', '')).stem.startswith(prefix)
                 and m.get('split') == split and m.get('building_id') == building]
        if metadata and len(found) != 1:
            raise ValueError('Ambiguous artifact identity: ' + title)
        sequence = found[0]['sequence_id'] if found else building
        pairs.append({'split': split, 'building_id': building, 'sequence_id': sequence,
                      'year_from': int(before), 'year_to': int(after), 'title': title,
                      'panels': by_caption, 'source_artifact': found[0] if found else None})
    return pairs, parser.title, root


def get_font(size, bold=False, override=None):
    candidates = ([override] if override else []) + [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        'DejaVuSans.ttf', 'Arial.ttf', 'C:/Windows/Fonts/arial.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise RuntimeError('A Cyrillic-capable font is required. Pass --font /path/to/font.ttf')


def wrap(text, draw, font, width):
    lines = []
    current = ''
    for token in text.split():
        candidate = (current + ' ' + token).strip()
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return '\n'.join(lines)


def make_sheet(items, title, subtitle, footer, destination, tile, font_path):
    rows = math.ceil(len(items) / 5)
    columns = math.ceil(len(items) / rows)
    margin, gap, head, label_h, foot = 36, 20, 118, 76, 74
    width = 2 * margin + columns * tile + (columns - 1) * gap
    height = head + rows * (tile + label_h + gap) + foot
    canvas = Image.new('RGB', (width, height), 'white')
    d = ImageDraw.Draw(canvas)
    heading = get_font(30, True, font_path)
    normal = get_font(20, False, font_path)
    small = get_font(17, False, font_path)
    while d.textlength(title, font=heading) > width - 2 * margin:
        heading = get_font(heading.size - 1, True, font_path)
        if heading.size < 14:
            raise ValueError('Title too long for the requested layout.')
    d.text((margin, 20), title, font=heading, fill='#172b42')
    d.multiline_text((margin, 64), wrap(subtitle, d, small, width-2*margin),
                     font=small, fill='#526071', spacing=4)
    for i, (label, path) in enumerate(items):
        x = margin + (i % columns) * (tile + gap)
        y = head + (i // columns) * (tile + label_h + gap)
        d.multiline_text((x, y), wrap(label, d, normal, tile), font=normal,
                         fill='#172b42', spacing=3)
        image = Image.open(path).convert('RGB')
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        ix, iy = x + (tile-image.width)//2, y + label_h + (tile-image.height)//2
        canvas.paste(image, (ix, iy))
        d.rectangle((x-1, y+label_h-1, x+tile, y+label_h+tile), outline='#d5dbe1')
    d.multiline_text((margin, height-foot+8), wrap(footer, d, small, width-2*margin),
                     font=small, fill='#526071', spacing=4)
    canvas.save(destination)


def slug(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')[:100] or 'facade'


def run(args, report, input_descriptor):
    all_pairs, report_title, root = load_report(report)
    splits = set(args.split)
    if 'train' in splits:
        splits.update(['train_fit', 'train_fusion_oof'])
    pairs = [p for p in all_pairs if ('all' in splits or p['split'] in splits)
             and (not args.facade or any(s.casefold() in p['building_id'].casefold() for s in args.facade))]
    pairs.sort(key=lambda p:(p['split'],p['building_id'],p['sequence_id'],p['year_to']))
    if not pairs:
        raise ValueError('No matching pairs. Available splits: ' + ', '.join(sorted({p['split'] for p in all_pairs})))
    methods = args.methods
    if methods is None:
        methods = ['semantic_delta' if 'semantic_delta' in pairs[0]['panels'] else 'semantic', 'semantic_lpips']
    for p in pairs:
        missing = set(methods) - set(p['panels'])
        if missing:
            raise ValueError(f"Missing methods {sorted(missing)} for {p['title']}. Available: {list(p['panels'])}")
    groups = {}
    for p in pairs:
        groups.setdefault((p['split'],p['building_id'],p['sequence_id']),[]).append(p)
    print(f'Pairs: {len(pairs)}; sequences: {len(groups)}; facades: {len({p[1] for p in groups})}', flush=True)
    for (split, building, _), pp in groups.items():
        years = sorted({y for p in pp for y in (p['year_from'],p['year_to'])})
        print(f'{split:18} {building:40} {len(pp):2} pairs | ' + ', '.join(map(str,years)))
    print('Selected methods: ' + ', '.join(methods))
    if args.list:
        return
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'pairs').mkdir();(out/'sequences').mkdir()
    run_file = root/'run.json'
    contract = json.loads(run_file.read_text()) if run_file.is_file() else {}
    protocol = contract.get('feature_design', contract.get('kind', report_title))
    gallery = []
    manifest_rows = []
    source_hashes = {}
    pair_count = len(pairs)
    for i,p in enumerate(pairs):
        key = hashlib.sha256((p['split']+'|'+p['sequence_id']+'|'+str(p['year_to'])).encode()).hexdigest()[:10]
        filename = f"{slug(p['building_id'])}_{p['year_from']}-{p['year_to']}_{key}.png"
        rel = 'pairs/'+filename
        panel_keys = ['RGB раньше','RGB позже','GT'] + methods
        labels = ['Предыдущее наблюдение','Текущее наблюдение','GT: изменение аннотаций'] + methods
        items = [(label,p['panels'][k]['path']) for label,k in zip(labels,panel_keys)]
        make_sheet(items, f"{p['building_id']} · {p['year_from']} → {p['year_to']}",
                   f"{p['split']} · {protocol} · сохранённые предсказания",
                   'GT — изменение аннотаций. Зелёный — TP; красный — FP; синий — FN. Маски и пороги не изменялись.',
                   out/rel,args.tile,args.font)
        for k in panel_keys:
            fig=p['panels'][k]
            source_hashes[fig['src']]=sha256(fig['path'])
        manifest_rows.append({k:p[k] for k in ['split','building_id','sequence_id','year_from','year_to']} |
                             {'image':rel,'methods':';'.join(methods)})
        gallery.append((p['split'],p['building_id'],p['title'],rel))
        print(f'Pair {i+1}/{pair_count}: {rel}',flush=True)
    sequence_links = []
    for (split,building,sequence), pp in groups.items():
        frames={}
        for p in pp:
            for year,panel in [(p['year_from'],'RGB раньше'),(p['year_to'],'RGB позже')]:
                f=p['panels'][panel]['path']
                image=Image.open(f).convert('RGB')
                digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                if year in frames and frames[year][1]!=digest:
                    raise ValueError(f'Conflicting RGB for {sequence}, {year}. Refusing to merge different views into one sequence.')
                frames[year]=(f,digest)
        ordered=sorted(frames)
        key=hashlib.sha256((split+'|'+sequence).encode()).hexdigest()[:10]
        rel=f'sequences/{slug(building)}_{key}.png'
        items=[(str(y),frames[y][0]) for y in ordered]
        make_sheet(items,building,f'{split} · {len(ordered)} наблюдений · {ordered[0]}–{ordered[-1]}',
                   'Хронологическая последовательность RGB из выбранных пар. Геометрия и цвета сохранены; подписи добавлены при компоновке.',
                   out/rel,args.tile,args.font)
        sequence_links.append((split,building,rel,len(ordered)))
    with (out/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest_rows[0]));w.writeheader();w.writerows(manifest_rows)
    provenance = {'created_utc':datetime.now(timezone.utc).isoformat(),'source':input_descriptor,
                  'source_report_sha256':sha256(report),'source_run':contract,
                  'script_sha256':sha256(Path(__file__)),'selection':vars(args)|{'out':str(out)},
                  'pairs':manifest_rows,'source_image_sha256':source_hashes,
                  'operation':'Display-only composition of saved panels; no inference, new masks, calibration or metrics',
                  'selection_policy':'All matching pairs, sorted by facade and year; no selection by performance',
                  'sequence_policy':'Same sequence and split only; duplicate-year RGB must be pixel-identical',
                  'display_resize':'Thumbnail to tile size, preserving aspect; never upscale'}
    provenance['selection']={k:str(v) if isinstance(v,Path) else v for k,v in provenance['selection'].items()}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    esc=html.escape
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пары и последовательности фасадов</title>',
          '<style>body{font:17px system-ui,sans-serif;margin:30px;color:#172b42}img{max-width:100%;height:auto}article{margin:34px 0;border-top:1px solid #ddd;padding-top:14px}a{color:#17608e}p{max-width:1100px}</style>',
          '<h1>Пары и последовательности фасадов</h1>',
          f'<p>{len(pairs)} пар; {len(groups)} последовательностей. Протокол: {esc(protocol)}.</p>',
          '<p>Сохранённые предсказания из исходного отчёта. TRAIN служит для иллюстрации; это не независимая оценка обобщения. '
          'train_fusion_oof исключает фасад из обучения сочетания, но базовые модели могли видеть его.</p>',
          '<p>Эти изображения относятся к указанному протоколу; не переносите на них подписи или численные результаты другого опыта.</p>',
          '<h2>Последовательности</h2>']
    for split,building,rel,n in sequence_links:
        page += [f'<article><h3>{esc(building)} · {esc(split)} · {n} наблюдений</h3><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>']
    page += ['<h2>Сравнения пар</h2>']
    for split,building,title,rel in gallery:
        page += [f'<article><h3>{esc(title)}</h3><a href="{rel}"><img loading="lazy" src="{rel}"></a></article>']
    page += ['</html>']
    (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print('READY: '+str(out/'index.html'),flush=True)


def _main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive',type=Path,help='Saved review .tar.gz; extracted temporarily')
    source.add_argument('--report',type=Path,help='Report HTML with adjacent gallery/ files')
    ap.add_argument('--split',nargs='+',default=['train'],help='train (all TRAIN modes), val, test, all, or exact report split')
    ap.add_argument('--facade',nargs='+',help='Keep facade IDs containing any of these substrings')
    ap.add_argument('--methods',nargs='+',help='Exact method IDs from report; default semantic_delta/semantic + semantic_lpips')
    ap.add_argument('--out',type=Path,help='New or empty output directory')
    ap.add_argument('--tile',type=int,default=384,help='Display tile side in pixels; default 384')
    ap.add_argument('--font',type=str,help='Optional path to a Cyrillic-capable TTF font')
    ap.add_argument('--list',action='store_true',help='Print inventory without creating outputs')
    args=ap.parse_args(argv)
    if not args.list and args.out is None:ap.error('--out is required unless --list is used')
    if args.tile<256 or args.tile>1024:ap.error('--tile must be between 256 and 1024')
    if args.archive:
        path=args.archive.expanduser().resolve()
        descriptor={'archive_name':path.name,'archive_sha256':sha256(path)}
        with tempfile.TemporaryDirectory(prefix='facade-gallery-') as td:
            run(args,extract_archive(path,Path(td)),descriptor)
    else:
        path=args.report.expanduser().resolve()
        run(args,path,{'report_path':str(path),'report_sha256':sha256(path)})


def main(argv=None):
    try:
        _main(argv)
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
