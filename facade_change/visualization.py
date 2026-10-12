"""Shared static photo panels for semantic evaluation and temporal histories.

Rendering only: this module never changes predictions, labels or evaluation
support. Assets stay separate from HTML, so large galleries load progressively.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import quote


def write_static_gallery(out, title, sections, legend=(), notes=()):
    """Write one gallery.html from local image panels, with native image zoom.

    Sections contain title, optional caption and panels. Each panel has label,
    src and an optional href for its full-resolution image. All asset paths are
    relative to out. There are no menus, review controls or external resources.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def asset(value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Gallery assets must stay within the output directory")
        resolved = (out / path).resolve()
        if not resolved.is_relative_to(out) or not resolved.is_file():
            raise ValueError(f"Missing or external gallery asset: {value}")
        return html.escape(quote(path.as_posix(), safe="/"), quote=True)

    chips = []
    for entry in legend:
        color = str(entry["color"])
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Legend colors must use #RRGGBB")
        chips.append('<span class="key"><i style="background:' + color + '"></i>'
                     + html.escape(str(entry["name"])) + '</span>')
    content = []
    for section in sections:
        panels = []
        for panel in section.get("panels", ()):
            source = asset(panel["src"])
            original = asset(panel.get("href", panel["src"]))
            label = html.escape(str(panel["label"]))
            panels.append(f'<figure><figcaption>{label}</figcaption>'
                          f'<a href="{original}" target="_blank" rel="noopener">'
                          f'<img src="{source}" alt="{label}" loading="lazy" decoding="async"></a></figure>')
        content.append('<section><h2>' + html.escape(str(section["title"])) + '</h2>'
                       + ('<p class="caption">' + html.escape(str(section["caption"])) + '</p>'
                          if section.get("caption") else '')
                       + '<div class="panels">' + ''.join(panels) + '</div></section>')
    title = html.escape(str(title))
    document = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>''' + title + '''</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#14181d;color:#ecf1f5}
*{box-sizing:border-box}body{margin:0;padding:24px clamp(12px,2vw,40px) 64px}
h1{font-size:clamp(22px,2.5vw,34px);margin:0 0 12px;line-height:1.2}
h2{font-size:18px;line-height:1.4;margin:0 0 8px;overflow-wrap:anywhere}
p{line-height:1.5;max-width:1400px;color:#bfcbd6;font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;padding:16px 0;border-bottom:1px solid #35414c}
.key{display:inline-flex;align-items:center;gap:7px;font-size:12px}
.key i{display:inline-block;width:13px;height:13px;border:1px solid #8192a3;border-radius:2px}
section{margin-top:28px;padding-top:18px;border-top:1px solid #35414c}
.caption{white-space:pre-wrap;margin:0 0 14px}
.panels{display:flex;align-items:flex-start;gap:10px;overflow-x:auto;padding-bottom:6px}
figure{margin:0;flex:1 0 180px;min-width:180px;max-width:100%;background:#090c10;border:1px solid #28343f}
figcaption{font-size:12px;font-weight:600;padding:9px 10px;min-height:34px}
img{display:block;width:100%;height:auto;object-fit:contain}a{color:inherit;cursor:zoom-in}
@media(max-width:700px){figure{flex-basis:75vw}body{padding:18px 12px 40px}}
@media print{:root{color-scheme:light;background:#fff;color:#111}body{padding:0}section{break-inside:avoid}p{color:#333}.panels{overflow:visible}figure{min-width:0}}
</style><header><h1>''' + title + '</h1>' + ''.join(
        '<p>' + html.escape(str(note)) + '</p>' for note in notes) + (
        '<p>Нажмите на изображение, чтобы открыть его отдельно; используйте масштабирование браузера.</p>'
        '<div class="legend">' + ''.join(chips) + '</div></header>') + ''.join(content) + '</html>\n'
    path = out / "gallery.html"
    path.write_text(document, encoding="utf-8")
    return path
