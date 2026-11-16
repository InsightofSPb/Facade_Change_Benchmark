"""Private archive proposals, rendered on their exact assets; never online evidence.

Only JSON and raster bytes are consumed from the supplied package. Its HTML is
parsed for asset-data, not executed or served as an active page.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import html
from html.parser import HTMLParser
import importlib
import io
import json
import math
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw

from ..history_features import Fingerprints, digest
from ..io import sha256, write_json
from ..visualization import write_static_gallery
from .evaluation import box_iou
from .report import _copy
from .runner import complete_run, new_run

qa = importlib.import_module(".2026-11-10_questions", __package__)
archive = importlib.import_module(".2026-11-15_www_archive", __package__)
PROPOSALS = "2026-11-16-review-proposals.private.json"
REGISTER = "2026-11-16-asset-register.private.json"
GALLERY = "2026-11-16-side-by-side-gallery.private.html"
MAX_BYTES = 128 * 1024 * 1024
CROSS_SPLIT_POLICY = "Confirmed physical buildings crossing original splits are excluded from BOTH development and evaluation; guard unchanged"


class AssetData(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active, self.found, self.parts = False, 0, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "asset-data":
            if attrs.get("type") != "application/json":
                raise ValueError("asset-data must be inert application/json")
            self.active, self.found = True, self.found + 1

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)


def package_bytes(path, fingerprints):
    """Read named inert files without extracting or executing archive members."""
    path = Path(path).resolve()
    if path.is_dir():
        files = {p.relative_to(path).as_posix(): p for p in path.rglob("*") if p.is_file()}
        def read(name):
            source = files[name].resolve()
            if not source.is_relative_to(path) or source.stat().st_size > MAX_BYTES:
                raise ValueError("Private audit file escapes its directory or exceeds size limit")
            fingerprints.add(source)
            return source.read_bytes()
    else:
        fingerprints.add(path)
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate private audit ZIP members")
        files = dict.fromkeys(names)
        def read(name):
            with zipfile.ZipFile(path) as bundle:
                if bundle.getinfo(name).file_size > MAX_BYTES:
                    raise ValueError("Private audit member exceeds size limit")
                return bundle.read(name)
    chosen = {}
    for basename in (PROPOSALS, REGISTER, GALLERY):
        matches = [name for name in files if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous private audit member: " + basename)
        chosen[basename] = read(matches[0])
    for name in files:
        basename = Path(name).name
        if (basename not in {PROPOSALS, REGISTER, GALLERY} and ".private." in basename and
                any(word in basename.lower() for word in ("alias", "derivative", "locator", "audit")) and
                Path(name).suffix.lower() in {".json", ".csv", ".txt", ".md"}):
            if basename in chosen:
                raise ValueError("Ambiguous auxiliary audit filename")
            chosen[basename] = read(name)
    return chosen


def load_assets(raw, out):
    register = json.loads(raw[REGISTER])
    assets = register.get("assets", register)
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Asset register must contain an asset-key mapping")
    parser = AssetData()
    parser.feed(raw[GALLERY].decode("utf-8"))
    if parser.found != 1:
        raise ValueError("Expected exactly one asset-data JSON block")
    embedded = json.loads("".join(parser.parts))
    if not isinstance(embedded, dict):
        raise ValueError("asset-data must be an asset-key mapping")
    result = {}
    for key, record in assets.items():
        if not isinstance(record, dict) or not isinstance(record.get("provenance"), (dict, list, str)):
            raise ValueError("Asset requires SHA-256, native dimensions and preserved provenance: " + key)
        value = embedded.get(key)
        value = value.get("data_url") if isinstance(value, dict) else value
        if not isinstance(value, str) or not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("Missing PNG/JPEG data URL for registered asset: " + key)
        content = base64.b64decode(value.split(",", 1)[1], validate=True)
        if hashlib.sha256(content).hexdigest() != record.get("sha256"):
            raise ValueError("Private asset SHA-256 mismatch: " + key)
        with Image.open(io.BytesIO(content)) as image:
            width, height = record.get("width"), record.get("height")
            if (type(width) is not int or type(height) is not int or image.size != (width, height)
                    or image.format not in {"PNG", "JPEG"}):
                raise ValueError("Private asset native dimensions/format mismatch: " + key)
            image.verify()
            suffix = ".png" if image.format == "PNG" else ".jpg"
        relative = "assets/" + record["sha256"] + suffix
        (out / relative).write_bytes(content)
        result[key] = {**copy.deepcopy(record), "asset": key, "src": relative, "shape": [height, width]}
    return result


def checked_panel(panel, assets, public_records, evidence_ids=()):
    key = panel.get("asset")
    if key not in assets:
        raise ValueError("Panel references an unregistered asset: " + str(key))
    asset = assets[key]
    for name in ("image_sha256", "asset_sha256"):
        if name in panel and panel[name] != asset["sha256"]:
            raise ValueError("Panel coordinates belong to different image bytes: " + key)
    if panel.get("coordinate_system", "asset_native") != "asset_native":
        raise ValueError("Private panel boxes must use the registered asset_native grid")
    if panel.get("display_asset", key) != key or panel.get("transform") is not None:
        raise ValueError("Cross-image coordinate transfer requires an independently validated transform; this importer only displays the source asset")
    box = panel.get("box_xyxy")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError("Panel box needs four finite asset-native pixel coordinates")
        box_iou(box, box)
        if box[0] < 0 or box[1] < 0 or box[2] > asset["width"] or box[3] > asset["height"]:
            raise ValueError("Panel box exceeds its exact source asset")
    evidence_id = panel.get("evidence_id", asset.get("evidence_id"))
    associated = [evidence_id] if evidence_id else list(evidence_ids)
    public_assets = [{**asset, "evidence_id": identifier} for identifier in associated if identifier in public_records
                     for asset in archive.image_assets(public_records[identifier])]
    binding = ("same_asset_bytes" if any(a["sha256"] == asset["sha256"] for a in public_assets) else
               "different_public_asset_no_coordinate_transfer" if public_assets else "public_asset_not_available")
    return {**copy.deepcopy(panel), "asset": key, "asset_sha256": asset["sha256"], "shape": asset["shape"],
            "coordinate_system": "asset_native", "evidence_id": evidence_id, "public_binding": binding,
            "associated_evidence_ids": associated,
            "public_assets": [{k: a[k] for k in ("evidence_id", "sha256", "width", "height", "page_index")} for a in public_assets],
            "src": asset["src"], "provenance": asset["provenance"], "temporal_gold": False}


def candidate_link(row, candidates):
    """Resolve supplied identifiers only. An address or event description is not a crop match."""
    if candidates is None:
        return {"status": "dataset_not_supplied", "question_id": None, "missing_inputs": ["candidate_manifest", "exact_region_correspondence"], "candidates": []}
    constrained = [c for c in candidates if all(not row.get(k) or row[k] == c[k] for k in ("building_id", "view_id", "region_id", "question_id"))]
    identifiers = bool(row.get("question_id") or (row.get("region_id") and row.get("view_id")))
    matched = constrained[0] if identifiers and len(constrained) == 1 else None
    return {"status": "exact_identifiers" if matched else "unresolved", "question_id": matched["question_id"] if matched else None,
        "missing_inputs": [] if matched else ["exact_question_or_region_view_ids", "local_crop_to_archive_correspondence_review"],
        "candidates": [{k: c[k] for k in ("question_id", "question", "building_id", "view_id", "region_id", "task", "task_dates", "split")} for c in constrained],
        "scope": "identifier linkage only; comparison dates, event category and factual support remain unreviewed"}


def export(package, out, *, candidates_path=None, inventory_path=None, archive_corpus=None, identity_map=None,
           handoff_path=None, previous=None, progress=None):
    fingerprints = Fingerprints()
    raw = package_bytes(package, fingerprints)
    document = json.loads(raw[PROPOSALS])
    rows = document if isinstance(document, list) else document.get("rows", document.get("proposals"))
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("panels"), list) for r in rows):
        raise ValueError("Private proposals require rows with review_id and panels")
    if any(not isinstance(r.get("review_id"), str) or not r["review_id"] for r in rows) or len({r["review_id"] for r in rows}) != len(rows):
        raise ValueError("Private review IDs must be distinct nonempty strings")
    candidates = qa._load_candidates(candidates_path, fingerprints)["candidates"] if candidates_path else None
    if handoff_path:
        handoff = qa.read_bound(handoff_path, fingerprints)
        if not candidates_path or handoff.get("source_candidates_sha256") != sha256(candidates_path):
            raise ValueError("Private audit editor must preserve human decisions against their exact original candidate bytes")
    inventory = qa.read_bound(inventory_path, fingerprints) if inventory_path else {}
    corpus = archive.validate_package(qa.read_bound(archive_corpus, fingerprints), Path(archive_corpus).resolve().parent, fingerprints) if archive_corpus else None
    journal = new_run(out, {"private_package": str(Path(package).resolve()), "models_executed": False}, "www_private_archive_audit", progress)
    try:
        (journal.out / "assets").mkdir()
        (journal.out / "source.private").mkdir()
        if identity_map is None:
            identity_map = journal.out / "identity-review.json"
            write_json(identity_map, archive.identity.template())
        identities = archive.identity.Identities(identity_map, fingerprints)
        if inventory.get("identity_source") is not None and inventory["identity_source"] != identities.source:
            raise ValueError("Private review and inventory must retain the same identity mapping")
        source_files = {}
        for name, content in raw.items():
            target = journal.out / "source.private" / (name + ".txt" if name == GALLERY else name)
            target.write_bytes(content)  # Original HTML remains inert text.
            source_files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "saved_as": str(target.relative_to(journal.out))}
        assets = load_assets(raw, journal.out)
        records = {r["evidence_id"]: r for r in (corpus or {}).get("records", [])}
        items, sections = [], []
        for n, row in enumerate(rows):
            journal.event("case_started", case_id=row["review_id"], index=n+1, total=len(rows))
            linked = candidate_link(row, candidates)
            members = identities.candidate_members(row.get("building_id"))
            views = [v for v in inventory.get("views", []) if v["building_id"] in members]
            splits = sorted({c["split"] for c in candidates or [] if c["building_id"] in members} | {v["split"] for v in views})
            checked = []
            for panel in row["panels"]:
                try:
                    value = checked_panel(panel, assets, records, row.get("evidence_ids", []))
                    value["box_validation"] = "valid_on_exact_asset"
                except ValueError as exc:
                    value = {"original_panel": copy.deepcopy(panel), "asset": panel.get("asset"), "box_validation": "error",
                        "validation_error": str(exc), "temporal_gold": False, "public_binding": "not_validated"}
                    if panel.get("asset") in assets:
                        value["src"] = assets[panel["asset"]]["src"]
                checked.append(value)
            item = {"review_id": row["review_id"], "proposal": copy.deepcopy(row), "decision": "pending", "approved": False,
                "candidate_link": linked, "identity_review_pending": identities.pending(row.get("building_id")),
                "original_splits": splits, "eligible": None, "inventory_views": views,
                "cross_split_consequence": "same_building confirmation blocks BOTH development and evaluation selection; guard unchanged" if len(splits) > 1 else None,
                "event_category": row.get("event_category", "unreviewed"), "question_scope": row.get("question_scope", "unreviewed"),
                "panels": checked}
            panels = []
            for i, panel in enumerate(item["panels"]):
                if "src" not in panel:
                    continue
                relative = panel["src"]
                if panel["box_validation"] == "valid_on_exact_asset" and panel.get("box_xyxy") is not None:
                    with Image.open(journal.out / relative) as image:
                        image = image.convert("RGB")  # No EXIF rotation or coordinate transfer.
                        image.info.clear()  # The native-grid overlay must not be auto-rotated by a browser.
                        draw = ImageDraw.Draw(image)
                        draw.rectangle(panel["box_xyxy"], outline=(30, 160, 255), width=3)
                        draw.text((8, 8), "PRIVATE CORRESPONDENCE PROPOSAL - NOT EVENT GT", fill=(30, 160, 255))
                        relative = f"assets/proposal-{n}-{i}.png"
                        image.save(journal.out / relative)
                panels.append({"src": relative, "href": panel["src"], "label": panel["asset"] + " / " + panel["public_binding"]})
            for c in candidates or []:
                if c["question_id"] == linked["question_id"]:
                    for obs in c["observations"]:
                        fingerprints.add(obs["rgb_path"], obs["rgb_sha256"])
                        panels.append({"src": _copy(journal.out, obs["rgb_path"], journal.out, {}), "label": "Local " + obs["date"] + " / " + obs["observation_id"]})
            sections.append({"title": row["review_id"], "caption": json.dumps(item, ensure_ascii=False, indent=2), "panels": panels})
            items.append(item)
        for name, content in raw.items():
            if name not in {PROPOSALS, REGISTER, GALLERY}:
                sections.append({"title": "Pending source audit: " + name, "caption": content.decode("utf-8"), "panels": []})
        result = qa._private("www_archive_audit_review", rows=items, assets=assets, source_files=source_files,
            candidate_sha256=sha256(candidates_path) if candidates_path else None,
            archive_corpus_sha256=sha256(archive_corpus) if archive_corpus else None,
            identity_source=identities.source, input_sha256=fingerprints.files, temporal_gold=False)
        declared_corpus = document.get("corpus_sha256") if isinstance(document, dict) else None
        result["audit_claimed_corpus_sha256"] = declared_corpus
        result["corpus_binding"] = ("not_declared_in_private_package" if declared_corpus is None else
            "exact_corpus_hash" if declared_corpus == result["archive_corpus_sha256"] else "different_revision_pending_recheck")
        write_json(journal.out / "audit.private.json", result)
        linked_ids = sorted({i["candidate_link"]["question_id"] for i in items if i["candidate_link"]["question_id"]})
        linked_review = None
        if linked_ids:
            selection = journal.out / "pending-selection.private.json"
            write_json(selection, qa._private("www_question_selection", candidate_sha256=sha256(candidates_path),
                question_ids=linked_ids, policy={"basis": "exact identifiers for private review, independent of cohort eligibility"}))
            linked_review = importlib.import_module(".2026-11-15_www_curation", __package__).review_page(
                candidates_path, selection, journal.out / "candidate-review", archive_corpus=archive_corpus,
                identity_map=identity_map, handoff_path=handoff_path, previous=previous, progress=progress)
        write_static_gallery(journal.out, "PRIVATE — pending archive correspondence and unresolved candidates", sections, notes=[
            "All proposals remain pending, independently of batch eligibility. No aliases, labels or splits are approved here.",
            CROSS_SPLIT_POLICY,
            "Boxes belong to the exact registered image/SHA, not just an evidence ID. No cross-image transfer is performed.",
            "Earliest among supplied observations and earliest supported within the frozen collection are different question scopes.",
            "Architectural alterations are not automatically damage or repair. Use the existing question editor after resolving the exact candidate."])
        summary = {"status": "completed", "proposals": len(items), "pending": len(items), "approved": 0, "models_executed": False,
            "unresolved_candidate_links": sum(i["candidate_link"]["question_id"] is None for i in items),
            "cross_split_review_rows": sum(i["cross_split_consequence"] is not None for i in items),
            "cross_split_policy": CROSS_SPLIT_POLICY,
            "different_public_assets": sum(p["public_binding"] == "different_public_asset_no_coordinate_transfer" for i in items for p in i["panels"]),
            "invalid_panels": sum(p["box_validation"] == "error" for i in items for p in i["panels"]),
            "archive_corpus_sha256": result["archive_corpus_sha256"], "linked_candidate_review": linked_review,
            "corpus_binding": result["corpus_binding"],
            "report_path": str(journal.out / "index.html")}
        links = '<li><a href="gallery.html">Все pending-предложения и исходные панели</a></li>'
        if linked_review:
            links += '<li><a href="candidate-review/review.html">Редактор точно связанных кандидатов; eligibility не изменена</a></li>'
        (journal.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>PRIVATE pending audit</title>'
            '<h1>Приватный аудит: решения pending</h1><ul>' + links + '</ul><pre>' +
            html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
        write_json(journal.out / "summary.json", summary)
        fingerprints.verify()
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
