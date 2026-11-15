"""Frozen public archive interchange, using the existing Jina worker and cache."""
from __future__ import annotations

import copy
from datetime import datetime
import calendar
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from ..history_features import Fingerprints, _inside, digest
from ..io import read_json, sha256, write_json
from .models import JINA_REVISION, ModelPool, _resources_valid
from .retrieval import _space, _vector
from .runner import complete_run, new_run
from .store import _id, reject_forbidden, temporal_value


PACKAGE_KEYS = {"schema_version", "kind", "corpus_id", "created_at", "task_mode", "knowledge_cutoff", "records"}
RECORD_KEYS = {"evidence_id", "record_type", "building_id", "view_id", "identity_status", "source_page_url",
    "media_url", "title", "publisher", "creator", "source_text", "observation_date", "publication_date",
    "first_known_available_at", "collected_at", "available_at", "availability_basis", "date_source", "assets",
    "rights", "access_status", "source_warnings", "parent_evidence_id"}
ASSET_KEYS = {"path", "sha256", "media_type", "width", "height", "page_index"}


def _date(value, name, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, str) and len(value) == 7:
        try:
            year, month = map(int, value.split("-"))
            return temporal_value(f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}")
        except (ValueError, TypeError):
            raise ValueError(f"{name} needs an explicit valid calendar month") from None
    if not isinstance(value, str) or temporal_value(value) is None:
        raise ValueError(f"{name} needs an explicit valid date")
    return temporal_value(value)


def _timestamp(value, name):
    _date(value, name)
    if "T" not in value or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError(f"{name} needs an actual ISO timestamp with timezone")


def _url(value, name, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, str) or urlparse(value).scheme not in {"http", "https"} or not urlparse(value).netloc:
        raise ValueError(f"{name} must be an attributable HTTP(S) source URL")


def validate_package(document, root, fingerprints):
    reject_forbidden(document)
    if (not isinstance(document, dict) or set(document) != PACKAGE_KEYS or document["schema_version"] != 1
            or document["kind"] != "www_archival_evidence_package" or document["task_mode"] != "retrospective"):
        raise ValueError("Expected the shared www_archival_evidence_package v1 contract")
    _id(document["corpus_id"], "corpus_id")
    _timestamp(document["created_at"], "created_at")
    cutoff = _date(document["knowledge_cutoff"], "knowledge_cutoff")
    if not isinstance(document["records"], list):
        raise ValueError("Archive records must be a list")
    ids = set()
    for record in document["records"]:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ValueError("Archive record fields must match the public shared contract exactly")
        identifier = _id(record["evidence_id"], "evidence_id")
        if identifier in ids:
            raise ValueError("Duplicate archival evidence ID")
        ids.add(identifier)
        for key in ("building_id", "view_id", "parent_evidence_id"):
            if record[key] is not None:
                _id(record[key], key)
        if record["record_type"] not in {"image", "document", "image_in_document"}:
            raise ValueError("Unknown archive record type")
        if record["identity_status"] not in {"verified", "probable", "unresolved", "distractor"}:
            raise ValueError("Explicit identity_status is required")
        if record["identity_status"] == "verified" and not record["building_id"]:
            raise ValueError("Verified identity requires an actual dataset building_id")
        for key in ("source_page_url", "media_url"):
            _url(record[key], key, optional=key == "media_url")
        for key in ("title", "publisher", "creator", "source_text", "date_source"):
            if record[key] is not None and (not isinstance(record[key], str) or len(record[key]) > 100_000):
                raise ValueError(f"{key} must be bounded source text or null")
        date = record["observation_date"]
        if not isinstance(date, dict) or set(date) != {"start", "end", "precision", "basis"}:
            raise ValueError("Observation date needs start/end/precision/basis, with explicit nulls for unknown")
        start, end = (_date(date[key], "observation_date." + key, optional=True) for key in ("start", "end"))
        mixed_document = date["precision"] == "mixed_document"
        if mixed_document and (record["record_type"] != "document" or record["parent_evidence_id"] is not None
                               or start is not None or end is not None):
            raise ValueError("mixed_document is only an undated parent document, never a dated photograph")
        if date["precision"] not in {"day", "month", "year", "interval", "unknown", "mixed_document"}:
            raise ValueError("Unknown observation date precision")
        if ((start is None) != (end is None) or (start is not None and start > end)
                or (date["precision"] in {"unknown", "mixed_document"}) != (start is None)):
            raise ValueError("Inconsistent capture-date interval; unknown must remain unknown")
        if start is not None and (not date["basis"] or not record["date_source"]):
            raise ValueError("Capture dates require their source basis and locator")
        publication = _date(record["publication_date"], "publication_date", optional=True)
        known = _date(record["first_known_available_at"], "first_known_available_at", optional=True)
        _timestamp(record["collected_at"], "collected_at")
        if temporal_value(record["collected_at"]) > cutoff:
            raise ValueError("Evidence collected after the freeze needs a new consistent question/corpus cutoff")
        available = _date(record["available_at"], "available_at")
        basis = record["availability_basis"]
        anchor = {"documented_publication": publication, "documented_archive_snapshot": known,
                  "collection_time": temporal_value(record["collected_at"])}.get(basis)
        if anchor is None or available < anchor or available > cutoff:
            raise ValueError("Availability must be conservative, sourced and within the frozen corpus cutoff; never backdate")
        if start is not None and end > available:
            raise ValueError("A captured image cannot be available before its asserted observation interval")
        if not isinstance(record["rights"], dict) or set(record["rights"]) != {"statement", "url"}:
            raise ValueError("Record rights statement/URL explicitly, including nulls")
        _url(record["rights"]["url"], "rights.url", optional=True)
        if record["access_status"] not in {"downloaded", "metadata_only", "blocked"} or not isinstance(record["assets"], list):
            raise ValueError("Explicit access status and assets are required")
        if (record["access_status"] == "downloaded") != bool(record["assets"]):
            raise ValueError("Downloaded status must describe actual supplied assets")
        if not isinstance(record["source_warnings"], list) or any(not isinstance(x, str) for x in record["source_warnings"]):
            raise ValueError("source_warnings must contain source-intrinsic text, not relevance labels")
        for asset in record["assets"]:
            if not isinstance(asset, dict) or set(asset) != ASSET_KEYS:
                raise ValueError("Archive asset fields do not match the shared contract")
            path = _inside(root, asset["path"])
            if not isinstance(asset["sha256"], str) or len(asset["sha256"]) != 64:
                raise ValueError("Every asset needs an actual SHA-256")
            fingerprints.add(path, asset["sha256"])
            if not isinstance(asset["media_type"], str):
                raise ValueError("Asset media_type is required")
            if asset["media_type"].startswith("image/"):
                with Image.open(path) as image:
                    if image.size != (asset["width"], asset["height"]):
                        raise ValueError("Archive image dimensions disagree with actual bytes")
            elif asset["width"] is not None or asset["height"] is not None:
                raise ValueError("Non-image asset dimensions must be null")
            if asset["page_index"] is not None and (type(asset["page_index"]) is not int or asset["page_index"] < 1):
                raise ValueError("Document page indices are one-based positive integers")
        if record["record_type"] == "image_in_document" and (not record["parent_evidence_id"] or
                not any(a["page_index"] is not None for a in record["assets"])):
            raise ValueError("Extracted document images need the parent record and original page")
    for record in document["records"]:
        if record["parent_evidence_id"] is not None and record["parent_evidence_id"] not in ids - {record["evidence_id"]}:
            raise ValueError("Archive parent ID must name another included source record")
    return document


def import_archive(package_path, out):
    path = Path(package_path).expanduser().resolve()
    fingerprints = Fingerprints()
    fingerprints.add(path)
    package = validate_package(read_json(path), path.parent, fingerprints)
    journal = new_run(out, {"package_sha256": sha256(path)}, "www_archive_import")
    try:
        (journal.out / "assets").mkdir()
        # Retain exact source bytes (including legacy precision) as well as its recorded hash.
        shutil.copyfile(path, journal.out / "source_package.json")
        if sha256(journal.out / "source_package.json") != fingerprints.files[str(path)]:
            raise ValueError("Original archive package changed during import")
        public = copy.deepcopy(package)
        for record in public["records"]:
            for asset in record["assets"]:
                source = _inside(path.parent, asset["path"])
                relative = "assets/" + asset["sha256"] + source.suffix.lower()
                destination = journal.out / relative
                if not destination.exists():
                    shutil.copyfile(source, destination)
                if sha256(destination) != asset["sha256"]:
                    raise ValueError("Copied archive bytes changed")
                asset["path"] = relative
        write_json(journal.out / "corpus.json", public)
        write_json(journal.out / "record_hashes.json", {r["evidence_id"]: digest(r) for r in public["records"]})
        fingerprints.verify()
        summary = {"status": "completed", "records": len(public["records"]), "corpus": str(journal.out / "corpus.json"),
                   "corpus_sha256": sha256(journal.out / "corpus.json"), "empty_corpus": not public["records"]}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def image_assets(record):
    """The existing deterministic display order; one evidence ID exposes its first image."""
    return sorted((a for a in record["assets"] if a["media_type"].startswith("image/")),
                  key=lambda a: (a["page_index"] or 0, a["path"]))


def reviewed_earliest(case, label, fingerprints):
    """Validate a PRIVATE human match to a dated image in the frozen PUBLIC corpus.

    No retrieval, inference, automatic support approval or coordinate registration.
    The corpus hash also binds source/date basis, PDF lineage and all asset bytes.
    """
    binding = label["earliest_archive"]
    keys = {"corpus_path", "corpus_sha256", "record_sha256", "asset_sha256",
            "building_id", "view_id", "region_id", "match_basis"}
    if not isinstance(binding, dict) or set(binding) != keys:
        raise ValueError("earliest_archive needs the exact private corpus/record/asset and reviewed identity binding")
    for key in ("corpus_sha256", "record_sha256", "asset_sha256"):
        value = binding[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("earliest_archive requires actual SHA-256 values: " + key)
    if not isinstance(binding["corpus_path"], str) or not Path(binding["corpus_path"]).is_absolute():
        raise ValueError("earliest_archive.corpus_path must be the explicit absolute frozen corpus path")
    path = Path(binding["corpus_path"])
    fingerprints.add(path, binding["corpus_sha256"])
    corpus = validate_package(read_json(path), path.parent, fingerprints)
    if case["task_mode"] != "retrospective" or case["knowledge_cutoff"] != corpus["knowledge_cutoff"]:
        raise ValueError("Reviewed archive earliest requires the same retrospective question/corpus freeze")
    record = next((r for r in corpus["records"] if r["evidence_id"] == label.get("earliest_observation_id")), None)
    if record is None:
        raise ValueError("Reviewed earliest observation ID does not exist in the frozen archive")
    if digest(record) != binding["record_sha256"]:
        raise ValueError("Reviewed earliest archive record hash mismatch")
    if (record["identity_status"] != "verified" or record["building_id"] != case["building_id"]
            or record["view_id"] not in {None, case["view_id"]}
            or any(binding[k] != case[k] for k in ("building_id", "view_id", "region_id"))):
        raise ValueError("Reviewed archive image has incompatible or unverified building/view/region identity")
    if not isinstance(binding["match_basis"], str) or not binding["match_basis"].strip():
        raise ValueError("Record the human-reviewed visual match to this target region; identity is not automatic support")
    date = record["observation_date"]
    if record["record_type"] not in {"image", "image_in_document"} or date["precision"] not in {"year", "month", "day", "interval"} or date["start"] is None or date["end"] is None:
        raise ValueError("A dated earliest target must be a photograph with a documented capture interval; publication is not capture")
    assets = image_assets(record)
    if not assets or assets[0]["sha256"] != binding["asset_sha256"]:
        raise ValueError("Reviewed earliest asset must be the actual image exposed for this evidence ID")
    support = label.get("supporting_evidence_ids")
    if label.get("archival_status") != "verified_useful" or not isinstance(support, list) or record["evidence_id"] not in support:
        raise ValueError("Archive earliest needs explicit human-reviewed supporting_evidence_ids and archival_status=verified_useful")
    interval = {k: date[k] for k in ("start", "end", "precision")}
    if label.get("present_by_interval") != interval or label.get("present_by") is not None:
        raise ValueError("Archive present_by_interval must preserve the exact capture interval/precision; do not substitute a scalar present_by")
    return interval


def source_excerpt(record, maximum):
    text = "\n".join(record[key] for key in ("title", "publisher", "source_text") if record[key])
    return text[:maximum], {"policy": "natural record prefix; no chunking", "original_characters": len(text),
                           "used_characters": min(len(text), maximum), "truncated": len(text) > maximum}


def checked_embedding(response, synthetic=False):
    if response.get("status") != "ok" or len(response.get("vectors", [])) != 1:
        raise ValueError("Jina embedding failed; no lexical/fake fallback: " + str(response.get("error")))
    space = _space(response["space"])
    if not synthetic and (space["model"] != "jinaai/jina-clip-v2" or space["revision"] != JINA_REVISION
                          or space["representation"] != "joint_image_text"):
        raise ValueError("Expected the pinned real Jina-CLIP-v2 joint space")
    return _vector(response["vectors"][0]), space


def index_archive(corpus_path, config, out, *, text_chars=600, pool_factory=None, synthetic=False, progress=None):
    if type(text_chars) is not int or not 1 <= text_chars <= 8192:
        raise ValueError("Record prefix must be 1..8192 characters, recorded before retrieval")
    path = Path(corpus_path).expanduser().resolve()
    fingerprints = Fingerprints()
    fingerprints.add(path)
    package = validate_package(read_json(path), path.parent, fingerprints)
    if "jina" not in config["models"]:
        raise ValueError("Jina adapter must be configured; no lexical fallback")
    journal = new_run(out, {"corpus_sha256": sha256(path), "jina": config["models"]["jina"],
                           "text_chars": text_chars, "synthetic_fixture": synthetic}, "www_archive_index", progress)
    factory = pool_factory or ModelPool
    try:
        entries, calls, space = [], [], None
        with factory({"jina": config["models"]["jina"]}, journal.out / "workers", gpu_mode=config["gpu_mode"],
                     timeout=config["worker_timeout"], cache_dir=config["cache_root"]) as pool:
            pool.progress = progress
            for record in sorted(package["records"], key=lambda r: r["evidence_id"]):
                text, truncation = source_excerpt(record, text_chars)
                requests = [("text", {"op": "embed", "texts": [text]})] if text else []
                requests += [("image", {"op": "embed", "image_paths": [str(_inside(path.parent, a["path"]))]})
                             for a in record["assets"] if a["media_type"].startswith("image/")]
                for modality, request in requests:
                    response = pool.call("jina", request)
                    calls.append({"evidence_id": record["evidence_id"], "request": request, "response": response})
                    write_json(journal.out / "calls.json", calls)
                    vector, current = checked_embedding(response, synthetic)
                    if space is not None and space != current:
                        raise ValueError("Jina space changed within a frozen archive")
                    space = current
                    entries.append({"evidence_id": record["evidence_id"], "modality": modality,
                                    "vector": vector, "record_sha256": digest(record), "text_selection": truncation})
        index = {"schema_version": 1, "kind": "www_archive_jina_index", "corpus_path": str(path),
                 "corpus_sha256": sha256(path), "space": space, "entries": entries, "text_chars": text_chars,
                 "jina_options": config["models"]["jina"], "synthetic_fixture": synthetic,
                 "resource_contracts": [r["response"].get("provenance", {}).get("resources", []) for r in calls]}
        write_json(journal.out / "index.json", index)
        write_json(journal.out / "calls.json", calls)
        fingerprints.verify()
        summary = {"status": "completed", "records": len(package["records"]), "vectors": len(entries),
                   "index": str(journal.out / "index.json"), "empty_corpus": not package["records"]}
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


class ArchiveSearch:
    """An adapter over saved Jina vectors; archive images keep their own frames."""
    def __init__(self, index_path, config, fingerprints, *, synthetic=False):
        path = Path(index_path).expanduser().resolve()
        fingerprints.add(path)
        self.index = read_json(path)
        reject_forbidden(self.index)
        if self.index.get("kind") != "www_archive_jina_index" or self.index.get("synthetic_fixture") != synthetic:
            raise ValueError("Incompatible or synthetic archive index")
        if self.index["jina_options"] != config["models"].get("jina"):
            raise ValueError("Archive/query Jina configuration differs; rebuild the index")
        if not all(_resources_valid(r) for r in self.index["resource_contracts"]):
            raise ValueError("Jina cached resources are missing or changed")
        if self.index["entries"] and not synthetic:
            if not self.index["resource_contracts"] or not all(self.index["resource_contracts"]):
                raise ValueError("A real Jina index requires recorded model resources")
            checked_embedding({"status": "ok", "vectors": [self.index["entries"][0]["vector"]],
                               "space": self.index["space"]})
        self.root = Path(self.index["corpus_path"]).parent
        fingerprints.add(self.index["corpus_path"], self.index["corpus_sha256"])
        self.corpus = validate_package(read_json(self.index["corpus_path"]), self.root, fingerprints)
        self.records = {r["evidence_id"]: r for r in self.corpus["records"]}
        for entry in self.index["entries"]:
            if entry["record_sha256"] != digest(self.records[entry["evidence_id"]]):
                raise ValueError("Indexed record hash changed")
            entry["vector"] = _vector(entry["vector"])
        self.synthetic = synthetic
        self.sha256 = sha256(path)

    def eligible(self, question):
        if question["task_mode"] != "retrospective" or question["knowledge_cutoff"] != self.corpus["knowledge_cutoff"]:
            raise ValueError("Use a consistently frozen retrospective question/corpus version; no silent backdating")
        allowed, excluded = [], []
        for row in sorted(self.records.values(), key=lambda r: r["evidence_id"]):
            reason = None
            if row["identity_status"] != "verified":
                reason = "identity_not_verified; not corroboration"
            elif row["building_id"] != question["building_id"] or row["view_id"] not in {None, question["view_id"]}:
                reason = "different_building_or_view"
            elif _date(row["available_at"], "available_at") > _date(question["knowledge_cutoff"], "knowledge_cutoff"):
                reason = "after_knowledge_cutoff"
            elif not any(e["evidence_id"] == row["evidence_id"] for e in self.index["entries"]):
                reason = "no_indexable_source_content"
            if reason:
                excluded.append({"evidence_id": row["evidence_id"], "reason": reason})
            else:
                allowed.append(row["evidence_id"])
        return allowed, excluded

    def search(self, question, query, image_path, pool, limit, exposed=(), on_call=None, deadline=None):
        allowed, excluded = self.eligible(question)
        if not allowed:
            return {"status": "no_matches", "matches": [], "calls": [], "excluded": excluded}
        requests = [{"op": "embed", "texts": [query], "task": "retrieval.query"}]
        if image_path:
            requests.append({"op": "embed", "image_paths": [str(image_path)]})
        vectors, calls = [], []
        for request in requests:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("QA time budget exhausted before Jina request")
                request["timeout_seconds"] = remaining
            response = pool.call("jina", request)
            if on_call:
                on_call("jina", request, response)
            vector, space = checked_embedding(response, self.synthetic)
            if space != self.index["space"]:
                raise ValueError("Query and archive embeddings do not share the same verified Jina space")
            vectors.append(vector)
            calls.append({"request": request, "response": response})
        scores = {}
        for entry in self.index["entries"]:
            name = entry["evidence_id"]
            if name in allowed and name not in exposed:
                if any(len(q) != len(entry["vector"]) for q in vectors):
                    raise ValueError("Embedding dimensions differ inside Jina space")
                score = max(float(np.dot(q, entry["vector"])) for q in vectors)
                scores[name] = max(score, scores.get(name, -2.))
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        return {"status": "ok" if ranked else "no_matches", "matches": [
            {"evidence_id": name, "score": scores[name]} for name in ranked[:limit]], "calls": calls,
            "ranking": [{"evidence_id": name, "score": scores[name]} for name in ranked], "excluded": excluded,
            "ranking_policy": "max cosine across text/image query and record modalities; evidence_id tie-break"}
