"""Local causal retrieval and sourced concepts, added 2026-11-08.

Exact physical history is resolved by RegionStore, not by nearest neighbours.
Embedding computation is external; no synthetic replacement vector is produced
when DINO/Jina weights are unavailable. JSON query caches are derived indexes.
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..history_features import NORMALIZATION, digest
from ..io import read_json, sha256, write_json
from .store import (REGISTERED, RegionStore, _allow, _id, causally_available, checked_artifact,
                    reject_forbidden, temporal_value)


def _tokens(text):
    return re.findall(r"[\w]+", str(text).casefold().replace("_", " "), flags=re.UNICODE)


def _vector(value):
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not 1 <= vector.size <= 16384 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("A zero vector is not an embedding")
    return (vector / norm).tolist()


def dino_embedding_space(checkpoint_sha256, python_tree_sha256, *, encoder_contract=None):
    """Identify the actual frozen DINO visual space without importing torch.

    Both the live adapter and existing history extractor use DinoEncoder. Native
    historical crops are divisible by 14, so live bottom/right unsupported
    padding is an identity operation on those crops. Paths, device, library
    versions and optional CLS/patch-token storage remain execution provenance;
    checkpoint, source and encoder/loader code hashes remain space identity.
    Recorded preprocessing is validated, never silently replaced by defaults.
    """
    def fingerprint(value, label):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"DINO {label} must be a complete SHA-256")
        return value

    checkpoint = fingerprint(checkpoint_sha256, "checkpoint")
    source = fingerprint(python_tree_sha256, "source tree")
    required = {"schema_version": 1, "backbone": "dinov2_vits14", "dimensions": 384,
        "checkpoint_sha256": checkpoint, "python_tree_sha256": source,
        "normalization": NORMALIZATION, "resize": False, "augmentation": False,
        "invalid_pixels": "zero after normalization (ImageNet mean before normalization)",
        "pooling": "mean of fully supported x_norm_patchtokens, then L2 normalize",
        "patch_support": "all 196 pixels of each 14x14 patch are supported",
        "precision": "float32 inference and embeddings; optional float16 patch-token storage",
        "weights_frozen": True, "tf32": False}
    if encoder_contract is None:
        module = sha256(Path(__file__).parents[1] / "history_features.py")
        loader = sha256(Path(__file__).parents[1] / "methods" / "dinov2.py")
    else:
        if not isinstance(encoder_contract, dict):
            raise ValueError("DINO encoder contract must be a mapping")
        reject_forbidden(encoder_contract)
        for key, expected in required.items():
            if key not in encoder_contract or encoder_contract[key] != expected:
                raise ValueError(f"Unsupported or incomplete DINO encoder contract: {key}")
        module = fingerprint(encoder_contract.get("module_sha256"), "encoder implementation")
        loader = fingerprint(encoder_contract.get("loader_sha256"), "loader implementation")
    return {"schema_version": 1, "model": "dinov2_vits14", "revision": checkpoint,
        "representation": "visual", "dimensions": 384, "source_sha256": source,
        "implementation": {"encoder_sha256": module, "loader_sha256": loader},
        "preprocessing": {"normalization": copy.deepcopy(NORMALIZATION), "resize": False,
            "augmentation": False, "padding": "bottom_right_unsupported_to_14",
            "invalid_pixels": required["invalid_pixels"], "pooling": required["pooling"],
            "patch_support": required["patch_support"], "precision": "float32 inference and embeddings",
            "weights_frozen": True, "tf32": False}}


def _space(space):
    if not isinstance(space, dict):
        raise ValueError("Embedding needs its complete space contract")
    reject_forbidden(space)
    if not all(isinstance(space.get(key), str) and space[key] for key in ("model", "revision")):
        raise ValueError("Embedding space needs model and revision")
    if space.get("representation") not in {"visual", "joint_image_text", "history_baseline"}:
        raise ValueError("Unknown embedding representation; DINO and text vectors cannot be combined")
    if not isinstance(space.get("preprocessing"), dict):
        raise ValueError("Embedding space needs explicit preprocessing")
    return copy.deepcopy(space)


def _cache(root, key, compute):
    path = Path(root) / (key + ".json")
    if path.exists():
        saved = read_json(path)
        if saved.get("key") != key or digest(saved.get("result")) != saved.get("result_sha256"):
            raise ValueError("Corrupt derived retrieval cache")
        return {**saved["result"], "cache_key": key, "cache_hit": True}
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"key": key, "result": result, "result_sha256": digest(result)})
    return {**result, "cache_key": key, "cache_hit": False}


def _limits(limit, allowed_splits):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Retrieval limit must be 1..100")
    splits = tuple(sorted(set(allowed_splits)))
    if not splits or not set(splits).issubset({"train", "val"}):
        raise ValueError("Analogue library must explicitly use train/val, never hidden TEST")
    return splits


def history_descriptor(observations, embeddings_by_id, bins=4):
    """Deterministic ordered-feature/time baseline, not a trained dynamics model.

    Same-date alternatives are averaged. Up to `bins` newest dates remain in
    order, left-padded with zero feature blocks plus an occupancy mask. The
    descriptor concatenates ordered unit features, ordered feature differences,
    log1p inter-date gaps in years, and occupancy; no interpolation or future
    support intersection is used. Absolute dates remain metadata, not features.
    """
    if type(bins) is not int or not 2 <= bins <= 32:
        raise ValueError("History bins must be 2..32")
    if any(temporal_value(r.get("date")) is None for r in observations):
        return {"status": "undated_history", "method": "ordered_feature_time_v1"}
    rows = sorted(observations, key=lambda r: (temporal_value(r["date"]), r["observation_id"]))
    if not rows:
        return {"status": "no_comparable_history", "method": "ordered_feature_time_v1"}
    groups, space = {}, None
    for row in rows:
        moment = temporal_value(row.get("date"))
        embedding = embeddings_by_id.get(row["observation_id"])
        if moment is None or embedding is None:
            return {"status": "embedding_unavailable", "method": "ordered_feature_time_v1"}
        contract = _space(embedding["space"])
        if space is not None and contract != space:
            raise ValueError("Historical embeddings belong to different spaces")
        space = contract
        groups.setdefault(moment, []).append(np.asarray(_vector(embedding["vector"])))
    dates = sorted(groups)[-bins:]
    vectors = [np.asarray(_vector(np.mean(groups[d], axis=0))) for d in dates]
    width = vectors[0].size
    ordered = np.zeros((bins, width), dtype=np.float64)
    occupied = np.zeros(bins, dtype=np.float64)
    start = bins-len(dates)
    ordered[start:] = np.stack(vectors)
    occupied[start:] = 1
    changes = np.zeros_like(ordered)
    gaps = np.zeros(bins, dtype=np.float64)
    for offset in range(1, len(dates)):
        index = start + offset
        changes[index] = vectors[offset] - vectors[offset-1]
        gaps[index] = math.log1p((dates[offset]-dates[offset-1])/(86400*365.2425))
    descriptor = np.concatenate((ordered.ravel(), changes.ravel(), gaps, occupied))
    return {"status": "ok", "method": "ordered_feature_time_v1", "trained": False,
        "vector": _vector(descriptor), "space": {"model": "ordered_feature_time_v1", "revision": "1",
            "representation": "history_baseline", "preprocessing": {"bins": bins,
                "source_space": space, "selection": "latest distinct dates; chronological left padding",
                "gaps": "log1p years", "difference": "ordered adjacent unit feature difference"}},
        "observation_ids": [r["observation_id"] for r in rows],
        "observation_dates": [r["date"] for r in rows],
        "dates": [next(r["date"] for r in rows if temporal_value(r["date"]) == d) for d in dates],
        "delta_days": [0.] + [(b-a)/86400 for a, b in zip(dates, dates[1:])]}


class SearchIndex:
    def __init__(self, store):
        self.store = store
        self.cache_root = store.root / "retrieval_cache"

    def add_embedding(self, observation_id, vector, space):
        row = self.store.verify_observation(self.store.get_observation(observation_id))
        contract = _space(space)
        vector = _vector(vector)
        signature = {"observation_id": observation_id, "observation_sha256": row["record_sha256"],
            "input_sha256": row["crop_sha256"], "support_sha256": row["support_sha256"],
            "geometry_sha256": digest(row["geometry"]), "space": contract}
        identifier = "embedding-" + digest(signature)[:32]
        return self.store._save("embeddings", identifier, {"schema_version": 1,
            "embedding_id": identifier, "observation_id": observation_id,
            "signature": signature, "space": contract, "vector": vector,
            "dimensions": len(vector)})

    def list_embeddings(self, *, observation_id=None):
        records = self.store._list("embeddings")
        if observation_id is not None:
            # Select metadata before reading any RGB/support artifacts. A
            # query must not depend on excluded or future observation files.
            records = [row for row in records if row["observation_id"] == observation_id]
        for record in records:
            self._checked_embedding(record)
        return records

    def _checked_embedding(self, record):
        observation = self.store.verify_observation(self.store.get_observation(record["observation_id"]))
        if record["signature"]["observation_sha256"] != observation["record_sha256"]:
            raise ValueError("Embedding observation provenance changed")
        return record

    def get_embedding(self, observation_id, space=None):
        matches = [r for r in self.store._list("embeddings") if r["observation_id"] == observation_id
                   and (space is None or r["space"] == space)]
        if len(matches) > 1:
            raise ValueError("Choose an embedding space; this observation has multiple encoders")
        return self._checked_embedding(matches[0]) if matches else None

    @property
    def corpus_version(self):
        return digest({"observations": sorted((r["observation_id"], r["record_sha256"])
            for r in self.store.list_observations()), "embeddings": sorted((r["embedding_id"], r["record_sha256"])
            for r in self.list_embeddings()), "implementation_sha256": sha256(__file__)})

    def _eligible(self, cutoff, allowed_splits, exclude_buildings, exclude_observation_ids, exclude_hashes,
                  *, candidate_ids=None):
        if temporal_value(cutoff) is None:
            raise ValueError("Retrieval requires a cutoff")
        selected = []
        for row in self.store.list_observations():
            if candidate_ids is not None and row["observation_id"] not in candidate_ids:
                continue
            if (row["split"] not in allowed_splits or row["building_id"] in exclude_buildings
                    or row["observation_id"] in exclude_observation_ids
                    or row["original_sha256"] in exclude_hashes or row["crop_sha256"] in exclude_hashes
                    or not causally_available(row, cutoff)
                    or row["correspondence_status"] not in REGISTERED):
                continue
            selected.append(self.store.verify_observation(row))
        return selected

    def _query_version(self, rows, embeddings=()):
        permitted = {row["observation_id"] for row in rows}
        return digest({"observations": sorted((r["observation_id"], r["record_sha256"]) for r in rows),
            "embeddings": sorted((r["embedding_id"], r["record_sha256"]) for r in embeddings
                                 if r["observation_id"] in permitted),
            "implementation_sha256": sha256(__file__)})

    def similar(self, vector, space, cutoff, allowed_splits=("train",), exclude_buildings=(),
                exclude_observation_ids=(), exclude_hashes=(), limit=5):
        contract, query = _space(space), _vector(vector)
        splits = _limits(limit, allowed_splits)
        filters = {"cutoff": str(cutoff), "allowed_splits": splits,
            "exclude_buildings": sorted(set(exclude_buildings)),
            "exclude_observation_ids": sorted(set(exclude_observation_ids)),
            "exclude_hashes": sorted(set(exclude_hashes)), "limit": limit}
        # A visual candidate needs a matching representation as well as the
        # causal/split filters. Select both from metadata before touching RGB.
        candidates = [r for r in self.store._list("embeddings") if r["space"] == contract]
        eligible = self._eligible(cutoff, splits, exclude_buildings, exclude_observation_ids, exclude_hashes,
                                  candidate_ids={r["observation_id"] for r in candidates})
        eligible_ids = {row["observation_id"] for row in eligible}
        embeddings = {r["observation_id"]: self._checked_embedding(r) for r in candidates
                      if r["observation_id"] in eligible_ids}
        version = self._query_version(eligible, embeddings.values())
        key = digest({"op": "visual", "vector": query, "space": contract, "filters": filters, "corpus": version})
        def compute():
            matches, seen = [], set()
            for row in eligible:
                entry = embeddings.get(row["observation_id"])
                if entry is None:
                    continue
                if len(entry["vector"]) != len(query):
                    raise ValueError("Dimensions differ inside the same declared embedding space")
                score = float(np.dot(query, entry["vector"]))
                matches.append({"score": score, "score_kind": "cosine_similarity_not_probability",
                    "observation_id": row["observation_id"], "region_id": row["region_id"],
                    "observation": row, "embedding_id": entry["embedding_id"]})
            matches.sort(key=lambda r: (-r["score"], r["observation"]["observation_id"]))
            unique = []
            for item in matches:
                identity = item["observation"]["original_sha256"]
                if identity not in seen:
                    unique.append(item)
                    seen.add(identity)
            return {"status": "ok" if unique else "no_matches", "matches": unique[:limit],
                    "space": contract, "filters": filters, "corpus_version": version}
        return _cache(self.cache_root, key, compute)

    def text(self, query, cutoff, allowed_splits=("train",), exclude_buildings=(),
             exclude_observation_ids=(), exclude_hashes=(), limit=5):
        splits = _limits(limit, allowed_splits)
        terms = set(_tokens(query))
        if not terms:
            raise ValueError("Search text must contain a word")
        filters = {"cutoff": str(cutoff), "allowed_splits": splits,
            "exclude_buildings": sorted(set(exclude_buildings)),
            "exclude_observation_ids": sorted(set(exclude_observation_ids)),
            "exclude_hashes": sorted(set(exclude_hashes)), "limit": limit}
        rows = self._eligible(cutoff, splits, exclude_buildings, exclude_observation_ids, exclude_hashes)
        version = self._query_version(rows)
        key = digest({"op": "lexical", "query": sorted(terms), "filters": filters, "corpus": version})
        def compute():
            matches, seen = [], set()
            for row in rows:
                words = set(_tokens(row.get("description", "")))
                common = terms & words
                if common:
                    matches.append({"score": len(common)/len(terms | words), "score_kind": "lexical_jaccard",
                                    "observation_id": row["observation_id"], "region_id": row["region_id"],
                                    "observation": row})
            matches.sort(key=lambda r: (-r["score"], r["observation"]["observation_id"]))
            unique = []
            for item in matches:
                identity = item["observation"]["original_sha256"]
                if identity not in seen:
                    unique.append(item)
                    seen.add(identity)
            return {"status": "ok" if unique else "no_matches", "matches": unique[:limit],
                    "filters": filters, "corpus_version": version}
        return _cache(self.cache_root, key, compute)

    def similar_history(self, region_id, cutoff, allowed_splits=("train",), exclude_buildings=(),
                        exclude_observation_ids=(), exclude_hashes=(), limit=5, bins=4, space=None):
        splits = _limits(limit, allowed_splits)
        target = self.store.get_region(region_id)
        target_rows = self.store.history(region_id, cutoff, exclude_observation_ids)["observations"]
        embeddings = {r["observation_id"]: self.get_embedding(r["observation_id"], space) for r in target_rows}
        query = history_descriptor(target_rows, embeddings, bins)
        if query["status"] != "ok":
            return {"status": query["status"], "matches": [],
                    "corpus_version": self._query_version(target_rows, self.store._list("embeddings")),
                    "cache_hit": False, "cache_key": None}
        exclusions = set(exclude_buildings) | {target["building_id"]}
        filters = {"cutoff": str(cutoff), "allowed_splits": splits, "exclude_buildings": sorted(exclusions),
                   "exclude_observation_ids": sorted(set(exclude_observation_ids)),
                   "exclude_hashes": sorted(set(exclude_hashes)), "limit": limit, "bins": bins}
        eligible = self._eligible(cutoff, splits, exclusions, exclude_observation_ids, exclude_hashes)
        grouped = {}
        for row in eligible:
            grouped.setdefault(row["region_id"], []).append(row)
        version = self._query_version(eligible, self.store._list("embeddings"))
        key = digest({"op": "history", "query": query, "filters": filters, "corpus": version})
        def compute():
            matches = []
            for identifier, rows in grouped.items():
                entries = {r["observation_id"]: self.get_embedding(r["observation_id"], space) for r in rows}
                value = history_descriptor(rows, entries, bins)
                if value["status"] != "ok" or value["space"] != query["space"]:
                    continue
                matches.append({"region_id": identifier, "score": float(np.dot(query["vector"], value["vector"])),
                    "score_kind": "ordered_feature_time_cosine_not_probability", "history": rows,
                    "observation_ids": [row["observation_id"] for row in rows],
                    "descriptor": {k: v for k, v in value.items() if k != "vector"}})
            matches.sort(key=lambda r: (-r["score"], r["region_id"]))
            return {"status": "ok" if matches else "no_matches", "matches": matches[:limit],
                    "filters": filters, "corpus_version": version,
                    "method": "ordered_feature_time_v1", "trained": False}
        return _cache(self.cache_root, key, compute)

    def import_history_features(self, features_root):
        """Reuse actual verified DINO caches from history_features, not inferred vectors."""
        root = Path(features_root).expanduser().resolve()
        index = read_json(root / "index.json")
        if index.get("kind") != "frozen_history_features":
            raise ValueError("Expected the existing frozen_history_features index")
        from ..history_features import _inside
        report = _inside(root, index["report"])
        run = read_json(report / "run.json")
        if (not run.get("status", "").startswith("completed")
                or sha256(report / "index.json") != sha256(root / "index.json")
                or run.get("index_sha256") != sha256(root / "index.json")):
            raise ValueError("Feature index is not bound to a completed original extraction run")
        encoder = index["encoder"]
        space = dino_embedding_space(encoder["checkpoint_sha256"], encoder["python_tree_sha256"],
                                     encoder_contract=encoder)
        from ..history_features import _feature_cache
        count = 0
        for observation_id, record in index["features"].items():
            observation = self.store.get_observation(observation_id)
            shape = observation["geometry"].get("crop_shape")
            if not shape or any(size % 14 for size in shape):
                raise ValueError("Existing DINO history crops must use a native grid divisible by 14")
            if (record["signature"].get("rgb_sha256") != observation["crop_sha256"]
                    or record["signature"].get("support_sha256") != observation["support_sha256"]
                    or record["signature"].get("encoder_sha256") != digest(encoder)):
                raise ValueError("Feature signature does not belong to this observation/encoder")
            cached = _feature_cache(root, record["signature"])
            if cached != record:
                raise ValueError("Feature index disagrees with the original cache record")
            with np.load(root / cached["path"], allow_pickle=False) as arrays:
                embedded = self.add_embedding(observation_id, arrays["z"], space)
            self.store.add_evidence({"kind": "measurement", "region_id": observation["region_id"],
                "observation_ids": [observation_id], "available_at": observation["available_at"],
                "model": space["model"], "revision": space["revision"],
                "summary": {"embedding_id": embedded["embedding_id"], "representation": "frozen_visual_features"},
                "artifacts": [{"path": str(root / cached["path"]), "sha256": cached["sha256"],
                               "role": "visual_embedding", "media_type": "application/x-npz"}],
                "provenance": {"source_index_sha256": sha256(root / "index.json"),
                               "encoder_sha256": digest(encoder), "encoder_contract": encoder,
                               "feature_id": cached["feature_id"]}})
            count += 1
        return {"status": "ok", "count": count, "space": space, "source_index_sha256": sha256(root / "index.json")}


class KnowledgeBase:
    """Sourced local concepts, relations and user-supplied text fragments.

    No default textbook quotations or degradation causes are invented. Project
    taxonomy is imported with source_kind=project_taxonomy, explicitly separate
    from scientific assertions supplied by the user.
    """
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.documents = self.root / "documents"
        self.documents.mkdir(parents=True, exist_ok=True)

    def _documents(self):
        documents = []
        for path in sorted(self.documents.glob("*.json")):
            record = read_json(path)
            if record.get("payload_sha256") != digest(record.get("document")):
                raise ValueError("Knowledge document changed or is corrupt")
            documents.append(record["document"])
        return documents

    @property
    def corpus_version(self):
        return digest({"documents": self._documents(), "implementation_sha256": sha256(__file__)})

    def import_document(self, document):
        _allow(document, {"document_id", "source", "available_at", "concepts", "relations", "fragments"}, "knowledge document")
        value = copy.deepcopy(document)
        source = value.get("source")
        if not isinstance(source, dict) or not source.get("title") or not source.get("kind") or not source.get("locator"):
            raise ValueError("Knowledge source requires title, kind and locator")
        if source["kind"] not in {"project_taxonomy", "textbook", "publication", "user_document"}:
            raise ValueError("Unknown knowledge source kind")
        if temporal_value(value.get("available_at")) is None:
            raise ValueError("Knowledge needs an explicit availability date")
        value["available_at"] = str(value["available_at"])
        concepts = set()
        for concept in value.get("concepts", []):
            _allow(concept, {"concept_id", "name", "synonyms", "description"}, "concept")
            _id(concept["concept_id"], "concept_id")
            if not concept.get("name") or concept["concept_id"] in concepts:
                raise ValueError("Duplicate or unnamed concept")
            concepts.add(concept["concept_id"])
        existing = {c["concept_id"] for d in self._documents() for c in d.get("concepts", [])}
        for relation in value.get("relations", []):
            _allow(relation, {"subject", "predicate", "object"}, "relation")
            if not relation.get("predicate") or any(relation.get(k) not in concepts | existing for k in ("subject", "object")):
                raise ValueError("Relation references an unknown concept")
        fragments = set()
        for fragment in value.get("fragments", []):
            _allow(fragment, {"fragment_id", "text", "concept_ids", "locator"}, "fragment")
            _id(fragment["fragment_id"], "fragment_id")
            if not fragment.get("text") or fragment["fragment_id"] in fragments:
                raise ValueError("Duplicate or empty knowledge fragment")
            if not set(fragment.get("concept_ids", [])).issubset(concepts | existing):
                raise ValueError("Fragment references an unknown concept")
            fragments.add(fragment["fragment_id"])
        identifier = value.get("document_id") or "document-" + digest(value)[:32]
        _id(identifier, "document_id")
        value["document_id"] = identifier
        path = self.documents / (identifier + ".json")
        record = {"document": value, "payload_sha256": digest(value)}
        if path.exists() and read_json(path) != record:
            raise ValueError("Knowledge documents are immutable; use a new document ID for a revision")
        if not path.exists():
            write_json(path, record)
        return {"status": "ok", "document_id": identifier, "corpus_version": self.corpus_version}

    def seed_taxonomy(self, path, available_at):
        path, fingerprint = checked_artifact(path)
        taxonomy = read_json(path)
        groups = taxonomy.get("groups")
        if not isinstance(groups, list):
            raise ValueError("Expected the existing project semantic class groups")
        concepts, relations, fragments = {}, [], []
        for group in groups:
            name = group["name"]
            concepts[name] = {"concept_id": name, "name": name, "description": "Project reporting group"}
            for category in group["categories"]:
                concepts[category] = {"concept_id": category, "name": category, "description": "Project semantic category"}
                relations.append({"subject": category, "predicate": "reporting_group", "object": name})
            fragments.append({"fragment_id": "group-" + name,
                "text": name + ": " + ", ".join(group["categories"]),
                "concept_ids": [name, *group["categories"]], "locator": "groups/" + name})
        return self.import_document({"document_id": "project-taxonomy-" + fingerprint[:24],
            "source": {"title": "Project semantic reporting groups", "kind": "project_taxonomy",
                "locator": path, "sha256": fingerprint,
                "scope": "Class grouping only; not a scientific explanation of degradation causes"},
            "available_at": available_at, "concepts": list(concepts.values()),
            "relations": relations, "fragments": fragments})

    def get_concept(self, concept_id):
        _id(concept_id, "concept_id")
        matches = []
        for document in self._documents():
            for concept in document.get("concepts", []):
                if concept["concept_id"] == concept_id:
                    matches.append({**concept, "source": document["source"],
                        "available_at": document["available_at"], "document_id": document["document_id"]})
        if not matches:
            raise KeyError(f"Unknown concept: {concept_id}")
        return {"concept_id": concept_id, "name": matches[0]["name"], "records": matches}

    def related(self, concept_id, cutoff):
        end = temporal_value(cutoff)
        if end is None:
            raise ValueError("Knowledge query requires cutoff")
        relations = []
        documents = [d for d in self._documents() if temporal_value(d["available_at"]) <= end]
        if not any(c["concept_id"] == concept_id for d in documents for c in d.get("concepts", [])):
            raise ValueError("Concept is unavailable at the requested cutoff")
        for document in documents:
            for relation in document.get("relations", []):
                if concept_id in {relation["subject"], relation["object"]}:
                    relations.append({**relation, "source": document["source"],
                                      "document_id": document["document_id"], "available_at": document["available_at"]})
        return {"status": "ok" if relations else "no_matches", "concept_id": concept_id,
                "relations": relations, "corpus_version": digest(documents)}

    def search(self, query, cutoff, limit=5, concept_ids=()):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Knowledge limit must be 1..100")
        end = temporal_value(cutoff)
        if end is None:
            raise ValueError("Knowledge query requires cutoff")
        terms = set(_tokens(query))
        if not terms and not concept_ids:
            raise ValueError("Provide text or concept IDs")
        documents = [d for d in self._documents() if temporal_value(d["available_at"]) <= end]
        permitted_ids = {c["concept_id"] for d in documents for c in d.get("concepts", [])}
        if not set(concept_ids).issubset(permitted_ids):
            raise ValueError("Concept is unavailable at the requested cutoff")
        version = digest({"documents": documents, "implementation_sha256": sha256(__file__)})
        key = digest({"op": "knowledge", "query": sorted(terms), "concept_ids": sorted(set(concept_ids)),
                      "cutoff": str(cutoff), "limit": limit, "corpus": version})
        def compute():
            fragments, concepts = [], {}
            for document in documents:
                selected = set(concept_ids)
                for concept in document.get("concepts", []):
                    words = set(_tokens(" ".join([concept["name"], *concept.get("synonyms", [])])))
                    if concept["concept_id"] in selected or terms & words:
                        selected.add(concept["concept_id"])
                        concepts[concept["concept_id"]] = {**concept, "source": document["source"],
                            "available_at": document["available_at"], "document_id": document["document_id"]}
                for fragment in document.get("fragments", []):
                    words = set(_tokens(fragment["text"]))
                    overlap = terms & words
                    if overlap or selected & set(fragment.get("concept_ids", [])):
                        fragments.append({**fragment, "score": len(overlap)/max(1, len(terms | words)),
                            "source": document["source"], "available_at": document["available_at"],
                            "document_id": document["document_id"]})
            fragments.sort(key=lambda f: (-f["score"], f["document_id"], f["fragment_id"]))
            return {"status": "ok" if fragments or concepts else "no_matches", "fragments": fragments[:limit],
                    "concepts": sorted(concepts.values(), key=lambda c: c["concept_id"])[:limit],
                    "corpus_version": version, "cutoff": str(cutoff)}
        return _cache(self.root / "query_cache", key, compute)
