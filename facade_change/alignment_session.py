"""Read the owner's completed incremental alignment sessions, without inference.

Selection belongs to pairs.json. Quality rankings are advisory and never replace
that selection. The immutable review.csv binds each decision to exact run bytes.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .io import read_json, sha256
from .preparation import validate_partitions


IDENTITY = ("sha256", "building_id", "view_id", "year", "width", "height", "split")


def remember(path, fingerprints, expected=None):
    path = Path(path).expanduser().resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Alignment input hash mismatch (changed): {path}")
    if str(path) in fingerprints and fingerprints[str(path)] != actual:
        raise ValueError(f"Alignment input changed while reading: {path}")
    fingerprints[str(path)] = actual
    return path


def artifact(root, record, name, fingerprints):
    root = Path(root).resolve()
    path = (root / name).resolve()
    expected = record.get("artifact_sha256", {}).get(name)
    if root not in path.parents or not expected:
        raise ValueError(f"Missing or unsafe alignment artifact provenance: {name}")
    return remember(path, fingerprints, expected)


def resolve_session(value):
    root = Path(value).expanduser().resolve()
    if (root / "run.json").is_file():
        return root
    # A cache root is convenient only when unambiguous. Never guess 'latest'.
    base = root / "sessions"
    candidates = sorted(p.parent for p in base.glob("*/run.json"))
    if len(candidates) != 1:
        raise ValueError(f"Specify one completed session under {base}; found {len(candidates)}. "
                         "Use the exact Reports path printed by dataset_statistics.py")
    return candidates[0]


@dataclass
class AlignmentSession:
    path: Path
    manifest_path: Path
    manifest: dict
    pairs: list
    selected: dict
    fingerprints: dict
    quality_report: str | None

    def matches(self):
        found = defaultdict(list)
        for match in self.selected.values():
            observations = match["record"]["observations"]
            key = tuple(observations[role]["sha256"] for role in ("reference", "source"))
            if found[key]:
                raise ValueError("Ambiguous duplicate content pair in alignment session")
            found[key].append(match)
        attempts = [{"pair_id": row["pair_id"], "status": row["selection_status"],
                     "method": row["selected_method"], "session": str(self.path)} for row in self.pairs]
        return found, attempts


def load_session(value, fingerprints=None):
    """Validate immutable metadata and selected geometry; do not decode RGB."""
    hashes = {} if fingerprints is None else fingerprints
    for name in ("alignment_session.py", "saved_geometry.py", "geoscd.py"):
        remember(Path(__file__).with_name(name), hashes)
    root = resolve_session(value)
    record = read_json(remember(root / "run.json", hashes))
    if (record.get("kind") != "incremental_alignment_preparation"
            or record.get("status") not in {"completed_needs_review", "completed_with_issues", "completed"}
            or record.get("config", {}).get("plan_only")):
        raise ValueError("Alignment session is not completed; wait for preparation and quality evaluation to finish")
    documents = {name: read_json(artifact(root, record, name + ".json", hashes))
                 for name in ("inputs", "pairs", "summary")}
    inputs, pairs, summary = (documents[name] for name in ("inputs", "pairs", "summary"))
    manifest_path = remember(inputs["manifest"], hashes, inputs["manifest_sha256"])
    if record["config"].get("manifest_sha256") != inputs["manifest_sha256"]:
        raise ValueError("Session manifest fingerprints disagree")
    manifest = read_json(manifest_path)
    images = {}
    for raw in manifest["images"]:
        row = dict(raw)
        path = Path(row["image_path"]).expanduser()
        row["image_path"] = str((manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
        key = str(row["image_id"])
        if key in images:
            raise ValueError("Duplicate manifest image ID")
        images[key] = row
    # The temporal subset is reviewed; the full semantic dataset can additionally
    # contain unknown TRAIN images, whose separate leakage audit remains in force.
    current = {}
    for row in inputs["images"]:
        key = str(row["image_id"])
        expected = images.get(key)
        if key in current or expected is None or any(row.get(k) != expected.get(k) for k in IDENTITY):
            raise ValueError("Session observation identity differs from its manifest")
        if expected.get("metadata_status") != "reviewed":
            raise ValueError("Alignment session requires reviewed temporal metadata")
        current[key] = expected
    validate_partitions([row for row in images.values() if row.get("metadata_status") == "reviewed"
                         and row.get("building_id") and row.get("view_id")])
    validate_partitions(list(current.values()))
    review_path = artifact(root, record, "review.csv", hashes)
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        reviews = {}
        for row in csv.DictReader(stream):
            key = (row["pair_id"], row["method"])
            if key in reviews:
                raise ValueError("Duplicate alignment review row")
            reviews[key] = row
    selected, seen = {}, set()
    prepared = {(str(p["reference_id"]), str(p["source_id"])) for p in manifest.get("pairs", [])}
    for pair in pairs:
        pid = pair["pair_id"]
        if pid in seen:
            raise ValueError("Duplicate alignment session pair")
        seen.add(pid)
        key = (str(pair["reference_id"]), str(pair["source_id"]))
        if key not in prepared or pid != "-".join(key):
            raise ValueError("Session pair is absent from its prepared manifest")
        reference, source = (current[k] for k in key)
        if (reference["sha256"] == source["sha256"] or reference["year"] >= source["year"]
                or any(reference.get(k) != source.get(k) for k in ("building_id", "view_id", "split"))
                or reference["split"] not in {"train", "val", "test"}):
            raise ValueError("Session pair violates chronology, identity or reviewed splits")
        for name in ("building_id", "view_id", "split"):
            if pair.get(name) != reference[name]:
                raise ValueError(f"Session pair {name} differs from the manifest")
        if pair["reference_year"] != reference["year"] or pair["source_year"] != source["year"]:
            raise ValueError("Session pair years differ from the manifest")
        method = pair["selected_method"]
        if method is None:
            if pair.get("selection_status") != "unresolved":
                raise ValueError("Unselected session pair has a selection verdict")
            continue
        if method not in {"sift", "loftr", "geoscd"}:
            raise ValueError(f"Unsupported session alignment method: {method}")
        entry = pair["methods"][method]
        review = reviews.get((pid, method), {})
        if (review.get("reference_sha256") != reference["sha256"]
                or review.get("source_sha256") != source["sha256"]
                or review.get("decision", "") != entry.get("review", "")
                or review.get("state") != entry["state"] or not review.get("alignment_run_sha256")):
            raise ValueError("Session selection lacks its hash-bound review record")
        path = Path(entry["path"]).expanduser().resolve()
        original_record = read_json(remember(path / "run.json", hashes, review["alignment_run_sha256"]))
        status = str(original_record.get("status", ""))
        if not status.startswith("completed"):
            raise ValueError("Selected pair run is not completed")
        if method == "geoscd":
            if original_record.get("kind") != "geoscd_geometry_pair":
                raise ValueError("Selected GeoSCD pair has the wrong run kind")
            state = "computed_needs_review"
        else:
            if (original_record.get("kind") != "pair_alignment"
                    or original_record.get("config", {}).get("method") != method):
                raise ValueError("Selected homography method differs from the pair run")
            passed = original_record.get("quality_gate", {}).get("passed")
            state = "passed" if passed is True else "rejected" if passed is False else "invalid"
        decision = entry.get("review", "")
        expected_status = "reviewed" if decision == "accept" else "provisional_gate_pass"
        if (state != entry["state"] or state == "invalid" or decision == "reject"
                or (decision != "accept" and state != "passed")
                or pair["selection_status"] != expected_status):
            raise ValueError("Selected alignment is neither explicitly reviewed nor gate-passed")
        for role, expected in (("reference", reference), ("source", source)):
            actual = original_record.get("observations", {}).get(role, {})
            if any(actual.get(k) != expected.get(k) for k in IDENTITY):
                raise ValueError(f"Selected {role} identity differs from session")
        artifact(path, original_record, "geometry.json", hashes)
        if method == "geoscd":
            artifact(path, original_record, "correspondences.npz", hashes)
        # Cache reuse is content based: old COCO IDs/paths may differ. Preserve
        # the original run bytes, use verified current observations downstream.
        selection = {"session": str(root), "pair_id": pid, "method": method,
                     "selection_status": pair["selection_status"], "review": decision,
                     "alignment_run_sha256": review["alignment_run_sha256"],
                     "observations": {"reference": reference, "source": source}}
        selected[pid] = {"path": path, "method": method, "selection": selection,
                         "record": {**original_record, "observations": selection["observations"]}}
    quality = summary.get("quality_report")
    if quality:
        qroot = Path(quality).expanduser().resolve()
        qrecord = read_json(remember(qroot / "run.json", hashes, summary["quality_run_sha256"]))
        if not str(qrecord.get("status", "")).startswith("completed"):
            raise ValueError("Session quality report is incomplete")
        for name in ("pairs.json", "inputs.json"):
            if qrecord["config"]["session_sha256"].get(name) != hashes[str(root / name)]:
                raise ValueError("Quality report refers to another preparation selection")
        for name in ("protocol.json", "quality_summary.json", "quality_ranking.json",
                     "quality_cycles.json", "quality_drift.json"):
            artifact(qroot, qrecord, name, hashes)
    return AlignmentSession(root, manifest_path, {**manifest, "images": list(images.values())},
                            pairs, selected, hashes, quality)
