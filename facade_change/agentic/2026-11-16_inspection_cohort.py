"""CPU admission and reviewed subsets; no predictions, models or new label editor."""
from collections import Counter
import copy
import importlib
from itertools import zip_longest
from pathlib import Path

import numpy as np

from ..history_features import Fingerprints, _history_document, _inside, digest
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .evaluation import _csv
from .pipeline import _matrix
from .runner import new_run, complete_run
from .store import temporal_value


data = importlib.import_module(".2026-11-10_inspection_data", __package__)
qa = importlib.import_module(".2026-11-10_questions", __package__)
identity = importlib.import_module(".2026-11-16_inspection_identity", __package__)
STAGES = ("public_schema", "task", "compared_observations", "original_split",
          "physical_identity", "region_frame", "period_geometry", "earlier_dates", "files")


def load_admission(protocol, fp):
    if "admission" not in protocol:
        return None  # Original explicit owner-attested protocol remains readable.
    ref = protocol["admission"]
    fp.add(ref["path"], ref["sha256"])
    value = read_json(ref["path"])
    if (set(value) != {"kind", "questions_sha256", "history_manifest_sha256", "identity_source",
                       "confirmation_sha256", "cases"} or value["kind"] != "inspection_admission"
            or any(value[k] != protocol[k] for k in ("questions_sha256", "history_manifest_sha256"))
            or len({r["question_id"] for r in value["cases"]}) != len(value["cases"])):
        raise ValueError("Admission certificate does not bind the frozen cohort/history")
    return value


def check_config(config_path, protocol_path, out, *, derive=False, progress=None):
    from .config import load_config
    supplied = load_config(config_path)
    protocol = data.validate_protocol(read_json(protocol_path))
    journal = new_run(out, dict(config_source=str(Path(config_path).resolve()),
                               config_sha256=sha256(config_path), protocol_sha256=sha256(protocol_path)),
                      "inspection_config_check", progress)
    try:
        resolved = copy.deepcopy(supplied)
        changes = []

        def set_value(target, key, value, prefix=""):
            if target.get(key) != value:
                changes.append(dict(field=prefix+key, before=target.get(key), after=value))
                target[key] = value

        if derive:
            set_value(resolved, "methods", ["anychange"])
            set_value(resolved, "gpu_mode", "sequential")
            set_value(resolved, "worker_timeout", max(supplied["worker_timeout"], data.CAPS["max_seconds"]))
            for key, value in data.CAPS.items():
                set_value(resolved["budgets"], key, value, "budgets.")
            qwen = resolved["models"].setdefault("qwen", {})
            for key, cap in (("max_images", "max_images"), ("max_input_tokens", "max_input_tokens"),
                             ("max_new_tokens", "max_output_tokens")):
                set_value(qwen, key, max(qwen.get(key, 0), data.CAPS[cap]), "models.qwen.")
            write_json(journal.out / "2026-11-16-inspection.local.json", resolved)
        errors = {}
        for name, config in (("source", supplied), ("derived" if derive else "checked", resolved)):
            try:
                data.validate_config(config, protocol)
                errors[name] = None
            except ValueError as exc:
                errors[name] = str(exc)
        report = dict(status="completed" if errors["derived" if derive else "checked"] is None else "blocked",
                      changes=changes, validation=errors, actual_config=resolved,
                      gpu_smoke="not_run", qwen_weights_load="not_verified",
                      memory_fit="16k/8 images on 16GB requires real GPU smoke; numeric validation is insufficient")
        write_json(journal.out / "config-report.json", report)
        complete_run(journal, "completed" if report["status"] == "completed" else "failed", summary=report)
        return report
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def public_document(rows):
    return dict(schema_version=1, kind="facade_qa_questions", protocol=qa.PROTOCOL,
                questions=rows, retrieval_contract={})


def metadata(question, track, root, available_at, identities, tracks, report):
    """Exact joins using metadata only; file verification is a separate last stage."""
    passed = report.setdefault("passed", [])

    def check(stage, condition, reason):
        report["stage"] = stage
        if not condition:
            raise ValueError(reason)
        passed.append(stage)

    report["stage"] = "public_schema"
    qa.validate_public(public_document([question]))
    passed.append("public_schema")
    check("task", question["task"] == "temporal_change", "requires_temporal_change")
    observations = question["observations"]
    check("compared_observations", len(observations) == 2, "requires_exactly_two_compared_observations")
    check("original_split", question["split"] in {"train", "val"}, "TEST_unassigned_forbidden")
    problem = identities.problem(question["building_id"], tracks)
    check("physical_identity", not problem, problem)
    report["stage"] = "region_frame"
    if track is None:
        raise ValueError("Unknown canonical region")
    x, y, width, height = track["bbox_xywh"]
    frame = dict(coordinate_system="region_crop_pixels", region_id=track["track_id"],
                 shape=[height, width], anchor_sha256=track["anchor_sha256"],
                 anchor_bbox_xyxy=[x, y, x+width, y+height])
    region_key = {k: track[k] for k in ("building_id", "view_id", "anchor_sha256", "bbox_xywh")}
    check("region_frame", question["frame"] == frame and track["track_id"] == digest(region_key)[:24]
          and all(question[k] == track[k] for k in ("building_id", "view_id", "split")),
          "canonical_region_frame_or_identity_mismatch")
    report["stage"] = "period_geometry"
    lookup = {r["observation_id"]: r for r in track["observations"]}
    dates = [r["date"] for r in observations]
    if (question["task_mode"] not in {"retrospective", "as_of"}
            or dates != question["task_dates"] or temporal_value(dates[0]) >= temporal_value(dates[1])):
        raise ValueError("Compared chronology/task_dates/task_mode disagree")
    permitted, history, excluded = [], [], []
    current = [r["observation_id"] for r in observations]
    for row in track["observations"]:
        oid = row["observation_id"]
        known = str(row["year"]) if available_at == "observation_date" else available_at
        if oid not in current:
            if temporal_value(row["year"]) >= temporal_value(dates[0]):
                excluded.append(dict(observation_id=oid, reason="not_strictly_before_reference"))
                continue
            if max(temporal_value(known), temporal_value(row["year"])) > temporal_value(question["knowledge_cutoff"]):
                excluded.append(dict(observation_id=oid, reason="unavailable_at_knowledge_cutoff"))
                continue
        matrix = _matrix(row["alignment"]["matrix"])
        transform = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]]) @ matrix
        geometry = dict(source_to_crop=transform.tolist(), crop_to_source=np.linalg.inv(transform).tolist(),
                        crop_shape=[height, width], anchor_sha256=track["anchor_sha256"])
        expected = dict(observation_id=oid, date=str(row["year"]), available_at=known,
                        date_status="confirmed", rgb_path=str(_inside(root, row["rgb"])),
                        rgb_sha256=row["rgb_sha256"], support_path=str(_inside(root, row["support"])),
                        support_sha256=row["support_sha256"], geometry=geometry)
        if oid != digest([track["track_id"], row["image_sha256"]])[:24]:
            raise ValueError("Observation ID differs from canonical region/image identity")
        if oid in current:
            if observations[current.index(oid)] != expected:
                raise ValueError("Public observation SHA/date/availability/geometry mismatch")
        else:
            history.append(oid)
        permitted.append(dict(**expected, original_path=str(Path(row["original_image_path"]).expanduser().resolve()),
                              original_sha256=row["image_sha256"]))
    if not set(current) <= set(lookup) or len(set(current)) != 2:
        raise ValueError("Unknown or duplicate compared observation IDs")
    passed.append("period_geometry")
    history.sort(key=lambda oid: (lookup[oid]["year"], oid), reverse=True)
    report.update(eligible_history_ids=history, eligible_history_count=len(history), excluded=excluded,
                  distinct_history_dates=len({lookup[oid]["year"] for oid in history}))
    check("earlier_dates", report["distinct_history_dates"] >= 2, "requires_two_distinct_earlier_available_dates")
    universe = dict(region_id=question["region_id"], frame=frame, task_dates=dates,
                    knowledge_cutoff=question["knowledge_cutoff"], archive_available_at=available_at,
                    current_ids=current, history_ids=history, observations=permitted)
    return universe


def history_context(histories, fp, identity_map=None, expected=None):
    root = Path(histories).expanduser().resolve()
    root = root.parent if root.name == "manifest.json" else root
    document, manifest_sha = _history_document(root, fp)
    identities = identity.Identities(identity_map, fp, expected)
    hashes, originals = {}, {}
    for track in document["tracks"]:
        for row in track["observations"]:
            for field in ("image_sha256", "rgb_sha256"):
                if hashes.setdefault(row[field], track["split"]) != track["split"]:
                    raise ValueError("Canonical image/crop hash split conflict")
            owner = (track["building_id"], track["view_id"], row["year"])
            if originals.setdefault(row["image_sha256"], owner) != owner:
                raise ValueError("Canonical original image building/view/date conflict")
    return root, document, manifest_sha, identities


def compatibility(rows, root, document, available_at, identities, fp):
    tracks = {t["track_id"]: t for t in document["tracks"]}
    reports, universes = [], {}
    for question in rows:
        qid = question["question_id"]
        row = dict(question_id=qid, building_id=question["building_id"], split=question["split"],
                   physical_building_id=identities.physical(question["building_id"]),
                   status="ineligible", reason=None, passed=[], distinct_history_dates=None)
        try:
            track = tracks.get(question["region_id"])
            universe = metadata(question, track, root, available_at, identities, document["tracks"], row)
            universes[qid] = universe
            row["universe_sha256"] = digest(universe)
            row["stage"] = "files"
            missing = [r[k] for r in universe["observations"]
                       for k in ("original_path", "rgb_path", "support_path") if not Path(r[k]).is_file()]
            # Verify every available file even when another file is absent.
            for obs in universe["observations"]:
                for name in ("original", "rgb", "support"):
                    if Path(obs[name + "_path"]).is_file():
                        fp.add(obs[name + "_path"], obs[name + "_sha256"])
            if missing:
                row.update(status="pending", reason="required_files_missing", missing_paths=missing)
            else:
                lookup = {r["observation_id"]: r for r in track["observations"]}
                for obs in universe["observations"]:
                    checked, _ = data._observation(root, track, lookup[obs["observation_id"]],
                                                   dict(archive_available_at=available_at), fp)
                    height, width = question["frame"]["shape"]
                    data.crop_geometry(checked, [0, 0, width, height],
                                       int(np.prod(checked["geometry"]["native_shape"])))
                row["passed"].append("files")
                row["status"] = "eligible"
        except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        reports.append(row)
    return reports, universes


def inventory(source, histories, out, *, available_at, identity_map=None, ids=None, limit=30, progress=None):
    if available_at != "observation_date" and temporal_value(available_at) is None:
        raise ValueError("Explicit archive availability required")
    fp = Fingerprints()
    fp.add(source)
    supplied = read_json(source)
    if supplied.get("kind") == "facade_qa_candidates":
        # Project public fields immediately; provisional labels never drive selection.
        rows = [{k: copy.deepcopy(r[k]) for k in qa.PUBLIC_KEYS} for r in supplied["candidates"]]
    elif supplied.get("kind") == "facade_qa_questions":
        rows = qa.validate_public(supplied)["questions"]
    else:
        raise ValueError("Expected candidates.json or public questions.json, never predictions/reviews")
    qids = [r["question_id"] for r in rows]
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate input question IDs")
    root, document, history_sha, identities = history_context(histories, fp, identity_map)
    journal = new_run(out, dict(source=str(Path(source).resolve()), histories=str(root),
                               available_at=available_at), "inspection_compatibility", progress)
    try:
        journal.event("inspection_inventory_started", questions=len(rows))
        reports, universes = compatibility(rows, root, document, available_at, identities, fp)
        eligible = {r["question_id"] for r in reports if r["status"] == "eligible"}
        if ids is not None:
            fp.add(ids)
            selected = read_json(ids)
            if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                    or not set(selected) <= eligible):
                raise ValueError("Allowlist must contain distinct eligible known question IDs")
            rule = "explicit_allowlist_in_supplied_order; fixed_before_predictions"
        else:
            if type(limit) is not int or limit < 1:
                raise ValueError("limit must be positive")
            buildings = sorted({r["physical_building_id"] for r in reports if r["question_id"] in eligible})
            groups = [sorted(r["question_id"] for r in reports
                             if r["question_id"] in eligible and r["physical_building_id"] == building)
                      for building in buildings]
            selected = [qid for round_ in zip_longest(*groups) for qid in round_ if qid is not None][:limit]
            rule = "round_robin_sorted_physical_buildings_and_question_ids_first_limit; fixed_before_predictions"
        filters = []
        before = len(rows)
        for stage in STAGES:
            after = sum(stage in r["passed"] for r in reports)
            filters.append(dict(stage=stage, before=before, after=after, removed_or_pending=before-after))
            before = after
        distributions = {field: {status: dict(Counter(str(r[field]) for r in reports if r["status"] == status))
                         for status in ("eligible", "ineligible", "pending")}
                         for field in ("physical_building_id", "split", "distinct_history_dates")}
        report = qa._private("inspection_compatibility", status="completed", rows=reports, filters=filters,
                             distributions=distributions, permitted_rgb=universes,
                             counts=dict(Counter(r["status"] for r in reports)),
                             identity_source=identities.source,
                             touched_alias_groups=[g for g in identity.template()["groups"]
                                 if set(g["building_ids"]) & {r["building_id"] for r in rows}],
                             input_sha256=fp.files, predictions_read=False, models_loaded=False)
        write_json(journal.out / "inventory.private.json", report)
        _csv(journal.out / "compatibility.private.csv", reports)
        selection = dict(schema_version=1, kind="inspection_selection", question_ids=selected, rule=rule, limit=limit,
                         source=dict(path=str(Path(source).resolve()), sha256=sha256(source)),
                         histories=dict(path=str(root), sha256=history_sha), available_at=available_at,
                         identity_source=identities.source,
                         inventory=dict(path=str(journal.out / "inventory.private.json"),
                                        sha256=sha256(journal.out / "inventory.private.json")))
        write_json(journal.out / "selection.json", selection)
        write_json(journal.out / "assessability-template.private.json", qa._private(
            "inspection_assessability_review", selection_sha256=sha256(journal.out / "selection.json"),
            cases=[dict(question_id=qid, universe_sha256=digest(universes[qid]), decision="pending",
                        review_source_sha256=None, review_revision=None, reviewer="", rationale="",
                        basis=None, sources=[]) for qid in selected]))
        sections, copied = [], {}
        (journal.out / "assets").mkdir()
        for qid in selected:
            universe = universes[qid]
            panels = [{"src": qa._copy(journal.out, r["rgb_path"], journal.out, copied),
                       "label": r["date"] + " / " + r["observation_id"] +
                       (" compared" if r["observation_id"] in universe["current_ids"] else " permitted earlier history")}
                      for r in universe["observations"]]
            sections.append(dict(title=qid, panels=panels,
                                 caption="Exact original RGB paths, hashes and geometry: inventory.private.json; universe SHA " + digest(universe)))
        write_static_gallery(journal.out, "PRIVATE — permitted CVPR RGB history", sections, notes=[
            "Assessability/evidence boundaries only: preserve existing approved answer and boxes.",
            "Pending means no attestation. Incompatible WWW labels remain unchanged; defer/exclude or obtain separate CVPR review."])
        fp.verify()
        summary = dict(status="completed", counts=report["counts"], selected=len(selected),
                       selected_physical_buildings=len({identities.physical(r["building_id"]) for r in rows
                                                        if r["question_id"] in selected}),
                       target="20–30 desired, not guaranteed; approval/assessability remain separate",
                       report=str(journal.out / "inventory.private.json"))
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()


def prepare(review, selection, assessment_review, out, *, smoke_ids=None, progress=None):
    """Preserve approved rows verbatim; new confirmation concerns evidence scope only."""
    fp = Fingerprints()
    for path in (review, selection, assessment_review):
        fp.add(path)
    chosen = read_json(selection)
    if chosen.get("kind") != "inspection_selection":
        raise ValueError("Expected a frozen pre-prediction inspection selection")
    selection_record = Path(selection).resolve().parent / "run.json"
    fp.add(selection_record)
    record = read_json(selection_record)
    if (record.get("kind") != "inspection_compatibility" or record.get("status") != "completed"
            or record.get("artifact_sha256", {}).get(Path(selection).name) != sha256(selection)):
        raise ValueError("Selection differs from its completed pre-prediction inventory")
    for name in ("source", "inventory"):
        fp.add(chosen[name]["path"], chosen[name]["sha256"])
    original_report = read_json(chosen["inventory"]["path"])
    if chosen["identity_source"] != original_report["identity_source"]:
        raise ValueError("Identity mapping changed after inventory")
    reviewed = read_json(review)
    fp.add(reviewed["candidates_path"], reviewed["candidate_sha256"])
    candidates = read_json(reviewed["candidates_path"])["candidates"]
    lookup = {r["question_id"]: r for r in candidates}
    ids = chosen["question_ids"]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(lookup):
        raise ValueError("Unknown, duplicate or empty selection")
    source = read_json(chosen["source"]["path"])
    source_rows = {r["question_id"]: r for r in source.get("candidates", source.get("questions", []))}
    questions = []
    for qid in ids:
        state = reviewed["states"].get(qid, {})
        if state.get("status") != "approved":
            raise ValueError("Selected review is not approved: " + qid)
        row = {**{k: lookup[qid][k] for k in qa.PUBLIC_KEYS},
               "question": state.get("reviewed_question", lookup[qid]["question"])}
        # A later wording-only human review is allowed for candidate selections.
        keys = qa.PUBLIC_KEYS - ({"question"} if source["kind"] == "facade_qa_candidates" else set())
        if any(row[k] != source_rows[qid][k] for k in keys):
            raise ValueError("Selected public region/period/geometry/wording changed")
        questions.append(row)
    identity_source = chosen["identity_source"]
    root, document, history_sha, identities = history_context(chosen["histories"]["path"], fp,
        identity_source["path"] if identity_source else None, identity_source["sha256"] if identity_source else None)
    if history_sha != chosen["histories"]["sha256"]:
        raise ValueError("History SHA changed after selection")
    reports, universes = compatibility(questions, root, document, chosen["available_at"], identities, fp)
    if any(r["status"] != "eligible" for r in reports):
        raise ValueError("Selected cohort no longer eligible: " + str(reports))
    if any(universes[qid] != original_report["permitted_rgb"][qid] for qid in ids):
        raise ValueError("Permitted RGB universe changed after selection")
    confirmation = read_json(assessment_review)
    if (confirmation.get("kind") != "inspection_assessability_review"
            or confirmation.get("selection_sha256") != sha256(selection)):
        raise ValueError("Assessability review must bind this exact selection")
    checks = {r["question_id"]: r for r in confirmation["cases"]}
    if len(checks) != len(confirmation["cases"]) or set(checks) != set(ids):
        raise ValueError("Assessability decisions must cover the selection exactly once")
    for qid in ids:
        row = checks[qid]
        if (row.get("decision") != "compatible" or row.get("universe_sha256") != digest(universes[qid])
                or row.get("review_source_sha256") != sha256(review)
                or row.get("review_revision") != reviewed["states"][qid]["revision"]
                or not row.get("reviewer", "").strip() or not row.get("rationale", "").strip()
                or row.get("basis") not in {"existing_documented_confirmation", "narrow_human_scope_review"}
                or not row.get("sources")):
            raise ValueError("Assessability/evidence-boundary confirmation pending or incompatible: " + qid)
        for ref in row["sources"]:
            fp.add(ref["path"], ref["sha256"])
    smoke = ids[:2] if smoke_ids is None else read_json(smoke_ids)
    if smoke_ids:
        fp.add(smoke_ids)
    if not isinstance(smoke, list) or len(smoke) != 2 or len(set(smoke)) != 2 or not set(smoke) <= set(ids):
        raise ValueError("Smoke requires exactly two distinct IDs from the same full cohort")
    if len({q["task_mode"] for q in questions}) != 1:
        raise ValueError("A frozen cohort needs one task_mode")
    journal = new_run(out, dict(review=str(Path(review).resolve()), selection=str(Path(selection).resolve())),
                      "inspection_prepared_cohort", progress)
    try:
        evaluator = importlib.import_module(".2026-11-10_qa_evaluation", __package__)
        manifests = {}
        for name, subset in (("full", ids), ("smoke", smoke)):
            folder = journal.out / name
            exported = folder / "export"
            qa.export(review, exported, question_ids=subset)
            public, labels = evaluator.approved_cohort(exported / "questions.json", exported / "answers.private.json", fp)
            owned = {r["question_id"]: {**r, "physical_building_id": identities.physical(r["building_id"])}
                     for r in public["questions"]}
            evaluator._events(owned, labels)  # Deduplicate/validate physical event ownership before freezing.
            admission = dict(kind="inspection_admission", questions_sha256=sha256(exported / "questions.json"),
                history_manifest_sha256=history_sha, identity_source=identity_source,
                confirmation_sha256=sha256(assessment_review),
                cases=[dict(question_id=qid, universe_sha256=digest(universes[qid])) for qid in subset])
            write_json(folder / "admission.json", admission)
            protocol = data.protocol_document(history_sha, admission["questions_sha256"], available_at=chosen["available_at"])
            protocol["task_mode"] = questions[0]["task_mode"]
            protocol["admission"] = dict(path=str(folder / "admission.json"), sha256=sha256(folder / "admission.json"))
            if name == "smoke":
                protocol["policies"] = ["heuristic", "agent"]
            write_json(folder / "protocol.json", data.validate_protocol(protocol))
            manifests[name] = dict(questions_sha256=admission["questions_sha256"],
                                   answers_sha256=sha256(exported / "answers.private.json"),
                                   protocol_sha256=sha256(folder / "protocol.json"), question_ids=subset)
        write_json(journal.out / "selection.private.json", qa._private("inspection_cohort_provenance",
                   inputs=fp.files, selection=chosen, manifests=manifests,
                   smoke_rule="first_two_in_frozen_order" if smoke_ids is None else "explicit_full_cohort_subset"))
        fp.verify()
        summary = dict(status="completed", full=len(ids), smoke=2, manifests=manifests)
        write_json(journal.out / "summary.json", summary)
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
