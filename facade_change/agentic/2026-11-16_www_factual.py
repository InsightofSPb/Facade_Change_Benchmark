"""Blind human review of saved WWW outputs, including unscored legacy runs.

This exporter never creates references, a retrospective plan or accuracy scores.
Evidence is reconstructed from saved Qwen requests, not inferred from retrieval.
"""
from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path
import secrets

from ..history_features import Fingerprints, digest
from ..io import read_json, sha256, write_json
from ..visualization import write_static_gallery
from .report import _copy
from .runner import complete_run, new_run
from .store import reject_forbidden

qa = importlib.import_module(".2026-11-10_questions", __package__)
runner = importlib.import_module(".2026-11-15_www_runner", __package__)
curation = importlib.import_module(".2026-11-15_www_curation", __package__)
JUDGMENTS = {"target_event_match": ["", "matches", "different_event", "unresolved", "not_applicable"],
             "claim_support": ["", "supported", "contradicted", "unresolved"],
             "citation_support": ["", "supports", "contradicts", "unresolved", "no_citations"]}


def read_saved(path, fingerprints, root, hashes, *, public_evidence=True):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Saved review artifact escapes the inference run")
    fingerprints.add(path, hashes.get(path.relative_to(root).as_posix()))
    value = read_json(path)
    if public_evidence:
        reject_forbidden(value, private_only=True)
    return value


def blind_case(question, result, exposed, folder, out, fingerprints, blind_id, *, root, hashes=None, target=None):
    hashes = hashes or {}
    metadata, requested_images, issues, raw_responses, calls = {}, {}, [], [], []
    for path in sorted((folder / "calls").glob("*-qwen.json")):
        call = read_saved(path, fingerprints, root, hashes)
        calls.append({"path": str(path), "sha256": sha256(path)})
        response = call.get("response", {})
        if isinstance(response.get("text"), str):
            raw_responses.append(response["text"])
        for message in call.get("request", {}).get("messages", []):
            if message.get("role") != "user":
                continue
            for block in message.get("content", []):
                if block.get("type") == "image" and block.get("evidence_id"):
                    requested_images.setdefault(block["evidence_id"], set()).add(block["path"])
                elif block.get("type") == "text":
                    try:
                        value = json.loads(block["text"])
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, dict) and isinstance(value.get("evidence_id"), str):
                        metadata[value["evidence_id"]] = value
    if not calls:
        issues.append("no_saved_qwen_requests; exposure log alone does not prove presentation")
    answer = result.get("answer") or {}
    citations = answer.get("citations", [])
    evidence, panels, copy_sources = [], [], {}
    for identifier in sorted(set(metadata) | set(requested_images)):
        record = exposed.get(identifier, {})
        source = {"evidence_id": identifier, "cited": identifier in citations,
                  "shown_metadata": metadata.get(identifier), "image": None,
                  "presentation_basis": "submitted_in_saved_qwen_request; worker processing is not inferred"}
        paths = requested_images.get(identifier, set())
        if len(paths) > 1:
            issues.append(identifier + ": multiple requested image paths; no implicit coordinate transfer")
        elif paths:
            path = Path(next(iter(paths))).resolve()
            try:
                if not record.get("image_sha256") or path != Path(record.get("image_path", "")).resolve():
                    raise ValueError("requested image is not bound to exposure image SHA/path")
                fingerprints.add(path, record["image_sha256"])
                relative = _copy(out, path, out, copy_sources)
                source["image"] = {"src": relative, "sha256": record["image_sha256"],
                                   "shape": record.get("shape", record.get("frame", {}).get("shape"))}
                panels.append({"src": relative, "label": identifier + (" / cited" if identifier in citations else " / shown, not cited")})
            except (OSError, ValueError) as exc:
                issues.append(identifier + ": " + str(exc))
        evidence.append(source)
    for identifier in citations:
        if identifier not in metadata and identifier not in requested_images:
            issues.append(identifier + ": citation not found in saved model requests")
    row = {"blind_id": blind_id, "question": question["question"], "task": question["task"],
        "task_dates": question["task_dates"], "execution_status": result.get("status", "missing"),
        "error": result.get("error"), "decision": answer.get("decision"), "answer_text": answer.get("answer_text", ""),
        "citation_ids": citations, "target_event": target, **{key: "" for key in JUDGMENTS}, "reviewer": "", "notes": ""}
    item = {**row, "answer": answer, "last_raw_response": raw_responses[-1] if raw_responses else None,
            "evidence": evidence, "audit_issues": issues}
    section = {"title": blind_id, "caption": json.dumps({k: v for k, v in item.items() if k != "evidence"}, ensure_ascii=False) +
               "\n" + json.dumps(evidence, ensure_ascii=False), "panels": panels}
    trace = {"calls": calls, "exposure_ids_not_in_saved_requests": sorted(set(exposed) - set(metadata) - set(requested_images)),
             "audit_issues": issues, "costs": result.get("costs", {})}
    return row, item, section, trace


def write_blind(out, rows, items, sections, key):
    curation.csv_rows(out / "blind-review.private.csv", sorted(rows, key=lambda r: r["blind_id"]))
    payload = qa._private("www_factual_review", cases=sorted(items, key=lambda r: r["blind_id"]),
        judgment_options=JUDGMENTS, approved=0, factual_correctness="unreviewed; never inferred from class agreement")
    payload["review_id"] = digest(payload)
    write_json(out / "blind-key.private.json", qa._private("www_blind_review_key", review_id=payload["review_id"], cases=key))
    write_json(out / "blind-review.private.json", payload)
    write_static_gallery(out, "Blind factual-support review", sorted(sections, key=lambda r: r["title"]), notes=[
        "Open factual-review.html or edit blind-review.private.csv. Blank judgments mean unreviewed.",
        "Judge target-event identity, claim support and citation support separately; class agreement is not factual correctness.",
        "Condition mapping and technical trace are stored separately in blind-key.private.json.",
        "Evidence shown here comes from saved Qwen request blocks; staged retrieval alone is not presentation."])
    template = Path(__file__).with_name("2026-11-16_www_factual_review.html").read_text(encoding="utf-8")
    embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (out / "factual-review.html").write_text(template.replace("__REVIEW_DATA__", embedded), encoding="utf-8")


def audit(run_path, out, *, questions_path=None, candidates_path=None, handoff_path=None, progress=None):
    root, fingerprints = Path(run_path).resolve(), Fingerprints()
    questions_path = Path(questions_path).resolve() if questions_path else root / "questions.json"
    record = qa.read_bound(root / "run.json", fingerprints)
    hashes = record.get("artifact_sha256", {})
    # The offline execution config can contain disabled/private-path settings;
    # it is provenance, never an evidence block or a blind-review model input.
    contract = read_saved(root / "contract.json", fingerprints, root, hashes, public_evidence=False)
    fingerprints.add(questions_path, contract["questions_sha256"])
    public = runner.public_questions(questions_path)
    conditions = contract["conditions"]
    if len(set(conditions)) != len(conditions) or not set(conditions) <= set(runner.CONDITIONS):
        raise ValueError("Unknown/duplicate frozen conditions")
    questions = {q["question_id"]: q for q in public["questions"]}
    targets = {}
    if bool(candidates_path) != bool(handoff_path):
        raise ValueError("Optional human targets require the original candidates AND their handoff")
    if handoff_path:
        handoff = qa.read_bound(handoff_path, fingerprints)
        fingerprints.add(candidates_path, handoff["source_candidates_sha256"])
        candidates = {q["question_id"]: q for q in qa._load_candidates(candidates_path, fingerprints)["candidates"]}
        for target in handoff["cases"]:
            identifier = target["question_id"]
            if identifier in questions:
                if identifier not in candidates or any(questions[identifier][k] != candidates[identifier][k] for k in qa.PUBLIC_KEYS):
                    raise ValueError("Human target does not bind the exact original public question")
                targets[identifier] = target
    if (root / "results.json").is_file():
        payload = read_saved(root / "results.json", fingerprints, root, hashes)
        if payload.get("questions_sha256") != sha256(questions_path) or payload.get("contract_sha256") != digest(contract):
            raise ValueError("Saved results differ from the recorded execution contract")
        saved = payload["results"]
    else:
        saved = [read_saved(p, fingerprints, root, hashes) for p in sorted((root / "questions").glob("*/*/result.json"))]
    results = {}
    for result in saved:
        key = result["question_id"], result["condition"]
        if key in results or key[0] not in questions or key[1] not in conditions:
            raise ValueError("Unknown/duplicate saved question-condition")
        results[key] = result
    journal = new_run(out, {"inference_run": str(root), "questions_sha256": sha256(questions_path),
        "source_contract_sha256": digest(contract), "scope": "unscored_offline_human_audit; no retrospective plan"}, "www_factual_audit", progress)
    try:
        (journal.out / "assets").mkdir()
        salt = secrets.token_hex(16)
        rows, items, sections, key = [], [], [], []
        for n, question in enumerate(questions.values(), 1):
            identifier = question["question_id"]
            journal.event("case_started", case_id=identifier, index=n, total=len(questions))
            for condition in conditions:
                result = results.get((identifier, condition), {"status": "missing", "answer": None, "costs": {"incomplete": True}})
                folder = root / "questions" / identifier / condition
                exposed = read_saved(folder / "exposure.json", fingerprints, root, hashes)["evidence"] if (folder / "exposure.json").is_file() else {}
                blind_id = digest([salt, identifier, condition])[:16]
                row, item, section, trace = blind_case(question, result, exposed, folder, journal.out, fingerprints, blind_id,
                    root=root, hashes=hashes, target=targets.get(identifier))
                rows.append(row)
                items.append(item)
                sections.append(section)
                key.append({"blind_id": blind_id, "question_id": identifier, "condition": condition, **trace})
        write_blind(journal.out, rows, items, sections, key)
        summary = {"status": "completed", "questions": len(questions), "conditions": conditions, "expected_rows": len(questions)*len(conditions),
            "exported_rows": len(rows), "execution_status_counts": dict(Counter(r["execution_status"] for r in rows)),
            "rows_with_audit_issues": sum(bool(i["audit_issues"]) for i in items), "human_reviewed": 0,
            "models_executed": False, "scoring": "unscored_human_audit; no accuracy or retrospective plan",
            "source_run_status": record.get("status"), "source_contract_sha256": digest(contract),
            "input_sha256": fingerprints.files, "report_path": str(journal.out / "factual-review.html")}
        write_json(journal.out / "summary.json", summary)
        fingerprints.verify()
        complete_run(journal, "completed", summary=summary)
        return summary
    except BaseException as exc:
        complete_run(journal, "failed", error=str(exc))
        raise
    finally:
        journal.close()
