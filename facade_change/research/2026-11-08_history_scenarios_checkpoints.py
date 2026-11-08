"""Bounded, recoverable checkpoint retention for history-scenarios schema 2.

The operating threshold is selected by the evaluator. This module ranks the
resulting *VAL building-macro F1*, never recall used to choose that threshold.
One immutable weights file can satisfy several retention criteria. Only the
latest resume payload contains optimizer, scheduler, scaler and RNG state.
The existing RunStore event ledger records creation, pointer commits and explicit
retirement; historical artifact events do not imply that retired files are live.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch

from facade_change.io import read_json, sha256, sync_directory, write_json

PREFIX = "2026-11-08_retention-v2_"
SCHEMA = "history-scenarios-checkpoints/v2"


def _fault(stage):
    """Test seam for interruption at transaction boundaries."""


def _load(path):
    # These files are generated inside this run and verified before loading.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        result = type(value)((key, _cpu(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def state_dict_digest(state):
    """Hash tensor values rather than nondeterministic torch-save container bytes."""
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError("Checkpoint retention expects a dense tensor state_dict")
        value = tensor.detach().to("cpu").contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        result.update(header.encode("utf-8") + b"\0")
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        result.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return result.hexdigest()


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _metrics(values):
    # Unknown metrics remain NA, never a best score of zero.
    return {name: float(values[name]) if _finite(values.get(name)) else None
            for name in ("f1", "iou", "recall", "fpr", "threshold",
                         "prediction_score", "semantic_miou")}


def _eligible(row, criterion):
    metrics = row["metrics"]
    if not _finite(metrics.get(criterion)):
        return False
    if criterion in ("f1", "iou", "recall"):
        return _finite(metrics.get("fpr")) and metrics["fpr"] <= .05 + 1e-12
    return True


def _rank(row, criterion):
    value = row["metrics"][criterion]
    # Ties select the earlier epoch, then the content hash. No secondary metric
    # can silently affect patience, which remains the trainer's F1-only decision.
    return (value if criterion == "prediction_score" else -value,
            row["epoch"], row["state_dict_sha256"])


def _selection(candidates, contract):
    winners = {}
    ranked = sorted((r for r in candidates if _eligible(r, "f1")), key=lambda r: _rank(r, "f1"))
    winners["f1_top3"] = ranked[:3]
    for criterion in ("iou", "recall", "prediction_score", "semantic_miou"):
        eligible = [row for row in candidates if _eligible(row, criterion)]
        winners[criterion] = min(eligible, key=lambda row: _rank(row, criterion)) if eligible else None
    retained = OrderedDict()
    for rank, row in enumerate(winners["f1_top3"], 1):
        retained.setdefault(row["state_dict_sha256"], {**row, "reasons": []})["reasons"].append(f"f1_rank_{rank}")
    for criterion, row in winners.items():
        if criterion != "f1_top3" and row is not None:
            retained.setdefault(row["state_dict_sha256"], {**row, "reasons": []})["reasons"].append(criterion)
    return dict(schema=SCHEMA, contract=contract,
                criterion="VAL building-macro F1 at separately calibrated max-recall FPR<=0.05 threshold",
                tie_break="earlier epoch, then state_dict SHA-256",
                checkpoint=ranked[0] if ranked else None,
                prediction_checkpoint=winners["prediction_score"],
                semantic_checkpoint=winners["semantic_miou"],
                winners=winners, retained=list(retained.values()))


def live_checkpoint_artifacts(events):
    """Resolve append-only creation/retirement events without requiring retired files."""
    live = {}
    for event in events:
        kind = event.get("kind", event.get("event"))
        # ExperimentRun calls the discriminator event_type.
        kind = event.get("event_type", kind)
        if kind == "checkpoint_artifact_saved":
            live[event["path"]] = event["sha256"]
        elif kind == "artifact_retired" and event.get("owner") == SCHEMA:
            live.pop(event["path"], None)
    return live


class CheckpointManager:
    """Model-local retention adapter on the existing RunStore/append-only ledger.

    ``save_epoch`` receives the entire resumable trainer state. ``resume`` returns
    it verbatim except selected pointers, updated to the durable weights files.
    The manager owns only files with its unique prefix beneath ``root``.
    """

    def __init__(self, root, contract, ledger=None, restore_contract=None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.ledger = ledger
        self.restore_contract = restore_contract or {}
        self.latest_path = self.root / (PREFIX + "latest.json")
        self.selection_path = self.root / (PREFIX + "selection.json")
        self.journal_path = self.root / (PREFIX + "transaction.json")
        self.recover()

    def _event(self, kind, **values):
        if self.ledger is None:
            return None
        return self.ledger.event(kind, owner=SCHEMA, **values)

    def _owned(self, path):
        path = Path(path).resolve()
        if path.parent != self.root or not path.name.startswith(PREFIX):
            raise ValueError(f"Checkpoint retention refuses a foreign artifact: {path}")
        return path

    def _atomic_payload(self, path, payload):
        path = self._owned(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable checkpoint: {path}")
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)

    def _verify(self, pointer, *, resume=False):
        path = self._owned(pointer["path"])
        if not path.is_file():
            raise ValueError(f"Required live checkpoint is missing: {path}")
        actual = sha256(path)
        if pointer.get("sha256") and actual != pointer["sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch: {path}")
        if pointer.get("contract") != self.contract:
            raise ValueError("Checkpoint contract differs; use a new run")
        payload = _load(path)
        if payload.get("contract") != self.contract:
            raise ValueError("Checkpoint payload contract differs")
        if state_dict_digest(payload["model"]) != pointer["state_dict_sha256"]:
            raise ValueError("Checkpoint state_dict digest differs")
        if resume and payload["epoch"] != pointer["epoch"]:
            raise ValueError("Latest checkpoint epoch differs from pointer")
        pointer["sha256"] = actual
        return payload

    def _read_selection(self):
        if not self.selection_path.exists():
            return _selection([], self.contract)
        selection = read_json(self.selection_path)
        if selection.get("schema") != SCHEMA or selection.get("contract") != self.contract:
            raise ValueError("Checkpoint retention contract differs; use a new run")
        return selection

    def _retire(self, pointer, reason):
        path = self._owned(pointer["path"])
        if path.exists() and pointer.get("sha256") and sha256(path) != pointer["sha256"]:
            raise ValueError(f"Refusing to retire a changed checkpoint: {path}")
        self._event("artifact_retirement_requested", path=str(path), sha256=pointer.get("sha256"), reason=reason)
        if path.exists():
            path.unlink()
            sync_directory(path.parent)
        self._event("artifact_retired", path=str(path), sha256=pointer.get("sha256"), reason=reason)

    def _commit(self, transaction):
        latest, selection = transaction["latest"], transaction["selection"]
        self._verify(latest, resume=True)
        for pointer in selection["retained"]:
            self._verify(pointer)
        # Both pointers can be replayed from the durable journal after a crash.
        write_json(self.selection_path, selection)
        _fault("after_selection_pointer")
        write_json(self.latest_path, latest)
        _fault("after_latest_pointer")
        self._event("checkpoint_pointers_committed", latest=latest,
                    retained=[r["path"] for r in selection["retained"]])
        _fault("before_retirement")
        live = {latest["path"], *(row["path"] for row in selection["retained"])}
        for pointer in transaction["retire"]:
            if pointer["path"] not in live:
                self._retire(pointer, "superseded after verified latest and winner pointers committed")
        # An interruption before journal creation can leave a fully written
        # candidate weight file. It is safe to retire only after the new live
        # pointers have been verified and committed, never before that point.
        for pattern in (PREFIX + "weights-*.pt", PREFIX + "resume-*.pt", PREFIX + "*.pt.tmp"):
            for path in self.root.glob(pattern):
                if str(path) not in live:
                    self._retire(dict(path=str(path), sha256=sha256(path)),
                                 "unreferenced interrupted checkpoint after verified pointer commit")
        self.journal_path.unlink(missing_ok=True)
        sync_directory(self.root)

    def recover(self):
        if not self.journal_path.exists():
            # Validate pointer contracts even when no recovery was needed.
            self._read_selection()
            if self.latest_path.exists() and read_json(self.latest_path).get("contract") != self.contract:
                raise ValueError("Latest resume contract differs; use a new run")
            return
        transaction = read_json(self.journal_path)
        if transaction.get("schema") != SCHEMA or transaction.get("contract") != self.contract:
            raise ValueError("Interrupted checkpoint transaction contract differs")
        path = self._owned(transaction["latest"]["path"])
        if path.exists():
            # A crash after atomic payload publication but before recording its
            # byte hash is recoverable: verify the expected tensor digest/epoch.
            self._verify(transaction["latest"], resume=True)
            if not transaction.get("latest_creation_logged"):
                self._event("checkpoint_artifact_saved", artifact_type="latest_resume",
                            recovered=True, **transaction["latest"])
            self._commit(transaction)
            self._event("checkpoint_transaction_recovered", epoch=transaction["latest"]["epoch"])
        else:
            temporary = path.with_name(path.name + ".tmp")
            if temporary.exists():
                self._retire(dict(path=str(temporary)), "interrupted incomplete checkpoint write")
            self.journal_path.unlink()
            sync_directory(self.root)
            self._event("checkpoint_transaction_aborted", epoch=transaction["latest"]["epoch"],
                        reason="No atomically published resume payload; preceding epoch remains authoritative")

    def resume(self):
        self.recover()
        if not self.latest_path.exists():
            return None
        selection = self._read_selection()
        for pointer in selection["retained"]:
            self._verify(pointer)
        state = self._verify(read_json(self.latest_path), resume=True)
        state["selected"] = selection["checkpoint"]
        state["prediction_selected"] = selection["prediction_checkpoint"]
        return state

    def save_epoch(self, model_state, state, metrics, epoch):
        self.recover()
        previous = self._read_selection()
        previous_latest = read_json(self.latest_path) if self.latest_path.exists() else None
        if previous_latest is not None:
            self._verify(previous_latest, resume=True)
        for pointer in previous["retained"]:
            self._verify(pointer)
        if previous_latest is not None and int(epoch) <= previous_latest["epoch"]:
            raise ValueError("Checkpoint epochs must increase after the committed latest")
        digest = state_dict_digest(model_state)
        weight_path = self.root / (PREFIX + f"weights-{digest}.pt")
        candidate = dict(path=str(weight_path), state_dict_sha256=digest,
                         epoch=int(epoch), metrics=_metrics(metrics), contract=self.contract)
        candidates = [{k: v for k, v in row.items() if k != "reasons"} for row in previous["retained"]]
        duplicate = next((row for row in candidates if row["state_dict_sha256"] == digest), None)
        if duplicate is None:
            candidates.append(candidate)
        else:
            # Identical weights have identical deterministic VAL predictions.
            # Retain the first producing epoch, never another physical file.
            candidate = duplicate
        selection = _selection(candidates, self.contract)
        chosen = next((row for row in selection["retained"] if row["state_dict_sha256"] == digest), None)
        if chosen is not None and not weight_path.exists():
            self._atomic_payload(weight_path, dict(model=_cpu(model_state), contract=self.contract,
                restore_contract=copy.deepcopy(self.restore_contract), state_dict_sha256=digest))
            candidate["sha256"] = sha256(weight_path)
            self._event("checkpoint_artifact_saved", artifact_type="winner_weights", **candidate)
        # Pointer copies are intentionally small; propagate one byte hash to all
        # criterion references, including several reasons for the same weights.
        for row in candidates:
            if row["state_dict_sha256"] in {r["state_dict_sha256"] for r in selection["retained"]}:
                row["sha256"] = row.get("sha256") or sha256(row["path"])
        selection = _selection(candidates, self.contract)
        _fault("after_winner_write")
        resume_path = self.root / (PREFIX + f"resume-{int(epoch):04d}-{uuid.uuid4().hex}.pt")
        latest = dict(path=str(resume_path), epoch=int(epoch), contract=self.contract,
                      state_dict_sha256=digest)
        state = _cpu(dict(state, model=model_state, epoch=int(epoch), contract=self.contract,
                          selected=selection["checkpoint"], prediction_selected=selection["prediction_checkpoint"]))
        retirement = list(previous["retained"])
        if previous_latest is not None:
            retirement.append(previous_latest)
        transaction = dict(schema=SCHEMA, contract=self.contract, latest=latest,
                           selection=selection, retire=retirement)
        write_json(self.journal_path, transaction)
        _fault("before_resume_write")
        self._atomic_payload(resume_path, state)
        _fault("after_resume_write")
        self._verify(latest, resume=True)
        self._event("checkpoint_artifact_saved", artifact_type="latest_resume", **latest)
        transaction["latest_creation_logged"] = True
        write_json(self.journal_path, transaction)
        _fault("before_pointer_commit")
        self._commit(transaction)
        return {**selection, "latest": latest}

    def finalize(self):
        self.recover()
        selection = self._read_selection()
        for pointer in selection["retained"]:
            self._verify(pointer)
        latest = read_json(self.latest_path) if self.latest_path.exists() else None
        if latest is not None:
            self._verify(latest, resume=True)
        return {**selection, "latest": latest}
