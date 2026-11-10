"""Bounded evidence selection followed by a fresh, common Qwen judge."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import time

from ..history_features import digest
from ..io import sha256, write_json
from .reasoning import CASE_FIELDS, _content, _image_files, _prompt_context, _system, validate_final
from .store import PrivateEvidenceError, reject_forbidden


data = importlib.import_module(".2026-11-10_inspection_data", __package__)
JUDGE_PROMPT = _system("pair") + (
    " This is supplied-region inspection. Return at most one localization, in region_crop_pixels only. "
    "A change decision requires a nonempty localization. The historical images can inform interpretation "
    "but earlier damage does not establish onset or a transition absent in the compared pair. "
    "For inspection crops, use the supplied crop_local_to_region transform and native offset. "
    "Envelope context outside the requested polygon is not additional evaluated area. "
    "Tighter focus is not necessarily finer source sampling. No inspection selector verdict is provided.")
PLANNER_PROMPT = (
    "Select visual evidence for a separate fresh judge of visible change inside the supplied canonical region. "
    "Return exactly {\"type\":\"stop\"} or {\"type\":\"tool\",\"name\":\"inspect_region\","
    "\"arguments\":{\"observation_ids\":[...],\"bbox_xyxy\":[left,top,right,bottom]}}. "
    "No prose, verdict, answer, confidence or reasoning. IDs must be the ordered current pair together, "
    "or one ID from the historical catalogue. The integer half-open box may be anywhere inside region_crop_pixels; "
    "AnyChange proposals are suggestions ranked by area, not GT or calibrated confidence. "
    "Geometric support is not physical visibility. Only two actions are permitted; never repeat an action. "
    "Stop when acquired evidence is adequate. SAM3 and retrieval are disabled. Catalogue enumeration does not open RGB. "
    "Source pixels may not support finer detail than canonical pixels; never claim upsampling recovers detail.")


class BudgetLimit(RuntimeError):
    pass


class Budget:
    """Charge logical presentations even on cache hits and all measured OOM attempts.

    Before a call reserve two worst-case attempts, matching ModelPool's maximum
    of one OOM retry. Selection also reserves a complete fresh judge. Exact
    per-request processor limits are enforced by the unchanged Qwen adapter.
    """
    def __init__(self, caps):
        self.caps = caps
        self.started = time.monotonic()
        self.costs = dict(model_requests=0, model_calls=0, images=0, input_tokens=0, output_tokens=0,
                          cache_hits=0, model_loads=0, retries=0, actions=0, invalid_replies=0, calls=[],
                          cold_loading_seconds=None, warm_inference_seconds=None,
                          timing_reason="derived only from worker attempt measurements; absent values remain null",
                          accounting_scope="images/tokens are budget-accounted presentations including cache source usage and all OOM attempts; calls[].usage records actual current execution")

    def elapsed(self):
        return time.monotonic() - self.started

    def check(self, images, *, judge_images=0, requests=1):
        cap, used = self.caps, self.costs
        reserve = int(judge_images > 0)
        if self.elapsed() >= cap["max_seconds"]:
            raise BudgetLimit("time_budget_exhausted")
        if images > cap["max_images"] or judge_images > cap["max_images"]:
            raise BudgetLimit("prompt_image_budget_exhausted")
        if used["model_requests"] + requests + reserve > cap["max_model_calls"]:
            raise BudgetLimit("final_judge_model_request_reserved")
        if used["images"] + 2*(images + judge_images) > cap["max_total_images"]:
            raise BudgetLimit("image_budget_reserved_for_judge_and_bounded_oom_retry")
        limits = {}
        for name in ("input_tokens", "output_tokens"):
            remaining = cap["max_total_" + name] - used[name] - 2*reserve*cap["max_" + name]
            limits[name] = min(cap["max_" + name], remaining // 2)
            if limits[name] <= 0:
                raise BudgetLimit(name + "_reserved_for_judge_and_bounded_oom_retry")
        return limits

    def account(self, response, images, stage, elapsed):
        usage = response.get("usage", {})
        effective = response.get("source_usage") if response.get("cache_hit") else usage
        effective = effective or {}
        # Processing records are useful when a cache's original usage lacks a field.
        processing = response.get("processing", {})
        self.costs["model_requests"] += 1
        record = dict(stage=stage, elapsed_seconds=elapsed, usage=usage,
            source_usage=response.get("source_usage"), cache_key=response.get("cache_key"),
            processing=processing, generation=response.get("generation"), provenance=response.get("provenance"),
            status=response.get("status"), error=response.get("error"))
        self.costs["calls"].append(record)
        for key in ("input_tokens", "output_tokens"):
            value = effective.get(key, processing.get(key))
            if value is None and key == "output_tokens":
                value = response.get("generation", {}).get("actual_output_tokens")
            if type(value) is not int or value < 0:
                self.costs["incomplete"] = True
                raise RuntimeError("Missing/invalid measured " + key + "; usage cannot be assumed zero")
            self.costs[key] += value
        delivered = effective.get("images", images)
        if type(delivered) is not int or delivered < 0:
            raise RuntimeError("Invalid measured image presentations")
        self.costs["images"] += max(images, delivered)
        for key in ("model_calls", "cache_hits", "model_loads", "retries"):
            self.costs[key] += usage.get(key, 0)
        if usage.get("incomplete") or usage.get("unobserved_worker_cost"):
            self.costs["incomplete"] = True
            raise RuntimeError("Worker usage incomplete; cannot certify remaining budget")
        attempts = [a.get("usage", {}) for a in usage.get("attempts", [])]
        loads = [a["load_seconds"] for a in attempts if a.get("load_seconds") is not None]
        if loads:
            self.costs["cold_loading_seconds"] = (self.costs["cold_loading_seconds"] or 0.) + sum(loads)
        # Do not call wall time minus loading 'warm inference': I/O is included.
        for key in ("images", "input_tokens", "output_tokens"):
            if self.costs[key] > self.caps["max_total_" + key]:
                raise BudgetLimit("measured_" + key + "_exceeded_budget")


def heuristic_actions(inventory):
    proposal = (inventory["proposals"]["selected"] or [inventory["proposals"]["fallback"]])[0]
    box = proposal["bbox_xyxy"]
    return [dict(observation_ids=inventory["current_ids"], bbox_xyxy=box),
            dict(observation_ids=[inventory["history_ids"][0]], bbox_xyxy=box)]


def run_policy(policy, question, case, service, pool, out, protocol, *, event=None, started_at=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(protocol["caps"])
    if started_at is not None:
        budget.started = started_at
    inventory = service.inventory
    pair = service.base_pair()
    acquired, actions = [], []
    planner_messages, judge_messages = [], []
    seen_actions = set()
    termination = "no_selection_required"
    result = None
    sequence = 0
    context = {key: case[key] for key in CASE_FIELDS if key in case}
    context["question"] = question["question"]
    context = _prompt_context(context)
    selector = dict(inventory_sha256=service.inventory_hash, catalogue=service.catalogue(),
                    proposals={k: inventory["proposals"][k] for k in ("selected", "fallback", "ordering")},
                    proposal_preview_sha256=inventory["proposal_preview_sha256"],
                    current_ids=inventory["current_ids"], allowed_actions=inventory["allowed_actions"], caps=protocol["caps"])
    contract = dict(protocol_version=data.VERSION, policy=policy, inventory_sha256=service.inventory_hash,
                    caps=protocol["caps"], judge_prompt_sha256=digest(JUDGE_PROMPT),
                    model_config_sha256=digest(pool.models), final_frame="region_crop_pixels")
    write_json(out / "contract.json", contract)

    def trace(kind, **values):
        nonlocal sequence
        sequence += 1
        record = dict(sequence=sequence, timestamp_utc=datetime.now(timezone.utc).isoformat(), event_type=kind, **values)
        with (out / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if event:
            event("inspection_" + kind, case_id=case["case_id"], policy=policy, **values)

    def persist():
        budget.costs["elapsed_seconds"] = budget.elapsed()
        write_json(out / "planner_messages.json", planner_messages)
        write_json(out / "judge_messages.json", judge_messages)
        write_json(out / "messages.json", dict(planner=planner_messages, judge=judge_messages))
        write_json(out / "state.json", dict(contract=contract, phase="done" if result else "running", actions=actions,
                                           selection_termination=termination, result=result))
        reported = copy.deepcopy(budget.costs)
        if reported.get("incomplete"):
            keys = ("model_calls", "images", "input_tokens", "output_tokens", "model_loads", "retries")
            reported["observed_lower_bounds"] = {key: reported[key] for key in keys}
            reported.update({key: None for key in keys})
            reported["measurement_reason"] = "Worker failed without complete usage; totals are unknown, recorded measurements are lower bounds"
        write_json(out / "costs.json", reported)

    def fresh_judge():
        content = [dict(type="text", text=json.dumps(context)), *_content(pair, "current_pair", case)]
        for evidence in acquired:
            content.extend(_content(evidence, "inspect_region", case))
        return [dict(role="system", content=[dict(type="text", text=JUDGE_PROMPT)]),
                dict(role="user", content=content)]

    def request(messages, stage, judge_images=0):
        reject_forbidden(messages)
        images = _image_files(messages)
        expected_images = {a["path"]: a["sha256"] for r in service.evidence.values() for a in r.get("artifacts", [])}
        expected_images[inventory["proposal_preview_path"]] = inventory["proposal_preview_sha256"]
        for path in images:
            if path not in expected_images or sha256(path) != expected_images[path]:
                raise ValueError("Model-visible evidence is missing from the frozen set or changed")
        limits = budget.check(len(images), judge_images=judge_images)
        payload = dict(op="generate", messages=copy.deepcopy(messages), max_new_tokens=limits["output_tokens"],
                       max_input_tokens=limits["input_tokens"], max_images=protocol["caps"]["max_images"],
                       timeout_seconds=protocol["caps"]["max_seconds"] - budget.elapsed())
        semantic_messages = copy.deepcopy(messages)
        for message in semantic_messages:
            for block in message["content"]:
                if block["type"] == "image":
                    block["sha256"] = sha256(block.pop("path"))
        input_hash = digest(dict(messages=semantic_messages, request_limits=limits,
                                 model_config_sha256=contract["model_config_sha256"]))
        if stage == "judge":
            write_json(out / "final_judge_input.json", dict(input_sha256=input_hash, messages=messages,
                image_sha256={p: sha256(p) for p in images}, contract=contract))
        persist()
        trace("model_requested", stage=stage, input_sha256=input_hash, images=len(images), request_limits=limits)
        started = time.monotonic()
        try:
            response = pool.call("qwen", payload)
        except Exception:
            budget.costs["model_requests"] += 1
            budget.costs["incomplete"] = True
            raise
        # Retain malformed/truncated generation diagnostics before validation/accounting.
        write_json(out / (f"response-{budget.costs['model_requests']+1}.json"), response)
        trace("model_returned", stage=stage, status=response.get("status"), generation=response.get("generation"),
              processing=response.get("processing"), error=response.get("error"))
        budget.account(response, len(images), stage, time.monotonic()-started)
        if response.get("status") != "ok":
            raise RuntimeError("Model request failed: " + str(response.get("error")))
        text = response.get("text")
        if not isinstance(text, str) or len(text) > 100000:
            raise ValueError("Model response must be bounded text")
        messages.append(dict(role="assistant", content=[dict(type="text", text=text)]))
        persist()
        if budget.elapsed() > protocol["caps"]["max_seconds"]:
            raise BudgetLimit("time_budget_exhausted")
        reject_forbidden(text)
        return text

    def invalid(messages, error):
        budget.costs["invalid_replies"] += 1
        trace("invalid_reply", error=str(error), count=budget.costs["invalid_replies"])
        if budget.costs["invalid_replies"] > protocol["caps"]["max_invalid_replies"]:
            raise RuntimeError("invalid_reply_limit: " + str(error))
        messages.append(dict(role="user", content=[dict(type="text", text="Invalid JSON/schema: " + str(error) + ". Return the documented JSON object only.")]))

    def act(arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"observation_ids", "bbox_xyxy"}:
            raise ValueError("Only observation_ids and bbox_xyxy are allowed")
        service.preview_action(**arguments)  # Bounds, raw pixel sizes and correspondence before any opening.
        key = digest(arguments)
        if key in seen_actions:
            raise ValueError("An identical inspection action cannot be repeated")
        if budget.costs["actions"] >= protocol["caps"]["max_actions"]:
            raise BudgetLimit("action_cap")
        # Reserve a full judge with the new images; never silently drop acquired evidence.
        count = len(_image_files(fresh_judge())) + len(arguments["observation_ids"])
        budget.check(count)
        evidence = service.execute("inspect_region", arguments)
        acquired.append(evidence)
        seen_actions.add(key)
        budget.costs["actions"] += 1
        row = dict(step=len(actions)+1, eligible_ids=inventory["current_ids"] + inventory["history_ids"],
                   selected_ids=arguments["observation_ids"], requested_bbox_xyxy=arguments["bbox_xyxy"],
                   delivered=[r["geometry"] for r in evidence["evidence"]],
                   evidence_ids=[r["evidence_id"] for r in evidence["evidence"]])
        actions.append(row)
        trace("inspection_delivered", **row)
        persist()
        return evidence

    try:
        if policy == "fixed_history":
            height, width = inventory["frame"]["shape"]
            for oid in inventory["history_ids"][:2]:
                act(dict(observation_ids=[oid], bbox_xyxy=[0, 0, width, height]))
            termination = "fixed_latest_history_complete"
        elif policy == "heuristic":
            trace("selector_start", **selector)
            for arguments in heuristic_actions(inventory):
                try:
                    act(arguments)
                except BudgetLimit as exc:
                    termination = "budget_limited: " + str(exc)
                    break
            else:
                termination = "heuristic_schedule_complete"
        elif policy == "agent":
            trace("selector_start", **selector)
            proposal_image = dict(type="image", path=inventory["proposal_preview_path"], evidence_id="proposal-preview")
            planner_messages.extend([
                dict(role="system", content=[dict(type="text", text=PLANNER_PROMPT)]),
                dict(role="user", content=[dict(type="text", text=json.dumps(context)), *_content(pair, "current_pair", case),
                    dict(type="text", text=json.dumps(selector)), proposal_image])])
            termination = "planner_stop"
            pending_invalid_reply = False
            while budget.costs["actions"] < protocol["caps"]["max_actions"]:
                try:
                    text = request(planner_messages, "planner", judge_images=len(_image_files(fresh_judge())))
                except BudgetLimit as exc:
                    if pending_invalid_reply:
                        raise RuntimeError("invalid_reply_retry_budget_exhausted: " + str(exc)) from exc
                    termination = "budget_limited: " + str(exc)
                    break
                try:
                    choice = json.loads(text)
                    if choice == {"type": "stop"}:
                        break
                    if (not isinstance(choice, dict) or set(choice) != {"type", "name", "arguments"}
                            or choice["type"] != "tool" or choice["name"] != "inspect_region"):
                        raise ValueError("Expected stop or inspect_region, with no planner prose/verdict")
                    evidence = act(choice["arguments"])
                    pending_invalid_reply = False
                    planner_messages.append(dict(role="user", content=_content(evidence, "inspect_region", case)))
                except BudgetLimit as exc:
                    termination = "budget_limited: " + str(exc)
                    break
                except (ValueError, TypeError) as exc:
                    if isinstance(exc, PrivateEvidenceError):
                        raise
                    invalid(planner_messages, exc)
                    pending_invalid_reply = True
            else:
                termination = "action_cap_reached; direct_fresh_judge"
        elif policy != "pair":
            raise ValueError("Unknown inspection policy")
        # Deliberately no selector messages, tool-request prose, or tentative answer.
        judge_messages.extend(fresh_judge())
        seen = {row["evidence_id"] for group in [pair, *acquired] for row in group["evidence"]}
        while True:
            text = request(judge_messages, "judge")
            try:
                value = json.loads(text)
                location = value.get("localization") if isinstance(value, dict) else None
                if location is not None and location.get("coordinate_system") != "region_crop_pixels":
                    raise ValueError("Inspection judge localization must use region_crop_pixels")
                if isinstance(value, dict) and value.get("decision") == "change" and location is None:
                    raise ValueError("Change prediction requires exactly one nonempty box")
                result = validate_final(value, case, seen, service.evidence)
                break
            except (ValueError, TypeError, AttributeError) as exc:
                if isinstance(exc, PrivateEvidenceError):
                    raise
                invalid(judge_messages, exc)
    except PrivateEvidenceError:
        raise
    except Exception as exc:
        result = dict(schema_version=1, case_id=case["case_id"], region_id=case["region_id"], status="error",
            decision=None, localization=None, evidence_ids=[], error=f"{type(exc).__name__}: {exc}",
            reason="budget_limited" if isinstance(exc, BudgetLimit) else "technical_error")
    finally:
        if result is not None:
            result.update(policy=policy, protocol_version=data.VERSION, inventory_sha256=service.inventory_hash,
                          selection_termination=termination, judge_prompt_sha256=contract["judge_prompt_sha256"])
            if result["status"] in {"completed", "abstained"}:
                hypothesis = service.store.add_evidence(dict(kind="agent_hypothesis", region_id=case["region_id"],
                    observation_ids=inventory["current_ids"], available_at=case["cutoff"], summary=result,
                    source_evidence_ids=result["evidence_ids"], model=pool.models["qwen"].get("model_id", "synthetic"),
                    revision=pool.models["qwen"].get("revision"), prompt_sha256=contract["judge_prompt_sha256"],
                    config_sha256=contract["model_config_sha256"], parameters=dict(policy=policy, protocol_version=data.VERSION), artifacts=[]))
                result["hypothesis_evidence_id"] = hypothesis["evidence_id"]
            write_json(out / "result.json", result)
        persist()
    trace("completed", status=result["status"], selection_termination=termination)
    return result
