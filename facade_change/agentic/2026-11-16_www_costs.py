"""Measured saved WWW costs by model role; no inference and no quality ranking."""
from collections import Counter
import math
from pathlib import Path

FIELDS = ("input_tokens", "output_tokens", "model_calls", "elapsed_seconds", "images", "cache_hits", "retries", "worker_starts", "model_loads")


def empty_role():
    return {**dict.fromkeys(FIELDS, 0), "requests": 0, "status_counts": Counter(), "attempt_status_counts": Counter(),
            "missing_measurements": Counter(), "incomplete_requests": 0,
            "resource_verification_seconds": 0., "resource_bytes_hashed": 0,
            "cached_source_input_tokens": 0, "cached_source_output_tokens": 0}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def case_costs(result, folder, read):
    """Use call artifacts plus ledger fallbacks, counting technical attempts only once."""
    ledger_path = folder / "costs.json"
    ledger = read(ledger_path) if ledger_path.is_file() else result.get("costs", {})
    roles = {role: empty_role() for role in ("qwen", "jina")}
    issues, trace = [], []
    resumed = result.get("resumed_from") or ledger.get("resumed_from")
    if resumed:
        # Copied historical call files are not new worker invocations.
        return {"roles": roles, "retrieval_calls": 0, "retrieval_status_counts": {},
            "elapsed_seconds": ledger.get("elapsed_seconds", 0), "result_cache_hits": ledger.get("cache_hits", 1),
            "reused_result": True, "source_costs": result.get("source_costs", ledger.get("source_costs")),
            "incomplete": bool(ledger.get("incomplete")), "issues": issues, "calls": trace}
    entries = {}
    for entry in ledger.get("calls", []):
        name = Path(entry["artifact"]).name
        if name in entries:
            raise ValueError("Duplicate saved cost call entry")
        entries[name] = entry
    paths = {p.name: p for p in (folder / "calls").glob("*.json")}
    for name in sorted(set(entries) | set(paths)):
        entry = entries.get(name, {})
        if name in paths:
            call = read(paths[name], entry.get("sha256"))
            response = call.get("response", {})
            role = call.get("role", entry.get("role"))
            usage = response.get("usage", {})
            if entry.get("role") and entry["role"] != role:
                raise ValueError("Saved role differs from cost ledger")
            if name not in entries:
                issues.append("saved_call_absent_from_cost_ledger:" + name)
        else:
            response, role, usage = {}, entry.get("role"), entry.get("usage", {})
            issues.append("missing_call_artifact_using_recorded_usage:" + name)
        if role not in roles:
            issues.append("unknown_model_role:" + str(role))
            continue
        measured = roles[role]
        measured["requests"] += 1
        measured["status_counts"][response.get("status", "unknown")] += 1
        incomplete = bool(usage.get("incomplete") or usage.get("unobserved_worker_cost") or name not in paths)
        for field in FIELDS:
            value = usage.get(field)
            if field == "cache_hits" and value is None and response.get("cache_hit"):
                value = 1
            if number(value):
                measured[field] += value
            else:
                measured["missing_measurements"][field] += 1
                incomplete = True
        # ModelPool usage already sums its OOM/technical attempts. Do not sum them again.
        for attempt in usage.get("attempts", []):
            measured["attempt_status_counts"][attempt.get("status", "unknown")] += 1
        verification = usage.get("resource_validation", {})
        if number(verification.get("seconds")):
            measured["resource_verification_seconds"] += verification["seconds"]
        if number(verification.get("bytes_hashed")):
            measured["resource_bytes_hashed"] += verification["bytes_hashed"]
        source = response.get("source_usage", entry.get("source_usage")) or {}
        for field in ("input_tokens", "output_tokens"):
            if number(source.get(field)):
                measured["cached_source_" + field] += source[field]
        measured["incomplete_requests"] += incomplete
        trace.append({"artifact": str(paths.get(name, folder / "calls" / name)), "role": role,
            "status": response.get("status", "unknown"), "error": response.get("error"),
            "usage": usage, "source_usage": source, "incomplete": incomplete})
    retrieval = []
    for path in sorted(folder.glob("retrieval-*.json")):
        tool = read(path)
        retrieval.append("error" if tool.get("error") else tool.get("result", {}).get("status", "unknown"))
    searches = ledger.get("retrieval_rounds")
    if not number(searches):
        searches = len(retrieval)
        issues.append("retrieval_count_inferred_from_saved_tool_files")
    elif searches != len(retrieval):
        issues.append("retrieval_ledger_and_artifact_count_differ")
        searches = max(searches, len(retrieval))
    if number(ledger.get("qwen_requests")) and ledger["qwen_requests"] != roles["qwen"]["requests"]:
        issues.append("qwen_request_ledger_and_observed_call_count_differ")
    for field in FIELDS:
        # Case wall time includes preparation/retrieval, so it is not a sum of worker times.
        if field != "elapsed_seconds" and number(ledger.get(field)) and ledger[field] != sum(r[field] for r in roles.values()):
            issues.append("saved_total_not_fully_attributed_by_role:" + field)
    elapsed = ledger.get("elapsed_seconds")
    if not number(elapsed):
        elapsed = None
        issues.append("case_elapsed_time_not_measured")
    return {"roles": roles, "retrieval_calls": searches, "retrieval_status_counts": dict(Counter(retrieval)),
        "elapsed_seconds": elapsed, "reused_result": False, "result_cache_hits": 0,
        "incomplete": bool(issues or ledger.get("incomplete") or any(r["incomplete_requests"] for r in roles.values()) or result.get("status") == "missing"),
        "issues": issues, "calls": trace,
        "ledger_totals_all_roles": {k: ledger[k] for k in (*FIELDS, "qwen_requests", "retrieval_rounds") if k in ledger}}


def aggregate(rows, conditions):
    result = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        roles = {role: empty_role() for role in ("qwen", "jina")}
        for row in subset:
            for role, values in row["costs"]["roles"].items():
                for field in roles[role]:
                    if isinstance(roles[role][field], Counter):
                        roles[role][field].update(values[field])
                    else:
                        roles[role][field] += values[field]
        result[condition] = {"rows": len(subset), "status_counts": dict(Counter(r["status"] for r in subset)),
            "roles": roles, "retrieval_calls": sum(r["costs"]["retrieval_calls"] for r in subset),
            "elapsed_seconds_observed": sum(r["costs"]["elapsed_seconds"] or 0 for r in subset),
            "elapsed_seconds_missing_rows": sum(r["costs"]["elapsed_seconds"] is None for r in subset),
            "incomplete_rows": sum(r["costs"]["incomplete"] for r in subset),
            "reused_result_rows": sum(r["costs"]["reused_result"] for r in subset)}
    return result
