"""Physical identity validation adapted from WWW 41ee86e6; no retrieval/exposure policy.

Original building/view IDs and split/exposure records are never rewritten.
Pending aliases are review work, not permission to merge buildings or views.
"""
from pathlib import Path

from ..io import read_json, sha256
from .store import _id, reject_forbidden


def template():
    return {"schema_version": 1, "kind": "www_physical_identity_review", "groups": [
        {"physical_building_id": names[0], "building_ids": list(names), "decision": "pending",
         "reviewer": "", "rationale": "V4 proposed alias; verify physical building, views, splits and exposure before accepting",
         "sources": []} for names in (("Voskova_10", "Voskova_12"), ("baskov_32", "baskov_34"))]}


class Identities:
    def __init__(self, path=None, fingerprints=None, expected=None):
        self.source, self.groups, self.by_id = None, [], {}
        if path is None:
            self.by_id = {name: group for group in template()["groups"] for name in group["building_ids"]}
            return
        path = Path(path).resolve()
        fingerprints.add(path, expected)
        document = read_json(path)
        reject_forbidden(document)
        if (set(document) != {"schema_version", "kind", "groups"} or document["schema_version"] != 1
                or document["kind"] != "www_physical_identity_review" or not isinstance(document["groups"], list)):
            raise ValueError("Expected a reviewed WWW physical identity mapping")
        for group in document["groups"]:
            if set(group) != {"physical_building_id", "building_ids", "decision", "reviewer", "rationale", "sources"}:
                raise ValueError("Identity group requires original IDs, explicit decision, reviewer, rationale and sources")
            members = group["building_ids"]
            if not isinstance(members, list) or len(members) < 2 or len(set(members)) != len(members):
                raise ValueError("Identity groups require at least two distinct original IDs")
            for name in members:
                _id(name, "building_id")
                if name in self.by_id:
                    raise ValueError("Overlapping identity groups are ambiguous")
                self.by_id[name] = group
            if group["physical_building_id"] not in members or group["decision"] not in {"pending", "same_building", "distinct"}:
                raise ValueError("Explicit identity decision and a canonical original building ID are required")
            if not isinstance(group["sources"], list):
                raise ValueError("Identity sources must be a list of actual files/hashes")
            if not isinstance(group["reviewer"], str) or not isinstance(group["rationale"], str):
                raise ValueError("Identity reviewer and rationale must be strings")
            if group["decision"] != "pending" and (not group["reviewer"].strip() or not group["rationale"].strip() or not group["sources"]):
                raise ValueError("A reviewed identity decision needs a human reviewer, rationale and source provenance")
            for source in group["sources"]:
                if set(source) != {"path", "sha256"}:
                    raise ValueError("Identity provenance requires a source path and actual SHA-256")
                fingerprints.add(source["path"], source["sha256"])
            self.groups.append(group)
        for group in template()["groups"]:
            for name in group["building_ids"]:
                self.by_id.setdefault(name, group)
        self.source = {"path": str(path), "sha256": sha256(path)}

    def physical(self, building):
        group = self.by_id.get(building)
        return group["physical_building_id"] if group and group["decision"] == "same_building" else building

    def pending(self, building):
        return self.by_id.get(building, {}).get("decision") == "pending"

    def candidate_members(self, building):
        group = self.by_id.get(building)
        return group["building_ids"] if group and group["decision"] != "distinct" else [building]

    def problem(self, building, tracks, *, require_members=True):
        if self.pending(building):
            return "physical_identity_review_pending"
        members = set(self.candidate_members(building))
        found = {r["building_id"] for r in tracks if r["building_id"] in members}
        if require_members and found != members:
            return "physical_identity_member_missing_from_history_audit"
        splits = {r["split"] for r in tracks if self.physical(r["building_id"]) == self.physical(building)}
        if len(splits) > 1:
            return "physical_building_crosses_original_splits"
        return None
