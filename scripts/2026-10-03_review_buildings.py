"""Propose building groups from a prepared manifest without marking guesses reviewed."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facade_change.io import finish_record, new_directory, read_json, run_record, sha256, write_json
from facade_change.preparation import possible_group_overlaps

DETAIL = re.compile(r"(?:balcony(?:v\d+)?|ornament(?:\d+|balcony)?|bottom|left|right|top|roof|"
                    r"wall|facade|closeup|repairs|vase|side|chuch|church|v\d+)", re.IGNORECASE)


def proposed_building(view):
    """Keep compound addresses and house letters; remove only named view details."""
    view = re.sub(r"_full_(?=\d)", "_", view, flags=re.IGNORECASE)
    tokens = view.split("_")
    end = next((i for i, token in enumerate(tokens) if DETAIL.fullmatch(token)), len(tokens))
    address = "_".join(tokens[:end])
    return address if re.search(r"\d", address) else None


def review_buildings(manifest_path, out):
    manifest_path = Path(manifest_path).expanduser().resolve()
    digest = sha256(manifest_path)
    manifest = read_json(manifest_path)
    groups, unresolved = defaultdict(list), []
    for row in manifest["images"]:
        year = row.get("year")
        if (row.get("image_status") != "ready" or not row.get("view_id")
                or type(year) is not int or not 1800 <= year <= 2100):
            unresolved.append(row)
            continue
        address = row.get("building_id") if row.get("metadata_status") == "reviewed" else None
        address = address or proposed_building(row["view_id"])
        groups[(address or row["view_id"]).casefold()].append(row)
    addresses = [proposed_building(rows[0]["view_id"]) for rows in groups.values()]
    overlaps = possible_group_overlaps([value for value in addresses if value])
    result = []
    for _, rows in sorted(groups.items()):
        proposed = proposed_building(rows[0]["view_id"])
        confirmed = {r["building_id"] for r in rows
                     if r.get("metadata_status") == "reviewed" and r.get("building_id")}
        building = next(iter(confirmed)) if len(confirmed) == 1 else proposed
        notes = []
        if not proposed and not confirmed:
            notes.append("Address lacks a house number; confirm building_id")
        if len(confirmed) > 1:
            notes.append("Conflicting confirmed building IDs: " + ";".join(sorted(confirmed)))
        reviewed = bool(building) and len(confirmed) == 1 and all(
            r.get("metadata_status") == "reviewed" and r.get("building_id") == building for r in rows)
        if proposed in overlaps and not reviewed:
            notes.append("Check shared address with: " + ";".join(overlaps[proposed]))
        result.append({"building_id": building or "", "view_ids": ";".join(sorted({r["view_id"] for r in rows})),
                       "image_count": len(rows), "years": ";".join(map(str, sorted({r["year"] for r in rows}))),
                       "reviewed": "true" if reviewed else "false", "notes": "; ".join(notes)})
    out = new_directory(out)
    record = run_record("building_review", {"manifest_path": str(manifest_path), "manifest_sha256": digest,
                        "runner_sha256": sha256(Path(__file__)), "source_images_opened": False})
    write_json(out / "run.json", record)
    try:
        with (out / "buildings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["building_id", "view_ids", "image_count", "years", "reviewed", "notes"])
            writer.writeheader()
            writer.writerows(result)
        write_json(out / "unresolved_images.json", [{"image_id": r["image_id"], "file_name": r["file_name"]}
                                                    for r in unresolved])
        if sha256(manifest_path) != digest:
            raise ValueError("Manifest changed during building review")
        summary = {"candidate_groups": len(result), "eligible_images": sum(r["image_count"] for r in result),
                   "unresolved_images": len(unresolved), "scope": "Filename proposals; unreviewed groups need confirmation"}
        write_json(out / "summary.json", summary)
        finish_record(out, record, "completed")
        return result, summary
    except Exception as exc:
        finish_record(out, record, "failed", str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _, summary = review_buildings(args.manifest, args.out)
    print(summary)
    print((args.out / "buildings.csv").read_text(encoding="utf-8"), end="")
