"""Transport-only decoding. Never repair the CVPR selector or localization schema."""
import json
import math
import re


def parse_reply(raw, changes):
    if not isinstance(raw, str) or len(raw) > 100_000:
        raise ValueError("Expected bounded JSON text")
    text = raw.strip()
    if text != raw:
        changes.append("removed_outer_whitespace")
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL)
    if fence:
        text = fence[1].strip()
        changes.append("removed_single_json_fence")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Non-finite JSON number: " + value)

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            constant(value)
        return number

    value = json.loads(text, object_pairs_hook=unique, parse_constant=constant,
                       parse_float=finite)
    if not isinstance(value, dict):
        raise ValueError("Expected exactly one JSON object")
    return value


def numeric_box(location):
    if not isinstance(location, dict):
        raise ValueError("localization must be an object")
    box = location.get("bbox_xyxy")
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise ValueError("bbox_xyxy requires four finite numeric coordinates")
