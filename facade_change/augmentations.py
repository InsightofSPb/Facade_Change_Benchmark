"""Deterministic same-image state edits and independent observation nuisances.

All outputs keep the native pixel grid. Procedural edits label their actual
changed RGB bytes, not real facade damage. Photometric operations explicitly
use linear-light sRGB; JPEG intentionally uses the ordinary encoded sRGB image.
"""
from __future__ import annotations

import hashlib
import math
import re
from io import BytesIO

import numpy as np
from PIL import Image


def _inputs(rgb, support):
    rgb, support = np.asarray(rgb), np.asarray(support)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 8:
        raise ValueError("Expected native uint8 RGB with both dimensions >=8; no resize is performed")
    if support.dtype != np.bool_ or support.shape != rgb.shape[:2]:
        raise ValueError("Expected a same-grid boolean support mask")
    if not support.any():
        raise ValueError("At least one supported original pixel is required")
    return rgb, support


def _linear(rgb):
    encoded = rgb.astype(np.float64) / 255
    return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)


def _encoded(linear):
    linear = np.clip(linear, 0, 1)
    values = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    return np.clip(np.rint(values * 255), 0, 255).astype(np.uint8)


def _boundary(region):
    import cv2
    inner = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return region & ~inner


def _patch_alpha(shape):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    x, y = (xx + .5) / w, (yy + .5) / h
    distance = np.sqrt(((x - .52) / .15) ** 2 + ((y - .55) / .13) ** 2)
    return np.clip((1.04 - distance) / .08, 0, 1)


def render_state(rgb, support, state_id):
    """Return H0, a fixed H1 proxy, or the bit-identical self-paste control."""
    import cv2
    rgb, support = _inputs(rgb, support)
    if not isinstance(state_id, str) or state_id not in {"unchanged", "crack", "paint_patch", "self_paste"}:
        raise ValueError(f"Unknown deterministic state: {state_id}")
    output = rgb.copy()
    region = np.zeros_like(support)
    parameters = {"state_id": state_id, "coordinate_grid": "unchanged native RGB",
                  "label_scope": "actual pre-nuisance changed bytes of a procedural proxy; not real damage GT"}
    if state_id == "crack":
        h, w = support.shape
        points = np.array([[.42, .18], [.48, .30], [.44, .42], [.54, .53], [.48, .66], [.57, .82]])
        vertices = np.rint(points * [w - 1, h - 1]).astype(np.int32)
        width = max(1, int(round(min(h, w) * .006)))
        template = np.zeros_like(support, dtype=np.uint8)
        cv2.polylines(template, [vertices], isClosed=False, color=255, thickness=width, lineType=cv2.LINE_AA)
        alpha = template.astype(np.float64) / 255
        region = alpha > 0
        candidate = _encoded(_linear(rgb) * (1 - .88 * alpha[..., None]))
        output[support] = candidate[support]
        parameters.update(proxy="fixed thin dark polyline", vertices_normalized=points.tolist(),
                          width_pixels=width, attenuation_fraction=.88, compositing="linear-light sRGB, antialiased mask")
    elif state_id in {"paint_patch", "self_paste"}:
        alpha = _patch_alpha(support.shape)
        region = alpha > 0
        parameters.update(region="fixed ellipse with soft edge", center_normalized=[.52, .55],
                          radii_normalized=[.15, .13], edge_relative_radius=.08)
        if state_id == "paint_patch":
            color = np.array([172, 164, 151], dtype=np.uint8)
            blend = .72 * alpha[..., None]
            candidate = _encoded(_linear(rgb) * (1 - blend) + _linear(color) * blend)
            output[support] = candidate[support]
            parameters.update(proxy="neutral paint/repair-like patch", color_rgb=color.tolist(),
                              blend_alpha=.72, compositing="linear-light sRGB")
        else:
            # Use the same region/insertion path, copying exactly the same pixels.
            visible = region & support
            output[visible] = rgb[visible].copy()
            parameters.update(proxy="same-image self-paste sham", compositing="exact uint8 copy; no interpolation")
    changed = np.any(output != rgb, axis=2) & support
    if state_id in {"crack", "paint_patch"} and not changed.any():
        raise ValueError(f"State {state_id} has no visible supported RGB edit")
    parameters["actual_changed_pixels"] = int(changed.sum())
    return {"rgb": output, "edit_mask": changed, "insertion_boundary": _boundary(region) & support,
            "parameters": parameters}


def _number(value, name, low, high, *, lower_open=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (value <= low if lower_open else value < low) or value > high:
        raise ValueError(f"{name} outside its permitted range")
    return value


def _spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Each nuisance scenario must be an object")
    identifier = spec.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Scenario id must be a nonempty filename-safe string")
    kind = spec.get("kind")
    fields = {"identity": set(), "shadow": {"template", "strength", "edge_width"}, "exposure": {"gain"},
              "contrast": {"factor", "center"}, "white_balance": {"gains"}, "blur": {"sigma"},
              "jpeg": {"quality"}, "occlusion": {"rectangle", "color"}}
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError(f"Unknown nuisance kind: {kind}")
    unexpected = set(spec) - {"id", "kind"} - fields[kind]
    if unexpected:
        raise ValueError(f"Unexpected {kind} parameters: {sorted(unexpected)}")
    normalized = {"id": identifier, "kind": kind}
    if kind == "shadow":
        template = spec.get("template", "diagonal")
        if template not in {"diagonal", "band"}:
            raise ValueError("Shadow template must be diagonal or band")
        normalized.update(template=template, strength=_number(spec.get("strength", .35), "shadow strength", 0, .99, lower_open=True),
                          edge_width=_number(spec.get("edge_width", .035), "edge_width", 0, .25, lower_open=True))
    elif kind == "exposure":
        normalized["gain"] = _number(spec.get("gain", .75), "exposure gain", 0, 4, lower_open=True)
    elif kind == "contrast":
        normalized.update(factor=_number(spec.get("factor", .8), "contrast factor", 0, 4, lower_open=True),
                          center=_number(spec.get("center", .18), "linear contrast center", 0, 1))
    elif kind == "white_balance":
        gains = spec.get("gains", [1.12, 1., .88])
        if not isinstance(gains, (list, tuple)) or len(gains) != 3:
            raise ValueError("White balance requires three RGB gains")
        normalized["gains"] = [_number(value, "white balance gain", 0, 4, lower_open=True) for value in gains]
    elif kind == "blur":
        normalized["sigma"] = _number(spec.get("sigma", .8), "Gaussian sigma", 0, 8, lower_open=True)
    elif kind == "jpeg":
        quality = spec.get("quality", 70)
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be an integer in [1,100]")
        normalized["quality"] = quality
    elif kind == "occlusion":
        rectangle, color = spec.get("rectangle", [.68, .25, .88, .8]), spec.get("color", [62, 68, 58])
        if not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4:
            raise ValueError("Occlusion rectangle requires normalized [x0,y0,x1,y1]")
        rectangle = [_number(value, "rectangle coordinate", 0, 1) for value in rectangle]
        if not rectangle[0] < rectangle[2] or not rectangle[1] < rectangle[3]:
            raise ValueError("Occlusion rectangle must have positive width and height")
        if not isinstance(color, (list, tuple)) or len(color) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in color):
            raise ValueError("Occlusion color requires three uint8 RGB integers")
        normalized.update(rectangle=rectangle, color=list(color))
    return normalized


def validate_scenarios(scenarios):
    """Validate a frozen nuisance list and return normalized independent specs."""
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenarios must be a nonempty list")
    normalized = [_spec(spec) for spec in scenarios]
    if len({spec["id"] for spec in normalized}) != len(normalized):
        raise ValueError("Nuisance scenario ids must be unique")
    return normalized


def _filled_supported(rgb, support):
    """Nearest supported colors extend holes for filters; original holes stay intact."""
    if support.all():
        return rgb.copy()
    import cv2
    _, labels = cv2.distanceTransformWithLabels((~support).astype(np.uint8), cv2.DIST_L2, 5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    colors = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[labels[support]] = rgb[support]
    output = rgb.copy()
    output[~support] = colors[labels[~support]]
    return output


def _shadow(shape, template, edge_width):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    x, y = (xx + .5) / w, (yy + .5) / h
    if template == "diagonal":
        distance = (x - .30 - .35 * y) / math.sqrt(1 + .35 ** 2)
        alpha = np.clip(.5 + distance / edge_width, 0, 1)
    else:
        distance = np.abs(x - .55 - .15 * (y - .5)) / math.sqrt(1 + .15 ** 2)
        alpha = np.clip((.14 + edge_width / 2 - distance) / edge_width, 0, 1)
    return alpha.astype(np.float32)


def apply_nuisance(rgb, support, spec):
    """Apply one fresh observation transform; no geometry or random changes.

    nuisance_mask is the known applied footprint, independent of byte rounding.
    actual_change_mask separately reports RGB bytes that changed. Only known
    occlusion changes support/true_visibility; shadows remain visible pixels.
    """
    import cv2
    rgb, support = _inputs(rgb, support)
    spec = _spec(spec)
    kind = spec["kind"]
    output, source_support = rgb.copy(), support.copy()
    true_visibility = support.copy()
    footprint = np.zeros_like(support) if kind == "identity" else support.copy()
    alpha = np.zeros_like(support, dtype=np.float32)
    parameters = {**spec, "coordinate_grid": "unchanged native RGB", "geometry_transform": "identity",
                  "nuisance_mask_scope": "known applied footprint; actual changed RGB bytes are recorded separately"}
    if kind == "shadow":
        alpha = _shadow(support.shape, spec["template"], spec["edge_width"])
        alpha[~support] = 0
        footprint = (alpha > 0) & support
        candidate = _encoded(_linear(rgb) * (1 - spec["strength"] * alpha[..., None]))
        output[support] = candidate[support]
        parameters.update(color_space="linear-light sRGB", formula="linear_RGB * (1 - strength * fixed_alpha)",
                          template_parameters={"diagonal": "x - .30 - .35*y", "band": "abs(x - .55 - .15*(y-.5)); half-width .14"}[spec["template"]])
    elif kind in {"exposure", "contrast", "white_balance"}:
        linear = _linear(rgb)
        if kind == "exposure":
            linear *= spec["gain"]
        elif kind == "contrast":
            linear = spec["center"] + spec["factor"] * (linear - spec["center"])
        else:
            linear *= np.array(spec["gains"])
        candidate = _encoded(linear)
        output[support] = candidate[support]
        parameters.update(color_space="linear-light sRGB", clipping="linear [0,1], encoded uint8 round-to-nearest")
    elif kind == "blur":
        kernel = 2 * int(math.ceil(3 * spec["sigma"])) + 1
        filled = _linear(_filled_supported(rgb, support)).astype(np.float32)
        blurred = cv2.GaussianBlur(filled, (kernel, kernel), spec["sigma"], sigmaY=spec["sigma"], borderType=cv2.BORDER_REFLECT_101)
        candidate = _encoded(blurred)
        output[support] = candidate[support]
        parameters.update(color_space="linear-light sRGB", kernel_size=[kernel, kernel], border="reflect101",
                          invalid_pixel_handling="nearest supported color before filter; original unsupported bytes restored")
    elif kind == "jpeg":
        buffer = BytesIO()
        Image.fromarray(_filled_supported(rgb, support)).save(buffer, format="JPEG", quality=spec["quality"],
                                                             subsampling=2, optimize=False, progressive=False)
        encoded = buffer.getvalue()
        with Image.open(BytesIO(encoded)) as image:
            candidate = np.array(image.convert("RGB"))
        output[support] = candidate[support]
        parameters.update(color_space="encoded sRGB", codec="Pillow JPEG encode/decode", subsampling="4:2:0",
                          encoded_jpeg_sha256=hashlib.sha256(encoded).hexdigest(), encoded_jpeg_bytes=len(encoded),
                          invalid_pixel_handling="nearest supported color before codec; original unsupported bytes restored")
    elif kind == "occlusion":
        h, w = support.shape
        yy, xx = np.ogrid[:h, :w]
        x, y = (xx + .5) / w, (yy + .5) / h
        x0, y0, x1, y1 = spec["rectangle"]
        rectangle = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        footprint = rectangle & support
        output[footprint] = spec["color"]
        source_support[rectangle] = False
        true_visibility[rectangle] = False
        alpha[footprint] = 1
        parameters.update(proxy="known fixed opaque rectangle; not a realistic object segmentation", color_space="encoded sRGB")
    actual = np.any(output != rgb, axis=2) & support
    parameters.update(actual_changed_pixels=int(actual.sum()), footprint_pixels=int(footprint.sum()))
    result = {"rgb": output, "support": source_support, "nuisance_mask": footprint,
              "actual_change_mask": actual, "true_visibility": true_visibility,
              "alpha": alpha, "parameters": parameters}
    if kind in {"identity", "shadow", "occlusion"}:
        result["intensity"] = alpha * spec["strength"] if kind == "shadow" else alpha.copy()
    return result
