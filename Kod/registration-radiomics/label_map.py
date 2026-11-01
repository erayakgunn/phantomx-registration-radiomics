"""Resolve TotalSegmentator label numbers by organ name.

Label numbering differs between TotalSegmentator versions/custom models, so the
pipeline must map names (spleen, aorta, vertebrae, ribs, sternum) to IDs using
the per-series `label_map.json` instead of hard-coded numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def canonical_name(name: str) -> str:
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    match = re.match(r"rib (left|right) (\d+)$", text)
    if match:
        return f"rib_{match.group(1)}_{match.group(2)}"
    match = re.match(r"(left|right) rib (\d+)$", text)
    if match:
        return f"rib_{match.group(1)}_{match.group(2)}"
    return text


def load_label_map(seg_path: Path) -> dict[str, int]:
    path = Path(seg_path).parent / "label_map.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for key, name in raw.items():
        try:
            mapping[canonical_name(name)] = int(key)
        except (TypeError, ValueError):
            continue
    return mapping


def resolve_target_labels(seg_path: Path, names) -> list[int]:
    mapping = load_label_map(seg_path)
    return [mapping[name] for name in names if name in mapping]


def registration_labels(seg_path: Path) -> list[int]:
    mapping = load_label_map(seg_path)
    labels = []
    for name, label in mapping.items():
        if name.startswith("vertebrae") or name.startswith("rib_") or name == "sternum":
            labels.append(label)
    return sorted(set(labels))


def resolve_organs(seg_path: Path, organ_specs) -> list[tuple[int, str, float]]:
    """Resolve ("liver", 30.0) style specs to (label, name, size) triples."""
    mapping = load_label_map(seg_path)
    resolved = []
    for name, size in organ_specs:
        if name in mapping:
            resolved.append((mapping[name], name, float(size)))
    return resolved
