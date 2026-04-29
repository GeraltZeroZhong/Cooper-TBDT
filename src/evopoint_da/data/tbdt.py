from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evopoint_da.data.structure import format_residue_id, parse_residue_id

REGION_UNKNOWN = "unknown"
REGION_BARREL_CORE = "barrel_core"
REGION_PLUG = "plug"
REGION_EXTRACELLULAR_LOOP = "extracellular_loop"
REGION_TONB_BOX = "tonb_box"
REGION_SUBSTRATE_CONTACT = "substrate_contact"

REGION_VOCAB: dict[str, int] = {
    REGION_UNKNOWN: 0,
    REGION_BARREL_CORE: 1,
    REGION_PLUG: 2,
    REGION_EXTRACELLULAR_LOOP: 3,
    REGION_TONB_BOX: 4,
    REGION_SUBSTRATE_CONTACT: 5,
}

STATE_VOCAB: dict[str, int] = {
    "unknown": 0,
    "apo": 1,
    "metal_only": 2,
    "substrate_bound": 3,
    "productive_substrate_bound": 4,
    "tonb_bound": 5,
    "uncertain": 6,
}

SUBSTRATE_VOCAB: dict[str, int] = {
    "unknown": 0,
    "none": 1,
    "ferric_citrate": 2,
    "ferrichrome": 3,
    "enterobactin": 4,
    "cobalamin": 5,
    "siderophore": 6,
    "heme": 7,
    "unknown_siderophore": 8,
}

FAMILY_VOCAB: dict[str, int] = {
    "unknown": 0,
    "tbdt": 1,
    "feca": 2,
    "fhua": 3,
    "fepa": 4,
    "btub": 5,
    "cira": 6,
    "fyua": 7,
}

REGION_ALIASES: dict[str, str] = {
    "barrel": REGION_BARREL_CORE,
    "core": REGION_BARREL_CORE,
    "barrel_core": REGION_BARREL_CORE,
    "plug": REGION_PLUG,
    "cork": REGION_PLUG,
    "extracellular_loop": REGION_EXTRACELLULAR_LOOP,
    "extracellular_loops": REGION_EXTRACELLULAR_LOOP,
    "loop": REGION_EXTRACELLULAR_LOOP,
    "loops": REGION_EXTRACELLULAR_LOOP,
    "tonb": REGION_TONB_BOX,
    "tonb_box": REGION_TONB_BOX,
    "substrate_contact": REGION_SUBSTRATE_CONTACT,
    "substrate_contacts": REGION_SUBSTRATE_CONTACT,
    "contact": REGION_SUBSTRATE_CONTACT,
    "contacts": REGION_SUBSTRATE_CONTACT,
}

REGION_MASK_KEYS: dict[str, str] = {
    REGION_BARREL_CORE: "barrel_core_mask",
    REGION_PLUG: "plug_mask",
    REGION_EXTRACELLULAR_LOOP: "extracellular_loop_mask",
    REGION_TONB_BOX: "tonb_box_mask",
    REGION_SUBSTRATE_CONTACT: "substrate_contact_mask",
}

DEFAULT_REGION_LOSS_WEIGHTS: dict[str, float] = {
    REGION_UNKNOWN: 1.0,
    REGION_BARREL_CORE: 0.1,
    REGION_PLUG: 2.0,
    REGION_EXTRACELLULAR_LOOP: 2.0,
    REGION_TONB_BOX: 3.0,
    REGION_SUBSTRATE_CONTACT: 3.0,
}

DEFAULT_EVAL_REGIONS = (
    REGION_PLUG,
    REGION_EXTRACELLULAR_LOOP,
    REGION_TONB_BOX,
    REGION_SUBSTRATE_CONTACT,
)


def normalize_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return label or default


def vocab_id(vocab: dict[str, int], value: Any, *, default: str = "unknown") -> int:
    return int(vocab.get(normalize_label(value, default=default), vocab[default]))


def state_id(value: Any) -> int:
    return vocab_id(STATE_VOCAB, value)


def substrate_id(value: Any) -> int:
    return vocab_id(SUBSTRATE_VOCAB, value)


def family_id(value: Any) -> int:
    return vocab_id(FAMILY_VOCAB, value)


def region_id(value: Any) -> int:
    return vocab_id(REGION_VOCAB, normalize_region_name(value))


def normalize_region_name(value: Any) -> str:
    label = normalize_label(value)
    return REGION_ALIASES.get(label, label if label in REGION_VOCAB else REGION_UNKNOWN)


def load_region_annotation(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {}
    with open(path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    if not isinstance(annotation, dict):
        raise ValueError(f"TBDT region annotation must be a JSON object: {path}")
    return annotation


def _entry_list(raw_region: Any) -> list[dict[str, Any]]:
    if raw_region is None:
        return []
    if isinstance(raw_region, dict):
        return [raw_region]
    if isinstance(raw_region, list):
        entries: list[dict[str, Any]] = []
        for item in raw_region:
            if isinstance(item, dict):
                entries.append(item)
            else:
                entries.append({"residues": [item]})
        return entries
    return [{"residues": [raw_region]}]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _range_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 2 and not any(isinstance(item, (list, tuple)) for item in value):
            return [value]
        return value
    return [value]


def _as_int_residue(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text[0] == "-":
        sign = -1
        text = text[1:]
    else:
        sign = 1
    digits = []
    for char in text:
        if not char.isdigit():
            break
        digits.append(char)
    return sign * int("".join(digits)) if digits else None


def _expand_range(range_value: Any) -> list[int]:
    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2:
        return []
    start = _as_int_residue(range_value[0])
    end = _as_int_residue(range_value[1])
    if start is None or end is None:
        return []
    step = 1 if end >= start else -1
    return list(range(start, end + step, step))


def _entry_residue_ids(entry: dict[str, Any], default_chain: str | None) -> set[str]:
    chain = str(entry.get("chain") or default_chain or "").strip()
    residue_ids: set[str] = {str(rid).strip() for rid in _as_list(entry.get("residue_ids")) if str(rid).strip()}

    for residue in _as_list(entry.get("residues")):
        if isinstance(residue, str) and "_" in residue:
            residue_ids.add(residue.strip())
            continue
        resseq = _as_int_residue(residue)
        if resseq is not None and chain:
            residue_ids.add(format_residue_id(chain, resseq))

    for range_value in _range_list(entry.get("ranges")):
        if isinstance(range_value, str) and "-" in range_value:
            start, end = range_value.split("-", 1)
            range_value = [start, end]
        for resseq in _expand_range(range_value):
            if chain:
                residue_ids.add(format_residue_id(chain, resseq))

    return residue_ids


def get_region_residue_ids(
    annotation: dict[str, Any],
    region_name: str,
    *,
    default_chain: str | None = None,
) -> set[str]:
    region_key = normalize_region_name(region_name)
    regions = annotation.get("regions", {})
    if not isinstance(regions, dict):
        return set()

    residue_ids: set[str] = set()
    for raw_name, raw_region in regions.items():
        if normalize_region_name(raw_name) != region_key:
            continue
        for entry in _entry_list(raw_region):
            residue_ids.update(_entry_residue_ids(entry, default_chain or annotation.get("default_chain")))
    return residue_ids


def _annotation_region_sets(annotation: dict[str, Any], default_chain: str | None) -> dict[str, set[str]]:
    region_sets = {name: set() for name in REGION_MASK_KEYS}
    regions = annotation.get("regions", {})
    if not isinstance(regions, dict):
        return region_sets

    chain = default_chain or annotation.get("default_chain")
    for raw_name, raw_region in regions.items():
        name = normalize_region_name(raw_name)
        if name not in region_sets:
            continue
        for entry in _entry_list(raw_region):
            region_sets[name].update(_entry_residue_ids(entry, chain))
    return region_sets


def _annotation_weight_map(annotation: dict[str, Any]) -> dict[str, float]:
    raw_weights = annotation.get("loss_weights", annotation.get("eval_weights", {}))
    weights = dict(DEFAULT_REGION_LOSS_WEIGHTS)
    if isinstance(raw_weights, dict):
        for raw_name, raw_value in raw_weights.items():
            name = normalize_region_name(raw_name)
            if name in weights:
                weights[name] = float(raw_value)
            elif normalize_label(raw_name) == "default":
                weights[REGION_UNKNOWN] = float(raw_value)
    return weights


def _annotation_eval_regions(annotation: dict[str, Any]) -> tuple[str, ...]:
    raw_eval_regions = annotation.get("eval_regions")
    if not raw_eval_regions:
        return DEFAULT_EVAL_REGIONS
    return tuple(
        name
        for name in (normalize_region_name(value) for value in _as_list(raw_eval_regions))
        if name in REGION_MASK_KEYS
    )


def build_region_features(
    residue_ids: list[str],
    annotation: dict[str, Any] | None,
    *,
    default_chain: str | None = None,
) -> dict[str, np.ndarray]:
    annotation = annotation or {}
    n_residues = len(residue_ids)
    region_sets = _annotation_region_sets(annotation, default_chain)

    masks = {
        REGION_MASK_KEYS[name]: np.asarray([rid in ids for rid in residue_ids], dtype=bool)
        for name, ids in region_sets.items()
    }

    primary_region_id = np.full(n_residues, REGION_VOCAB[REGION_UNKNOWN], dtype=np.int64)
    for name in (
        REGION_BARREL_CORE,
        REGION_SUBSTRATE_CONTACT,
        REGION_EXTRACELLULAR_LOOP,
        REGION_PLUG,
        REGION_TONB_BOX,
    ):
        primary_region_id[masks[REGION_MASK_KEYS[name]]] = REGION_VOCAB[name]

    weights = _annotation_weight_map(annotation)
    loss_weight = np.full(n_residues, weights[REGION_UNKNOWN], dtype=np.float32)
    for name in (
        REGION_BARREL_CORE,
        REGION_EXTRACELLULAR_LOOP,
        REGION_PLUG,
        REGION_TONB_BOX,
        REGION_SUBSTRATE_CONTACT,
    ):
        loss_weight[masks[REGION_MASK_KEYS[name]]] = weights[name]

    eval_mask = np.zeros(n_residues, dtype=bool)
    for name in _annotation_eval_regions(annotation):
        eval_mask |= masks[REGION_MASK_KEYS[name]]

    raw_eval_residue_ids = annotation.get("eval_residue_ids", [])
    if raw_eval_residue_ids:
        eval_residue_ids = {str(rid).strip() for rid in _as_list(raw_eval_residue_ids) if str(rid).strip()}
        eval_mask |= np.asarray([rid in eval_residue_ids for rid in residue_ids], dtype=bool)

    if not any(mask.any() for mask in masks.values()):
        eval_mask[:] = True

    return {
        "region_id": primary_region_id,
        **masks,
        "eval_mask": eval_mask,
        "loss_weight": loss_weight,
    }


def residue_number_index(residue_ids: list[str]) -> dict[tuple[str, int], int]:
    index: dict[tuple[str, int], int] = {}
    for i, rid in enumerate(residue_ids):
        chain, resseq, _icode = parse_residue_id(rid)
        index[(chain, resseq)] = i
    return index
