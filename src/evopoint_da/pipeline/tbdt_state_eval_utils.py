"""Evaluate TBDT state-displacement samples by structural regions.

The primary metrics here are residue-region displacement metrics, not
full-chain RMSD. Inputs are processed ``.pt`` samples containing AF2-to-target
CA displacement targets and optional prediction ``.pt`` files containing
predicted displacement vectors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


TARGET_KEYS = ("y_delta", "target_delta", "target_displacement", "delta_r", "y")
PREDICTION_KEYS = (
    "pred_delta",
    "prediction_delta",
    "predicted_delta",
    "y_pred",
    "pred",
    "prediction",
    "delta",
    "displacement",
)
REGION_CONTAINER_KEYS = ("regions", "region_masks", "tbdt_regions", "tbdt_region_masks")
BARREL_CORE_KEYS = (
    "barrel_core_mask",
    "tbdt_barrel_core_mask",
    "core_mask",
    "barrel_core_indices",
    "tbdt_barrel_core_indices",
)
DIRECT_REGION_MASK_KEYS = {
    "barrel_core": ("barrel_core_mask",),
    "plug": ("plug_mask",),
    "plug_core": ("plug_core_mask",),
    "plug_apical_loop": ("plug_apical_loop_mask", "plug_apical_loops_mask"),
    "plug_extension_nt": ("plug_extension_nt_mask", "n_terminal_plug_extension_mask"),
    "extracellular_loop": ("extracellular_loop_mask",),
    "tonb_box": ("tonb_box_mask",),
    "substrate_contact": ("substrate_contact_mask",),
    "eval": ("eval_mask",),
}


def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    data: dict[str, Any] = {}
    for key in ("x", "pos", "y", "y_delta", "residue_ids", "plddt"):
        if hasattr(obj, key):
            data[key] = getattr(obj, key)
    return data


def _to_tensor(value: Any, *, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu()


def _vector_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = _to_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1 and tensor.numel() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.size(-1) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(tensor.shape)}")
    return tensor


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    metadata = mapping.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
    return None


def _extract_target(sample: dict[str, Any], sample_path: Path) -> torch.Tensor:
    value = _first_present(sample, TARGET_KEYS)
    if value is None and "af2_pos" in sample and "holo_pos" in sample:
        value = _to_tensor(sample["holo_pos"]) - _to_tensor(sample["af2_pos"])
    if value is None:
        raise ValueError(f"{sample_path} has no displacement target; expected one of {TARGET_KEYS}")
    return _vector_tensor(value, f"target in {sample_path}")


def _extract_prediction(pred_obj: Any, sample_stem: str, n: int) -> torch.Tensor:
    if pred_obj is None:
        return torch.zeros((n, 3), dtype=torch.float32)
    if isinstance(pred_obj, torch.Tensor):
        return _vector_tensor(pred_obj, f"prediction for {sample_stem}")

    pred = _as_mapping(pred_obj)
    if sample_stem in pred:
        return _extract_prediction(pred[sample_stem], sample_stem, n)
    value = _first_present(pred, PREDICTION_KEYS)
    if value is None:
        raise ValueError(f"Prediction for {sample_stem} has no vector key; expected one of {PREDICTION_KEYS}")
    tensor = _vector_tensor(value, f"prediction for {sample_stem}")
    if tensor.size(0) != n:
        raise ValueError(f"Prediction length mismatch for {sample_stem}: pred={tensor.size(0)}, target={n}")
    return tensor


def _collect_pt_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.pt")))
        elif path.is_file() and path.suffix == ".pt":
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path is not a .pt file or directory: {path}")
    deduped = sorted({p.resolve(): p for p in files}.values())
    if not deduped:
        raise FileNotFoundError("No processed .pt samples found.")
    return deduped


def _load_region_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("--region-json must contain a JSON object.")
    return payload


def _residue_id_lookup(sample: dict[str, Any]) -> dict[str, int]:
    residue_ids = sample.get("residue_ids")
    if residue_ids is None:
        return {}
    return {str(rid): i for i, rid in enumerate(list(residue_ids))}


def _mask_from_value(value: Any, n: int, residue_lookup: dict[str, int], region_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bool:
            mask = tensor.flatten()
        else:
            idx = tensor.flatten().long()
            mask = torch.zeros(n, dtype=torch.bool)
            if idx.numel():
                mask[idx] = True
        if mask.numel() != n:
            raise ValueError(f"Region {region_name} mask length {mask.numel()} != sample length {n}")
        return mask.bool()

    if isinstance(value, dict):
        for key in ("mask", "indices", "residue_ids", "residues"):
            if key in value:
                return _mask_from_value(value[key], n, residue_lookup, region_name)
        raise ValueError(f"Region {region_name} dict must contain mask, indices, residue_ids, or residues.")

    if isinstance(value, (list, tuple)):
        if len(value) == n and all(isinstance(v, bool) for v in value):
            return torch.tensor(value, dtype=torch.bool)
        mask = torch.zeros(n, dtype=torch.bool)
        if all(isinstance(v, str) for v in value):
            missing = [str(v) for v in value if str(v) not in residue_lookup]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(f"Region {region_name} references residue_ids not present in sample: {preview}")
            for rid in value:
                mask[residue_lookup[str(rid)]] = True
            return mask
        idx = torch.as_tensor(value, dtype=torch.long).flatten()
        if idx.numel():
            mask[idx] = True
        return mask

    raise TypeError(f"Unsupported region value for {region_name}: {type(value).__name__}")


def _extract_regions(
    sample: dict[str, Any],
    sample_stem: str,
    n: int,
    region_json: dict[str, Any],
    include_all: bool,
) -> dict[str, torch.Tensor]:
    residue_lookup = _residue_id_lookup(sample)
    raw_regions: dict[str, Any] = {}

    for key in REGION_CONTAINER_KEYS:
        value = _first_present(sample, (key,))
        if isinstance(value, dict):
            raw_regions.update(value)

    for key in BARREL_CORE_KEYS:
        value = _first_present(sample, (key,))
        if value is not None:
            raw_regions.setdefault("barrel_core", value)

    for region_name, keys in DIRECT_REGION_MASK_KEYS.items():
        value = _first_present(sample, keys)
        if value is not None:
            raw_regions.setdefault(region_name, value)

    for scope_key in ("*", sample_stem):
        scoped = region_json.get(scope_key)
        if isinstance(scoped, dict):
            raw_regions.update(scoped)

    regions = {
        str(name): _mask_from_value(value, n, residue_lookup, str(name))
        for name, value in raw_regions.items()
    }
    if include_all:
        regions.setdefault("all", torch.ones(n, dtype=torch.bool))
    return {name: mask for name, mask in regions.items() if bool(mask.any())}


def _derive_plug_regions(
    regions: dict[str, torch.Tensor],
    n: int,
    *,
    plug_apical_fraction: float = 0.35,
    plug_extension_residues: int = 12,
) -> None:
    plug_mask = regions.get("plug")
    if plug_mask is None or not bool(plug_mask.any()):
        return
    tonb_mask = regions.get("tonb_box", torch.zeros(n, dtype=torch.bool))
    non_tonb_plug = plug_mask & ~tonb_mask
    plug_idx = torch.nonzero(plug_mask, as_tuple=False).flatten()
    non_tonb_idx = torch.nonzero(non_tonb_plug, as_tuple=False).flatten()

    if "plug_extension_nt" not in regions:
        n_ext = max(1, min(int(plug_extension_residues), int(plug_idx.numel())))
        mask = torch.zeros(n, dtype=torch.bool)
        if tonb_mask.any():
            overlap = plug_mask & tonb_mask
            if bool(overlap.any()):
                mask = overlap
            else:
                mask[plug_idx[:n_ext]] = True
        else:
            mask[plug_idx[:n_ext]] = True
        regions["plug_extension_nt"] = mask

    if "plug_apical_loop" not in regions and non_tonb_idx.numel() > 0:
        frac = min(1.0, max(0.0, float(plug_apical_fraction)))
        n_apical = max(1, int(math.ceil(float(non_tonb_idx.numel()) * frac)))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[non_tonb_idx[-n_apical:]] = True
        regions["plug_apical_loop"] = mask

    if "plug_core" not in regions:
        excluded = torch.zeros(n, dtype=torch.bool)
        for name in ("tonb_box", "plug_apical_loop", "plug_extension_nt"):
            mask = regions.get(name)
            if mask is not None:
                excluded |= mask
        core = plug_mask & ~excluded
        if bool(core.any()):
            regions["plug_core"] = core


def _prediction_index(predictions: str | None, samples: list[Path]) -> tuple[dict[str, Path], Any | None]:
    if not predictions:
        return {}, None
    path = Path(predictions)
    if path.is_dir():
        return {p.stem: p for p in sorted(path.rglob("*.pt"))}, None
    if not path.is_file():
        raise FileNotFoundError(f"Prediction path does not exist: {path}")
    loaded = _load_pt(path)
    if len(samples) == 1:
        return {}, loaded
    if isinstance(loaded, dict) and any(sample.stem in loaded for sample in samples):
        return {}, loaded
    raise ValueError(
        "A single --predictions file with multiple samples must be a dict keyed by sample stem. "
        "Otherwise pass a prediction directory."
    )
