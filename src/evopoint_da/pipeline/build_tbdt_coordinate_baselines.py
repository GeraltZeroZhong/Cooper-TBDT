"""Build interpretable coordinate baselines for TBDT displacement prediction.

These baselines deliberately use only training-like donor splits, then export
ordinary per-residue displacement prediction files so the standard
``eval_tbdt_state`` region endpoint can score them without special handling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evopoint_da.data.dataset import build_split_file_lists
from evopoint_da.pipeline.tbdt_state_eval_core import evaluate as evaluate_regions
from evopoint_da.pipeline.tbdt_state_eval_metrics import _json_safe
from evopoint_da.pipeline.tbdt_state_eval_utils import (
    _as_mapping,
    _derive_plug_regions,
    _extract_regions,
    _extract_target,
    _first_present,
    _load_pt,
    _vector_tensor,
)


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}

GROUPINGS: tuple[tuple[str, ...], ...] = (
    (),
    ("family",),
    ("state",),
    ("family", "state"),
)
REGION_MEAN_BASELINES: dict[str, tuple[str, ...]] = {
    "global_region_mean": (),
    "family_region_mean": ("family",),
    "state_region_mean": ("state",),
    "family_state_region_mean": ("family", "state"),
}
RIGID_REGION_BASELINES: dict[str, tuple[str, ...]] = {
    "region_centroid_shift": (),
    "family_region_centroid_shift": ("family",),
    "state_region_centroid_shift": ("state",),
    "family_state_region_centroid_shift": ("family", "state"),
}
TARGETED_RIGID_BASELINES: dict[str, str] = {
    "plug_rigid_shift": "plug",
    "plug_apical_loop_rigid_shift": "plug_apical_loop",
    "tonb_box_rigid_shift": "tonb_box",
}
LINEAR_BASELINES = {"barrel_frame_ridge"}
ALL_BASELINES = tuple(
    [
        *REGION_MEAN_BASELINES,
        *RIGID_REGION_BASELINES,
        *TARGETED_RIGID_BASELINES,
        *sorted(LINEAR_BASELINES),
    ]
)
DEFAULT_REGION_PRIORITY = (
    "tonb_box",
    "substrate_contact",
    "plug_extension_nt",
    "plug_apical_loop",
    "plug_core",
    "plug",
    "extracellular_loop",
    "barrel_core",
    "eval",
    "all",
)
LINEAR_REGION_FEATURES = (
    "barrel_core",
    "plug",
    "plug_core",
    "plug_apical_loop",
    "plug_extension_nt",
    "extracellular_loop",
    "tonb_box",
    "substrate_contact",
    "eval",
)
CONTINUOUS_FEATURE_NAMES = (
    "plddt_scaled",
    "sasa_scaled",
    "rsa",
    "residue_depth",
    "coordination_number",
    "hse_up",
    "hse_down",
    "dssp_helix",
    "dssp_sheet",
    "dssp_coil",
    "residue_position",
    "radial_distance",
    "axial_coordinate",
    "abs_axial_coordinate",
    "distance_to_plug_centroid",
    "distance_to_extracellular_centroid",
    "distance_to_barrel_center",
    "has_plug_centroid",
    "has_extracellular_centroid",
)
EPS = 1e-8


@dataclass(frozen=True)
class Sample:
    path: Path
    stem: str
    raw: dict[str, Any]
    pos: torch.Tensor
    target: torch.Tensor
    plddt: torch.Tensor
    x: torch.Tensor | None
    regions: dict[str, torch.Tensor]
    family: str
    state: str
    substrate: str


@dataclass
class VectorStat:
    total: torch.Tensor
    count: int = 0

    def update(self, value: torch.Tensor, weight: int = 1) -> None:
        self.total += value.float() * int(weight)
        self.count += int(weight)

    def mean(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.zeros(3, dtype=torch.float32)
        return self.total / float(self.count)


@dataclass(frozen=True)
class BarrelFrame:
    center: torch.Tensor
    axis: torch.Tensor
    radial_unit: torch.Tensor
    tangential_unit: torch.Tensor
    radial_distance: torch.Tensor
    axial_coordinate: torch.Tensor
    distance_to_plug: torch.Tensor
    distance_to_extracellular: torch.Tensor
    has_plug: torch.Tensor
    has_extracellular: torch.Tensor


@dataclass(frozen=True)
class LinearSchema:
    family_labels: tuple[str, ...]
    state_labels: tuple[str, ...]
    substrate_labels: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RidgeModel:
    schema: LinearSchema
    mean: torch.Tensor
    std: torch.Tensor
    weights: torch.Tensor
    alpha: float
    n_train_residues: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interpretable TBDT coordinate baselines.")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--donor-split", action="append", default=[], choices=["train", "val", "test", "all"])
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--baseline", action="append", default=[], choices=ALL_BASELINES)
    parser.add_argument("--output-root", default="artifacts/tbdt_v1/coordinate_baselines")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--region-priority", default=",".join(DEFAULT_REGION_PRIORITY))
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--max-prediction-norm",
        type=float,
        default=0.0,
        help="Optional per-residue clipping norm in Angstrom. Zero disables clipping.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--include-all-region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass-through setting for eval_tbdt_state and training region extraction.",
    )
    parser.add_argument(
        "--add-derived-regions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive plug_core, plug_apical_loop, and plug_extension_nt when explicit masks are absent.",
    )
    parser.add_argument("--plug-apical-fraction", type=float, default=0.35)
    parser.add_argument("--plug-extension-residues", type=int, default=12)
    parser.add_argument("--direction-threshold", type=float, default=1.0)
    parser.add_argument("--tonb-exposure-threshold", type=float, default=1.0)
    parser.add_argument("--bootstrap-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _normalize_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return label or default


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    row = _metadata(sample).get("manifest_row", {})
    return row if isinstance(row, dict) else {}


def _field(sample: dict[str, Any], keys: tuple[str, ...], fallback_id_key: str) -> str:
    row = _manifest_row(sample)
    metadata = _metadata(sample)
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return _normalize_label(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return _normalize_label(value)
        value = sample.get(key)
        if value not in (None, ""):
            return _normalize_label(value)
    fallback = sample.get(fallback_id_key)
    if fallback is not None:
        try:
            tensor = torch.as_tensor(fallback).view(-1)
            if tensor.numel():
                return f"id_{int(tensor[0].item())}"
        except Exception:
            return _normalize_label(fallback)
    return "unknown"


def _load_sample(path: Path, args: argparse.Namespace) -> Sample:
    raw = _as_mapping(_load_pt(path))
    target = _extract_target(raw, path)
    n = int(target.size(0))
    pos_value = _first_present(raw, ("pos", "af2_pos"))
    if pos_value is None:
        raise ValueError(f"{path} has no AF2/start coordinates; expected pos or af2_pos.")
    pos = _vector_tensor(pos_value, f"pos in {path}")
    if pos.size(0) != n:
        raise ValueError(f"{path} coordinate length {pos.size(0)} != target length {n}")

    plddt_value = _first_present(raw, ("plddt",))
    if plddt_value is None:
        plddt = torch.zeros(n, dtype=torch.float32)
    else:
        plddt = torch.as_tensor(plddt_value, dtype=torch.float32).reshape(-1)
        if plddt.numel() == 1:
            plddt = plddt.expand(n).clone()
        elif plddt.numel() != n:
            plddt = torch.zeros(n, dtype=torch.float32)

    x_value = _first_present(raw, ("x",))
    x = None
    if x_value is not None:
        x_tensor = torch.as_tensor(x_value, dtype=torch.float32)
        if x_tensor.dim() == 2 and x_tensor.size(0) == n:
            x = x_tensor

    regions = _extract_regions(raw, path.stem, n, {}, bool(args.include_all_region))
    if bool(args.add_derived_regions):
        _derive_plug_regions(
            regions,
            n,
            plug_apical_fraction=float(args.plug_apical_fraction),
            plug_extension_residues=int(args.plug_extension_residues),
        )
        regions = {name: mask for name, mask in regions.items() if bool(mask.any())}

    return Sample(
        path=path,
        stem=path.stem,
        raw=raw,
        pos=pos.float(),
        target=target.float(),
        plddt=plddt.float(),
        x=x,
        regions={name: mask.bool() for name, mask in regions.items()},
        family=_field(raw, ("family",), "family_id"),
        state=_field(raw, ("state_label", "state"), "state_id"),
        substrate=_field(raw, ("substrate_class", "substrate"), "substrate_id"),
    )


def _split_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    splits = build_split_file_lists(
        args.data_dir,
        DEFAULT_SPLIT_RANGES,
        int(args.split_seed),
        split_source=args.split_source,
    )
    target_paths = [Path(path) for path in splits.get(args.split, [])]
    donor_splits = args.donor_split or ["train", "val"]
    donor_paths = sorted({Path(path) for split in donor_splits for path in splits.get(split, [])})
    if not target_paths:
        raise FileNotFoundError(f"No target samples found for split={args.split!r} in {args.data_dir}")
    if not donor_paths:
        raise FileNotFoundError(f"No donor samples found for donor splits={donor_splits!r} in {args.data_dir}")
    return target_paths, donor_paths


def _group_values(sample: Sample, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(getattr(sample, field) for field in fields)


def _fallback_group_specs(sample: Sample, fields: tuple[str, ...]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    if fields == ("family", "state"):
        return [
            (("family", "state"), _group_values(sample, ("family", "state"))),
            (("family",), _group_values(sample, ("family",))),
            (("state",), _group_values(sample, ("state",))),
            ((), ()),
        ]
    if fields == ("family",):
        return [(("family",), _group_values(sample, ("family",))), ((), ())]
    if fields == ("state",):
        return [(("state",), _group_values(sample, ("state",))), ((), ())]
    return [((), ())]


def _region_order(regions: dict[str, torch.Tensor], priority_text: str) -> list[str]:
    priority = [item.strip() for item in str(priority_text).split(",") if item.strip()]
    seen: set[str] = set()
    ordered = []
    for name in priority:
        if name in regions and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(regions):
        if name not in seen and name != "all":
            ordered.append(name)
            seen.add(name)
    if "all" in regions and "all" not in seen:
        ordered.append("all")
    return ordered


def _new_stat() -> VectorStat:
    return VectorStat(total=torch.zeros(3, dtype=torch.float32), count=0)


def _fit_region_stats(samples: list[Sample], *, sample_weighted: bool) -> dict[tuple[tuple[str, ...], tuple[str, ...], str], VectorStat]:
    stats: dict[tuple[tuple[str, ...], tuple[str, ...], str], VectorStat] = defaultdict(_new_stat)
    for sample in samples:
        for region_name, mask in sample.regions.items():
            if not bool(mask.any()):
                continue
            vectors = sample.target[mask]
            value = vectors.mean(dim=0)
            weight = 1 if sample_weighted else int(mask.sum().item())
            for fields in GROUPINGS:
                key = (fields, _group_values(sample, fields), region_name)
                stats[key].update(value, weight=weight)
    return stats


def _lookup_region_stat(
    stats: dict[tuple[tuple[str, ...], tuple[str, ...], str], VectorStat],
    sample: Sample,
    fields: tuple[str, ...],
    region_name: str,
) -> tuple[torch.Tensor, str, int]:
    for fallback_fields, values in _fallback_group_specs(sample, fields):
        key = (fallback_fields, values, region_name)
        stat = stats.get(key)
        if stat is not None and stat.count > 0:
            label = "/".join(f"{name}={value}" for name, value in zip(fallback_fields, values, strict=False))
            return stat.mean(), label or "global", stat.count
    stat = stats.get(((), (), "all"))
    if stat is not None and stat.count > 0:
        return stat.mean(), "global_all", stat.count
    return torch.zeros(3, dtype=torch.float32), "zero", 0


def _assign_region_predictions(
    sample: Sample,
    *,
    stats: dict[tuple[tuple[str, ...], tuple[str, ...], str], VectorStat],
    fields: tuple[str, ...],
    priority_text: str,
    only_region: str | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    prediction = torch.zeros_like(sample.target)
    assigned = torch.zeros(sample.target.size(0), dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    for region_name in _region_order(sample.regions, priority_text):
        if only_region is not None and region_name != only_region:
            continue
        mask = sample.regions[region_name] & ~assigned
        if not bool(mask.any()):
            continue
        vector, fallback, donor_count = _lookup_region_stat(stats, sample, fields, region_name)
        prediction[mask] = vector.view(1, 3)
        assigned[mask] = True
        rows.append(
            {
                "sample_id": sample.stem,
                "region": region_name,
                "assigned_residues": int(mask.sum().item()),
                "fallback": fallback,
                "donor_count": donor_count,
                "vector_x": float(vector[0].item()),
                "vector_y": float(vector[1].item()),
                "vector_z": float(vector[2].item()),
            }
        )
    return prediction, rows


def _normalize_vector(vector: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
    vector = vector.float()
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= EPS:
        if fallback is None:
            fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        return fallback.float() / torch.linalg.vector_norm(fallback.float()).clamp_min(EPS)
    return vector / norm


def _first_region_centroid(sample: Sample, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        mask = sample.regions.get(name)
        if mask is not None and bool(mask.any()):
            return sample.pos[mask].mean(dim=0)
    return None


def _principal_axis(sample: Sample, center: torch.Tensor) -> torch.Tensor:
    mask = sample.regions.get("barrel_core")
    points = sample.pos[mask] if mask is not None and int(mask.sum().item()) >= 3 else sample.pos
    if points.size(0) >= 2:
        centered = points - points.mean(dim=0, keepdim=True)
        if float(torch.linalg.vector_norm(centered).item()) > EPS:
            try:
                _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
                axis = vh[0].float()
            except RuntimeError:
                axis = points[-1] - points[0]
        else:
            axis = points[-1] - points[0]
    else:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    axis = _normalize_vector(axis, fallback=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
    positive = _first_region_centroid(sample, ("extracellular_loop", "plug_apical_loop"))
    negative = _first_region_centroid(sample, ("tonb_box", "plug_extension_nt"))
    if positive is not None and negative is not None:
        if float(torch.dot(positive - negative, axis).item()) < 0.0:
            axis = -axis
    elif positive is not None:
        if float(torch.dot(positive - center, axis).item()) < 0.0:
            axis = -axis
    elif sample.pos.size(0) >= 2 and float(torch.dot(sample.pos[-1] - sample.pos[0], axis).item()) < 0.0:
        axis = -axis
    return axis


def _barrel_frame(sample: Sample) -> BarrelFrame:
    barrel_mask = sample.regions.get("barrel_core")
    if barrel_mask is not None and bool(barrel_mask.any()):
        center = sample.pos[barrel_mask].mean(dim=0)
    else:
        center = sample.pos.mean(dim=0)
    axis = _principal_axis(sample, center)
    offsets = sample.pos - center.view(1, 3)
    axial_coordinate = offsets @ axis
    radial = offsets - axial_coordinate.view(-1, 1) * axis.view(1, 3)
    radial_distance = torch.linalg.vector_norm(radial, dim=-1)

    fallback = torch.cross(axis, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), dim=0)
    if float(torch.linalg.vector_norm(fallback).item()) <= EPS:
        fallback = torch.cross(axis, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), dim=0)
    fallback = _normalize_vector(fallback)
    radial_unit = radial / radial_distance.clamp_min(EPS).view(-1, 1)
    radial_unit[radial_distance <= EPS] = fallback
    tangential_unit = torch.cross(axis.view(1, 3).expand_as(radial_unit), radial_unit, dim=-1)
    tangential_unit = tangential_unit / torch.linalg.vector_norm(tangential_unit, dim=-1).clamp_min(EPS).view(-1, 1)

    plug_centroid = _first_region_centroid(sample, ("plug", "plug_core"))
    extracellular_centroid = _first_region_centroid(sample, ("extracellular_loop", "plug_apical_loop"))
    if plug_centroid is None:
        distance_to_plug = torch.zeros(sample.pos.size(0), dtype=torch.float32)
        has_plug = torch.zeros(sample.pos.size(0), dtype=torch.float32)
    else:
        distance_to_plug = torch.linalg.vector_norm(sample.pos - plug_centroid.view(1, 3), dim=-1)
        has_plug = torch.ones(sample.pos.size(0), dtype=torch.float32)
    if extracellular_centroid is None:
        distance_to_extracellular = torch.zeros(sample.pos.size(0), dtype=torch.float32)
        has_extracellular = torch.zeros(sample.pos.size(0), dtype=torch.float32)
    else:
        distance_to_extracellular = torch.linalg.vector_norm(sample.pos - extracellular_centroid.view(1, 3), dim=-1)
        has_extracellular = torch.ones(sample.pos.size(0), dtype=torch.float32)

    return BarrelFrame(
        center=center,
        axis=axis,
        radial_unit=radial_unit,
        tangential_unit=tangential_unit,
        radial_distance=radial_distance,
        axial_coordinate=axial_coordinate,
        distance_to_plug=distance_to_plug,
        distance_to_extracellular=distance_to_extracellular,
        has_plug=has_plug,
        has_extracellular=has_extracellular,
    )


def _frame_components(vectors: torch.Tensor, frame: BarrelFrame) -> torch.Tensor:
    radial = torch.sum(vectors * frame.radial_unit, dim=-1)
    tangential = torch.sum(vectors * frame.tangential_unit, dim=-1)
    axial = vectors @ frame.axis
    return torch.stack([radial, tangential, axial], dim=-1)


def _components_to_vectors(components: torch.Tensor, frame: BarrelFrame) -> torch.Tensor:
    return (
        components[:, 0:1] * frame.radial_unit
        + components[:, 1:2] * frame.tangential_unit
        + components[:, 2:3] * frame.axis.view(1, 3)
    )


def _structural_tail(sample: Sample) -> dict[str, torch.Tensor]:
    n = sample.pos.size(0)
    zeros = torch.zeros(n, dtype=torch.float32)
    coil = torch.ones(n, dtype=torch.float32)
    if sample.x is None or sample.x.size(1) < 16:
        return {
            "sasa_scaled": zeros,
            "rsa": zeros,
            "residue_depth": zeros,
            "coordination_number": zeros,
            "hse_up": zeros,
            "hse_down": zeros,
            "dssp_helix": zeros,
            "dssp_sheet": zeros,
            "dssp_coil": coil,
        }
    tail = sample.x[:, -16:].float()
    return {
        "sasa_scaled": tail[:, 1],
        "rsa": tail[:, 2],
        "residue_depth": tail[:, 3],
        "coordination_number": tail[:, 4],
        "hse_up": tail[:, 5],
        "hse_down": tail[:, 6],
        "dssp_helix": tail[:, 13],
        "dssp_sheet": tail[:, 14],
        "dssp_coil": tail[:, 15],
    }


def _linear_schema(samples: list[Sample]) -> LinearSchema:
    family_labels = tuple(sorted({"unknown", *(sample.family for sample in samples)}))
    state_labels = tuple(sorted({"unknown", *(sample.state for sample in samples)}))
    substrate_labels = tuple(sorted({"unknown", *(sample.substrate for sample in samples)}))
    feature_names = (
        *(f"region:{name}" for name in LINEAR_REGION_FEATURES),
        *(f"family:{name}" for name in family_labels),
        *(f"state:{name}" for name in state_labels),
        *(f"substrate:{name}" for name in substrate_labels),
        *CONTINUOUS_FEATURE_NAMES,
    )
    return LinearSchema(
        family_labels=family_labels,
        state_labels=state_labels,
        substrate_labels=substrate_labels,
        feature_names=feature_names,
    )


def _one_hot_constant(n: int, labels: tuple[str, ...], value: str) -> torch.Tensor:
    out = torch.zeros((n, len(labels)), dtype=torch.float32)
    if value in labels:
        out[:, labels.index(value)] = 1.0
    elif "unknown" in labels:
        out[:, labels.index("unknown")] = 1.0
    return out


def _linear_features(sample: Sample, schema: LinearSchema, frame: BarrelFrame) -> torch.Tensor:
    n = sample.pos.size(0)
    columns: list[torch.Tensor] = []
    for name in LINEAR_REGION_FEATURES:
        mask = sample.regions.get(name)
        column = mask.float() if mask is not None else torch.zeros(n, dtype=torch.float32)
        columns.append(column.view(n, 1))
    columns.append(_one_hot_constant(n, schema.family_labels, sample.family))
    columns.append(_one_hot_constant(n, schema.state_labels, sample.state))
    columns.append(_one_hot_constant(n, schema.substrate_labels, sample.substrate))

    structural = _structural_tail(sample)
    residue_position = (
        torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32) if n > 1 else torch.zeros(1, dtype=torch.float32)
    )
    plddt_scaled = (sample.plddt.float() / 100.0).clamp(0.0, 1.0)
    distance_to_center = torch.linalg.vector_norm(sample.pos - frame.center.view(1, 3), dim=-1)
    continuous = {
        "plddt_scaled": plddt_scaled,
        "residue_position": residue_position,
        "radial_distance": frame.radial_distance,
        "axial_coordinate": frame.axial_coordinate,
        "abs_axial_coordinate": torch.abs(frame.axial_coordinate),
        "distance_to_plug_centroid": frame.distance_to_plug,
        "distance_to_extracellular_centroid": frame.distance_to_extracellular,
        "distance_to_barrel_center": distance_to_center,
        "has_plug_centroid": frame.has_plug,
        "has_extracellular_centroid": frame.has_extracellular,
        **structural,
    }
    for name in CONTINUOUS_FEATURE_NAMES:
        columns.append(continuous[name].float().view(n, 1))
    return torch.cat(columns, dim=1)


def _standardized_design(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x_std = (x - mean.view(1, -1)) / std.clamp_min(1e-6).view(1, -1)
    return torch.cat([torch.ones((x_std.size(0), 1), dtype=torch.float32), x_std], dim=1)


def _fit_barrel_frame_ridge(samples: list[Sample], *, alpha: float) -> RidgeModel:
    schema = _linear_schema(samples)
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    for sample in samples:
        frame = _barrel_frame(sample)
        x_parts.append(_linear_features(sample, schema, frame))
        y_parts.append(_frame_components(sample.target, frame))
    x = torch.cat(x_parts, dim=0).float()
    y = torch.cat(y_parts, dim=0).float()
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    design = _standardized_design(x, mean, std)
    xtx = design.T @ design
    reg = torch.eye(xtx.size(0), dtype=torch.float32) * float(alpha)
    reg[0, 0] = 0.0
    xty = design.T @ y
    try:
        weights = torch.linalg.solve(xtx + reg, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx + reg) @ xty
    return RidgeModel(
        schema=schema,
        mean=mean,
        std=std,
        weights=weights,
        alpha=float(alpha),
        n_train_residues=int(x.size(0)),
    )


def _predict_barrel_frame_ridge(sample: Sample, model: RidgeModel) -> torch.Tensor:
    frame = _barrel_frame(sample)
    x = _linear_features(sample, model.schema, frame)
    design = _standardized_design(x, model.mean, model.std)
    components = design @ model.weights
    return _components_to_vectors(components, frame)


def _clip_prediction(prediction: torch.Tensor, max_norm: float) -> torch.Tensor:
    if float(max_norm) <= 0.0:
        return prediction
    norms = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / norms.clamp_min(EPS), max=1.0)
    return prediction * scale


def _save_prediction(
    output_dir: Path,
    sample: Sample,
    prediction: torch.Tensor,
    *,
    baseline: str,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample.stem}.pt"
    torch.save(
        {
            "pair_id": sample.stem,
            "pred_delta": prediction.float(),
            "metadata": {
                "baseline": baseline,
                **metadata,
            },
        },
        path,
    )
    return path


def _evaluate_prediction_dir(
    args: argparse.Namespace,
    target_paths: list[Path],
    baseline_name: str,
    prediction_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_json = output_root / f"{baseline_name}_region_metrics.json"
    eval_args = argparse.Namespace(
        inputs=[str(path) for path in target_paths],
        predictions=str(prediction_dir),
        output_json=str(output_json),
        output_csv=str(output_root / f"{baseline_name}_region_metrics.csv"),
        region_json=None,
        include_all_region=bool(args.include_all_region),
        direction_threshold=float(args.direction_threshold),
        add_derived_regions=bool(args.add_derived_regions),
        plug_apical_fraction=float(args.plug_apical_fraction),
        plug_extension_residues=int(args.plug_extension_residues),
        bootstrap_iter=int(args.bootstrap_iter),
        bootstrap_seed=int(args.bootstrap_seed),
        paired_delta_csv=str(output_root / f"{baseline_name}_paired_delta.csv"),
        tonb_metrics_csv=str(output_root / f"{baseline_name}_tonb_state_metrics.csv"),
        tonb_exposure_threshold=float(args.tonb_exposure_threshold),
    )
    report = evaluate_regions(eval_args)
    return {
        "output_json": str(output_json),
        "output_csv": eval_args.output_csv,
        "paired_delta_csv": eval_args.paired_delta_csv,
        "tonb_metrics_csv": eval_args.tonb_metrics_csv,
        "n_samples": report["n_samples"],
    }


def build_baselines(args: argparse.Namespace) -> dict[str, Any]:
    target_paths, donor_paths = _split_paths(args)
    targets = [_load_sample(path, args) for path in target_paths]
    donors = [_load_sample(path, args) for path in donor_paths]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_baselines = tuple(args.baseline or ALL_BASELINES)

    residue_weighted_stats = _fit_region_stats(donors, sample_weighted=False)
    sample_weighted_stats = _fit_region_stats(donors, sample_weighted=True)
    ridge_model = (
        _fit_barrel_frame_ridge(donors, alpha=float(args.ridge_alpha))
        if "barrel_frame_ridge" in selected_baselines
        else None
    )

    prediction_summary_rows: list[dict[str, Any]] = []
    baseline_reports: dict[str, Any] = {}

    for baseline_name in selected_baselines:
        prediction_dir = output_root / baseline_name
        output_files: list[str] = []
        region_rows: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}

        for sample in targets:
            if baseline_name in REGION_MEAN_BASELINES:
                fields = REGION_MEAN_BASELINES[baseline_name]
                prediction, rows = _assign_region_predictions(
                    sample,
                    stats=residue_weighted_stats,
                    fields=fields,
                    priority_text=args.region_priority,
                )
                metadata = {"fit": "residue_weighted_region_mean", "group_fields": list(fields)}
                region_rows.extend({**row, "baseline": baseline_name} for row in rows)
            elif baseline_name in RIGID_REGION_BASELINES:
                fields = RIGID_REGION_BASELINES[baseline_name]
                prediction, rows = _assign_region_predictions(
                    sample,
                    stats=sample_weighted_stats,
                    fields=fields,
                    priority_text=args.region_priority,
                )
                metadata = {"fit": "sample_weighted_region_centroid_shift", "group_fields": list(fields)}
                region_rows.extend({**row, "baseline": baseline_name} for row in rows)
            elif baseline_name in TARGETED_RIGID_BASELINES:
                target_region = TARGETED_RIGID_BASELINES[baseline_name]
                prediction, rows = _assign_region_predictions(
                    sample,
                    stats=sample_weighted_stats,
                    fields=(),
                    priority_text=args.region_priority,
                    only_region=target_region,
                )
                metadata = {
                    "fit": "sample_weighted_target_region_centroid_shift",
                    "target_region": target_region,
                    "group_fields": [],
                }
                region_rows.extend({**row, "baseline": baseline_name} for row in rows)
            elif baseline_name == "barrel_frame_ridge":
                if ridge_model is None:
                    raise RuntimeError("Internal error: ridge model was not fitted.")
                prediction = _predict_barrel_frame_ridge(sample, ridge_model)
                metadata = {
                    "fit": "barrel_frame_ridge",
                    "ridge_alpha": ridge_model.alpha,
                    "n_train_residues": ridge_model.n_train_residues,
                    "feature_count": len(ridge_model.schema.feature_names),
                }
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            prediction = _clip_prediction(prediction, float(args.max_prediction_norm))
            output_path = _save_prediction(prediction_dir, sample, prediction, baseline=baseline_name, metadata=metadata)
            output_files.append(str(output_path))

        if region_rows:
            _write_csv(output_root / f"{baseline_name}_assignment_summary.csv", region_rows)
            prediction_summary_rows.extend(region_rows)

        metrics = None
        if not bool(args.skip_eval):
            metrics = _evaluate_prediction_dir(args, target_paths, baseline_name, prediction_dir, output_root)

        baseline_reports[baseline_name] = {
            "prediction_dir": str(prediction_dir),
            "n_predictions": len(output_files),
            "output_files": output_files,
            "metrics": metrics,
            "metadata": metadata,
        }

    if prediction_summary_rows:
        _write_csv(output_root / "coordinate_baseline_assignment_summary.csv", prediction_summary_rows)

    report = {
        "data_dir": args.data_dir,
        "split": args.split,
        "donor_splits": args.donor_split or ["train", "val"],
        "split_source": args.split_source,
        "target_count": len(targets),
        "donor_count": len(donors),
        "baselines": baseline_reports,
        "region_priority": [item.strip() for item in str(args.region_priority).split(",") if item.strip()],
        "ridge_alpha": float(args.ridge_alpha),
        "max_prediction_norm": float(args.max_prediction_norm),
        "fit_contract": {
            "region_mean": "residue-weighted train/val mean displacement by region and optional family/state group",
            "rigid_region": "sample-weighted train/val mean centroid shift by region and optional family/state group",
            "targeted_rigid": "only the named target region is shifted; all other residues remain zero-displacement",
            "barrel_frame_ridge": (
                "ridge regression predicts radial/tangential/axial displacement components in a per-sample barrel "
                "frame whose axis is oriented toward extracellular-loop/apical-plug geometry when available"
            ),
            "test_leakage": "target split samples are never used for fitting statistics or ridge coefficients",
        },
    }
    if ridge_model is not None:
        report["barrel_frame_ridge_schema"] = {
            "feature_names": list(ridge_model.schema.feature_names),
            "family_labels": list(ridge_model.schema.family_labels),
            "state_labels": list(ridge_model.schema.state_labels),
            "substrate_labels": list(ridge_model.schema.substrate_labels),
            "n_train_residues": ridge_model.n_train_residues,
        }
    report_path = Path(args.report_path) if args.report_path else output_root / "coordinate_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    report = build_baselines(parse_args())
    print(
        json.dumps(
            _json_safe(
                {
                    "report_path": report["report_path"],
                    "target_count": report["target_count"],
                    "donor_count": report["donor_count"],
                    "baselines": list(report["baselines"].keys()),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
