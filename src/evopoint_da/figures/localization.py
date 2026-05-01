"""Shared residue-shift localization helpers for Cooper-TBDT figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from evopoint_da.figures.io import finite_float, read_csv_rows
from evopoint_da.pipeline.eval_tbdt_classification_curves import _pr_curve, _roc_curve
from evopoint_da.pipeline.eval_tbdt_state import _as_mapping, _extract_prediction, _extract_regions, _extract_target, _load_pt


COOPER_METHOD_ID = "cooper_tbdt_single_5seed"
COOPER_METHOD_LABEL = "Cooper-TBDT 5-seed"
BLEND_METHOD_ID = "cooper_tbdt_scaffold_blend"
BLEND_METHOD_LABEL = "Cooper-TBDT blend"

EXTERNAL_LOCALIZATION_METHODS = (
    ("protcross_pocket_score", "ProtCross", "site", "protcross", "-"),
    ("p2rank_pocket_score", "P2Rank", "site", "p2rank", "--"),
    ("fpocket_pocket_score", "fpocket", "site", "fpocket", ":"),
    ("af2_low_plddt", "AF2 low pLDDT", "afdb_confidence", "raw", "-"),
    ("af2_surface_rsa", "AF2 surface RSA", "afdb_geometry", "peacock", "--"),
    ("prody_anm_mobility", "ANM", "dynamics", "anm", "-."),
    ("prody_gnm_mobility", "GNM", "dynamics", "gnm", ":"),
    ("iupred2a_long", "IUPred2A", "disorder", "iupred", "-."),
)

FILE_LOCALIZATION_METHODS = (
    (BLEND_METHOD_ID, BLEND_METHOD_LABEL, "model", "model_blend", "-"),
    *EXTERNAL_LOCALIZATION_METHODS,
)

LOCALIZATION_METHOD_ORDER = (BLEND_METHOD_ID, *(method[0] for method in EXTERNAL_LOCALIZATION_METHODS))


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def localization_method_label(method: str) -> str:
    if method == COOPER_METHOD_ID:
        return COOPER_METHOD_LABEL
    if method == BLEND_METHOD_ID:
        return BLEND_METHOD_LABEL
    for method_id, label, _, _, _ in EXTERNAL_LOCALIZATION_METHODS:
        if method == method_id:
            return label
    return method


def localization_method_group(method: str) -> str:
    if method == COOPER_METHOD_ID:
        return "model"
    if method == BLEND_METHOD_ID:
        return "model"
    for method_id, _, group, _, _ in FILE_LOCALIZATION_METHODS:
        if method == method_id:
            return group
    return "other"


def localization_method_color(method: str, style: Any) -> str:
    if method == COOPER_METHOD_ID:
        return style.palette["model_seed"]
    for method_id, _, _, color_key, _ in FILE_LOCALIZATION_METHODS:
        if method == method_id:
            return style.palette[color_key]
    return style.palette["neutral"]


def localization_method_linestyle(method: str) -> str:
    if method == COOPER_METHOD_ID:
        return "--"
    for method_id, _, _, _, linestyle in FILE_LOCALIZATION_METHODS:
        if method == method_id:
            return linestyle
    return "-"


def localization_method_linewidth(method: str) -> float:
    if method == COOPER_METHOD_ID:
        return 1.35
    if method == BLEND_METHOD_ID:
        return 1.75
    if method in {"af2_low_plddt", "protcross_pocket_score", "p2rank_pocket_score", "fpocket_pocket_score", "prody_anm_mobility"}:
        return 1.28
    return 1.05


def _sample_paths(sample_list: str | Path) -> list[Path]:
    path = Path(sample_list)
    with path.open("r", encoding="utf-8") as handle:
        paths = [Path(line.strip()) for line in handle if line.strip() and not line.startswith("#")]
    if not paths:
        raise FileNotFoundError(f"No sample paths found in {path}")
    return paths


def _prediction_dir(root: str | Path, seed: int, split: str) -> Path:
    path = Path(root) / f"seed_{seed}_best-selection_{split}"
    if not path.is_dir():
        raise FileNotFoundError(f"Seed prediction directory not found: {path}")
    return path


def _prediction_score(prediction_dir: Path, sample_path: Path, n_residues: int) -> torch.Tensor:
    pred_path = prediction_dir / f"{sample_path.stem}.pt"
    if not pred_path.exists():
        matches = sorted(prediction_dir.rglob(f"{sample_path.stem}.pt"))
        if not matches:
            raise FileNotFoundError(f"No prediction found for {sample_path.stem} in {prediction_dir}")
        pred_path = matches[0]
    prediction = _extract_prediction(_load_pt(pred_path), sample_path.stem, n_residues)
    return torch.linalg.vector_norm(prediction, dim=-1)


def _interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    unique_x = np.unique(x)
    unique_y = np.array([float(np.nanmax(y[x == value])) for value in unique_x], dtype=float)
    return np.interp(grid, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])


def _seed_curve_rows(
    *,
    region: str,
    curve: str,
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    source: str,
    grid_n: int,
) -> list[dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, grid_n)
    matrix = np.vstack([_interpolate_curve(x, y, grid) for x, y in seed_curves])
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
    low = np.clip(mean - sd, 0.0, 1.0)
    high = np.clip(mean + sd, 0.0, 1.0)
    return [
        {
            "region": region,
            "method": COOPER_METHOD_ID,
            "method_label": COOPER_METHOD_LABEL,
            "method_group": "model",
            "curve": curve,
            "x": float(x_value),
            "y": float(y_value),
            "y_low": float(y_low),
            "y_high": float(y_high),
            "y_sd": float(y_sd),
            "source": source,
        }
        for x_value, y_value, y_low, y_high, y_sd in zip(grid, mean, low, high, sd, strict=True)
    ]


def collect_cooper_seed_localization(
    *,
    sample_list: str | Path,
    prediction_root: str | Path,
    seeds: list[int],
    regions: tuple[str, ...],
    split: str = "test",
    positive_threshold: float = 1.0,
    grid_n: int = 201,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute seed-mean localization curves from primary single-model predictions."""

    samples = _sample_paths(sample_list)
    prediction_dirs = {seed: _prediction_dir(prediction_root, seed, split) for seed in seeds}
    labels_by_region: dict[str, list[np.ndarray]] = {region: [] for region in regions}
    scores_by_region_seed: dict[str, dict[int, list[np.ndarray]]] = {
        region: {seed: [] for seed in seeds} for region in regions
    }

    for sample_path in samples:
        sample = _as_mapping(_load_pt(sample_path))
        target = _extract_target(sample, sample_path)
        n_residues = int(target.size(0))
        target_mag = torch.linalg.vector_norm(target, dim=-1)
        region_masks = _extract_regions(sample, sample_path.stem, n_residues, {}, include_all=True)
        seed_scores = {
            seed: _prediction_score(prediction_dir, sample_path, n_residues)
            for seed, prediction_dir in prediction_dirs.items()
        }

        for region in regions:
            if region not in region_masks:
                continue
            mask = region_masks[region].bool()
            if not bool(mask.any()):
                continue
            labels_by_region[region].append((target_mag[mask] >= float(positive_threshold)).cpu().numpy().astype(bool))
            for seed, score in seed_scores.items():
                scores_by_region_seed[region][seed].append(score[mask].cpu().numpy().astype(float))

    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    seed_metric_rows: list[dict[str, Any]] = []
    source = ";".join(str(path) for path in prediction_dirs.values())

    for region in regions:
        if not labels_by_region[region]:
            continue
        labels = np.concatenate(labels_by_region[region]).astype(bool)
        seed_roc_curves: list[tuple[np.ndarray, np.ndarray]] = []
        seed_pr_curves: list[tuple[np.ndarray, np.ndarray]] = []
        aurocs: list[float] = []
        aps: list[float] = []
        for seed in seeds:
            scores = np.concatenate(scores_by_region_seed[region][seed])
            fpr, tpr, auroc = _roc_curve(labels, scores)
            recall, precision, _, ap = _pr_curve(labels, scores)
            seed_roc_curves.append((fpr, tpr))
            seed_pr_curves.append((recall, precision))
            aurocs.append(float(auroc))
            aps.append(float(ap))
            seed_metric_rows.append(
                {
                    "region": region,
                    "method": f"cooper_tbdt_seed_{seed}",
                    "method_label": f"Cooper-TBDT seed {seed}",
                    "method_group": "model_seed",
                    "seed": seed,
                    "n_residues": int(labels.size),
                    "n_positive": int(labels.sum()),
                    "n_negative": int(labels.size - labels.sum()),
                    "positive_rate": float(labels.mean()),
                    "auroc": float(auroc),
                    "average_precision": float(ap),
                    "source": str(prediction_dirs[seed]),
                }
            )
        auroc_array = np.array(aurocs, dtype=float)
        ap_array = np.array(aps, dtype=float)
        summary_rows.append(
            {
                "region": region,
                "method": COOPER_METHOD_ID,
                "method_label": COOPER_METHOD_LABEL,
                "method_group": "model",
                "n_residues": int(labels.size),
                "n_positive": int(labels.sum()),
                "n_negative": int(labels.size - labels.sum()),
                "positive_rate": float(labels.mean()),
                "auroc": float(np.mean(auroc_array)),
                "auroc_sd": float(np.std(auroc_array, ddof=1)) if len(aurocs) > 1 else 0.0,
                "average_precision": float(np.mean(ap_array)),
                "average_precision_sd": float(np.std(ap_array, ddof=1)) if len(aps) > 1 else 0.0,
                "n_seeds": len(seeds),
                "source": source,
            }
        )
        curve_rows.extend(
            _seed_curve_rows(region=region, curve="roc", seed_curves=seed_roc_curves, source=source, grid_n=grid_n)
        )
        curve_rows.extend(
            _seed_curve_rows(region=region, curve="pr", seed_curves=seed_pr_curves, source=source, grid_n=grid_n)
        )
    return summary_rows, curve_rows, seed_metric_rows


def load_external_localization_summary(path: str | Path, regions: tuple[str, ...]) -> list[dict[str, Any]]:
    method_ids = {method_id for method_id, _, _, _, _ in FILE_LOCALIZATION_METHODS}
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        region = row.get("region", "")
        method = row.get("method", "")
        if region not in regions or method not in method_ids:
            continue
        rows.append(
            {
                "region": region,
                "method": method,
                "method_label": localization_method_label(method),
                "method_group": localization_method_group(method),
                "n_residues": int(finite_float(row.get("n_residues"), field="n_residues", source=str(path))),
                "n_positive": int(finite_float(row.get("n_positive"), field="n_positive", source=str(path))),
                "n_negative": int(finite_float(row.get("n_negative"), field="n_negative", source=str(path))),
                "positive_rate": finite_float(row.get("positive_rate"), field="positive_rate", source=str(path)),
                "auroc": finite_float(row.get("auroc"), field="auroc", source=str(path)),
                "auroc_sd": "",
                "average_precision": finite_float(row.get("average_precision"), field="average_precision", source=str(path)),
                "average_precision_sd": "",
                "n_seeds": "",
                "source": str(path),
            }
        )
    order = {method: idx for idx, method in enumerate(LOCALIZATION_METHOD_ORDER)}
    return sorted(rows, key=lambda item: (str(item["region"]), order[str(item["method"])]))


def load_external_localization_curves(path: str | Path, regions: tuple[str, ...]) -> list[dict[str, Any]]:
    method_ids = {method_id for method_id, _, _, _, _ in FILE_LOCALIZATION_METHODS}
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        region = row.get("region", "")
        method = row.get("method", "")
        if region not in regions or method not in method_ids or row.get("curve") not in {"roc", "pr"}:
            continue
        rows.append(
            {
                "region": region,
                "method": method,
                "method_label": localization_method_label(method),
                "method_group": localization_method_group(method),
                "curve": row["curve"],
                "x": finite_float(row.get("x"), field="x", source=str(path)),
                "y": finite_float(row.get("y"), field="y", source=str(path)),
                "y_low": "",
                "y_high": "",
                "y_sd": "",
                "source": str(path),
            }
        )
    return rows
