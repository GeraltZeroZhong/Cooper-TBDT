"""Metric helpers for TBDT state-displacement evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from scipy.stats import rankdata, wilcoxon


def _rms_from_vectors(vectors: torch.Tensor) -> float:
    if vectors.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(torch.sum(vectors.float().square(), dim=-1))).item())


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bootstrap_ci(values: list[float], *, n_iter: int, seed: int, statistic: str = "median") -> tuple[float, float]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan"), float("nan")
    arr = torch.tensor(values, dtype=torch.float64).numpy()
    import numpy as np

    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_iter), dtype=float)
    for idx in range(int(n_iter)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[idx] = float(np.mean(sample) if statistic == "mean" else np.median(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _wilcoxon_stats(deltas: list[float]) -> dict[str, float | int | str]:
    values = [float(v) for v in deltas if math.isfinite(float(v)) and abs(float(v)) > 1e-12]
    if not values:
        return {
            "wilcoxon_n_nonzero": 0,
            "wilcoxon_statistic_less": float("nan"),
            "wilcoxon_p_less_method_lt_raw": float("nan"),
            "wilcoxon_statistic_two_sided": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
            "signed_rank_biserial_effect_method_lt_raw": float("nan"),
            "wilcoxon_status": "no_nonzero_deltas",
        }
    ranks = rankdata([abs(value) for value in values], method="average")
    improved_rank_sum = float(sum(rank for rank, value in zip(ranks, values) if value < 0.0))
    worsened_rank_sum = float(sum(rank for rank, value in zip(ranks, values) if value > 0.0))
    rank_total = improved_rank_sum + worsened_rank_sum
    effect = (improved_rank_sum - worsened_rank_sum) / rank_total if rank_total else float("nan")
    try:
        less = wilcoxon(values, alternative="less", zero_method="wilcox")
        two = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:
        return {
            "wilcoxon_n_nonzero": len(values),
            "wilcoxon_statistic_less": float("nan"),
            "wilcoxon_p_less_method_lt_raw": float("nan"),
            "wilcoxon_statistic_two_sided": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
            "signed_rank_biserial_effect_method_lt_raw": effect,
            "wilcoxon_status": str(exc),
        }
    return {
        "wilcoxon_n_nonzero": len(values),
        "wilcoxon_statistic_less": float(less.statistic),
        "wilcoxon_p_less_method_lt_raw": float(less.pvalue),
        "wilcoxon_statistic_two_sided": float(two.statistic),
        "wilcoxon_p_two_sided": float(two.pvalue),
        "signed_rank_biserial_effect_method_lt_raw": effect,
        "wilcoxon_status": "ok",
    }


def _region_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    *,
    direction_threshold: float = 1.0,
) -> dict[str, float | int]:
    t = target[mask]
    p = prediction[mask]
    error = p - t
    target_norm = torch.linalg.vector_norm(t, dim=-1)
    pred_norm = torch.linalg.vector_norm(p, dim=-1)
    error_norm = torch.linalg.vector_norm(error, dim=-1)
    target_rms = _rms_from_vectors(t)
    pred_rms = _rms_from_vectors(p)
    error_rms = _rms_from_vectors(error)
    improvement = target_rms - error_rms if math.isfinite(target_rms) and math.isfinite(error_rms) else float("nan")
    improvement_fraction = improvement / target_rms if target_rms and math.isfinite(improvement) else float("nan")
    baseline_mse = float(torch.mean(torch.sum(t.float().square(), dim=-1)).item()) if t.numel() else float("nan")
    prediction_mse = (
        float(torch.mean(torch.sum(error.float().square(), dim=-1)).item()) if error.numel() else float("nan")
    )
    mse_improvement_fraction = (
        (baseline_mse - prediction_mse) / baseline_mse
        if baseline_mse and math.isfinite(baseline_mse) and math.isfinite(prediction_mse)
        else float("nan")
    )
    better_than_zero = error_norm < target_norm
    moving = target_norm >= float(direction_threshold)
    if bool(moving.any()):
        direction_cosine = torch.nn.functional.cosine_similarity(p[moving], t[moving], dim=-1, eps=1e-8)
        direction_cosine_mean = float(direction_cosine.mean().item())
    else:
        direction_cosine_mean = float("nan")
    return {
        "n_residues": int(mask.sum().item()),
        "target_displacement_rms": target_rms,
        "target_displacement_mean": float(target_norm.mean().item()) if target_norm.numel() else float("nan"),
        "predicted_displacement_rms": pred_rms,
        "predicted_displacement_mean": float(pred_norm.mean().item()) if pred_norm.numel() else float("nan"),
        "prediction_error_rms": error_rms,
        "prediction_error_mean": float(error_norm.mean().item()) if error_norm.numel() else float("nan"),
        "zero_error_rms": target_rms,
        "improvement_vs_zero": improvement,
        "improvement_vs_zero_fraction": improvement_fraction,
        "mse_improvement_vs_zero_fraction": mse_improvement_fraction,
        "better_than_zero_rate": float(better_than_zero.float().mean().item()) if better_than_zero.numel() else float("nan"),
        "worse_than_zero_rate": (
            float((~better_than_zero).float().mean().item()) if better_than_zero.numel() else float("nan")
        ),
        "magnitude_mae": float(torch.mean(torch.abs(pred_norm - target_norm)).item()) if target_norm.numel() else float("nan"),
        "direction_cosine_mean": direction_cosine_mean,
    }


def _unit_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= 1e-8:
        return torch.zeros_like(vector)
    return vector / norm


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if float(torch.linalg.vector_norm(a).item()) <= 1e-8 or float(torch.linalg.vector_norm(b).item()) <= 1e-8:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1, eps=1e-8).item())


def _state_from_delta(delta: float, threshold: float) -> str:
    if not math.isfinite(delta):
        return "missing"
    if delta >= threshold:
        return "exposed_like"
    if delta <= -threshold:
        return "buried_like"
    return "unchanged"


def _centroid(pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    if mask is None or not bool(mask.any()):
        return None
    return pos[mask].float().mean(dim=0)


def _distance(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(torch.linalg.vector_norm(a - b).item())


def _tonb_state_metrics(
    *,
    pos: torch.Tensor | None,
    target: torch.Tensor,
    prediction: torch.Tensor,
    regions: dict[str, torch.Tensor],
    exposure_threshold: float,
) -> dict[str, Any] | None:
    tonb_mask = regions.get("tonb_box")
    barrel_mask = regions.get("barrel_core")
    if pos is None or tonb_mask is None or barrel_mask is None or not bool(tonb_mask.any()) or not bool(barrel_mask.any()):
        return None

    pos = pos.float()
    target_pos = pos + target.float()
    pred_pos = pos + prediction.float()

    plug_ref_mask = regions.get("plug_core")
    if plug_ref_mask is None:
        plug_ref_mask = regions.get("plug")
    extension_mask = regions.get("plug_extension_nt")

    tonb_af2 = _centroid(pos, tonb_mask)
    tonb_target = _centroid(target_pos, tonb_mask)
    tonb_pred = _centroid(pred_pos, tonb_mask)
    barrel_ref = _centroid(pos, barrel_mask)
    plug_ref = _centroid(pos, plug_ref_mask) if plug_ref_mask is not None else None

    target_disp = tonb_target - tonb_af2
    pred_disp = tonb_pred - tonb_af2
    disp_error = pred_disp - target_disp

    af2_barrel_dist = _distance(tonb_af2, barrel_ref)
    target_barrel_dist = _distance(tonb_target, barrel_ref)
    pred_barrel_dist = _distance(tonb_pred, barrel_ref)
    target_exposure_delta = target_barrel_dist - af2_barrel_dist
    pred_exposure_delta = pred_barrel_dist - af2_barrel_dist
    target_state = _state_from_delta(target_exposure_delta, float(exposure_threshold))
    pred_state = _state_from_delta(pred_exposure_delta, float(exposure_threshold))
    displacement_cosine = _cosine(pred_disp, target_disp)
    direction_compatible = math.isfinite(displacement_cosine) and displacement_cosine >= 0.5

    target_axis = _unit_vector(tonb_target - barrel_ref)
    pred_axis = _unit_vector(tonb_pred - barrel_ref)
    af2_axis = _unit_vector(tonb_af2 - barrel_ref)

    metrics: dict[str, Any] = {
        "tonb_n_residues": int(tonb_mask.sum().item()),
        "tonb_target_centroid_displacement": float(torch.linalg.vector_norm(target_disp).item()),
        "tonb_predicted_centroid_displacement": float(torch.linalg.vector_norm(pred_disp).item()),
        "tonb_centroid_displacement_error": float(torch.linalg.vector_norm(disp_error).item()),
        "tonb_centroid_displacement_cosine": displacement_cosine,
        "tonb_direction_compatible": direction_compatible,
        "tonb_af2_distance_to_barrel_ref": af2_barrel_dist,
        "tonb_target_distance_to_barrel_ref": target_barrel_dist,
        "tonb_predicted_distance_to_barrel_ref": pred_barrel_dist,
        "tonb_target_exposure_delta": target_exposure_delta,
        "tonb_predicted_exposure_delta": pred_exposure_delta,
        "tonb_exposure_delta_error": pred_exposure_delta - target_exposure_delta,
        "tonb_target_state": target_state,
        "tonb_predicted_state": pred_state,
        "tonb_state_correct": target_state == pred_state,
        "tonb_target_axis_cosine_vs_af2": _cosine(target_axis, af2_axis),
        "tonb_predicted_axis_cosine_vs_target": _cosine(pred_axis, target_axis),
    }
    if plug_ref is not None:
        af2_plug_dist = _distance(tonb_af2, plug_ref)
        target_plug_dist = _distance(tonb_target, plug_ref)
        pred_plug_dist = _distance(tonb_pred, plug_ref)
        metrics.update(
            {
                "tonb_af2_distance_to_plug_ref": af2_plug_dist,
                "tonb_target_distance_to_plug_ref": target_plug_dist,
                "tonb_predicted_distance_to_plug_ref": pred_plug_dist,
                "tonb_target_plug_distance_delta": target_plug_dist - af2_plug_dist,
                "tonb_predicted_plug_distance_delta": pred_plug_dist - af2_plug_dist,
                "tonb_plug_distance_delta_error": (pred_plug_dist - af2_plug_dist) - (target_plug_dist - af2_plug_dist),
            }
        )
    if extension_mask is not None and bool(extension_mask.any()):
        ext = _region_metrics(target, prediction, extension_mask, direction_threshold=0.0)
        metrics.update(
            {
                "nt_plug_extension_n_residues": ext["n_residues"],
                "nt_plug_extension_target_rms": ext["target_displacement_rms"],
                "nt_plug_extension_prediction_error_rms": ext["prediction_error_rms"],
                "nt_plug_extension_mse_improvement_vs_zero_fraction": ext[
                    "mse_improvement_vs_zero_fraction"
                ],
                "nt_plug_extension_direction_cosine_mean": ext["direction_cosine_mean"],
            }
        )
    return metrics


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: ("" if value is None else value) for key, value in row.items()}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scope",
        "sample_id",
        "region",
        "n_residues",
        "target_displacement_rms",
        "target_displacement_mean",
        "prediction_error_rms",
        "prediction_error_mean",
        "zero_error_rms",
        "improvement_vs_zero",
        "improvement_vs_zero_fraction",
        "mse_improvement_vs_zero_fraction",
        "better_than_zero_rate",
        "worse_than_zero_rate",
        "predicted_displacement_rms",
        "predicted_displacement_mean",
        "magnitude_mae",
        "direction_cosine_mean",
        "barrel_core_target_rms",
        "sample_count",
        "sample_improvement_rate",
        "sample_worse_rate",
        "sample_improvement_mean",
        "sample_improvement_median",
        "sample_prediction_error_rms_mean",
        "sample_prediction_error_rms_median",
        "sample_zero_error_rms_mean",
        "sample_zero_error_rms_median",
        "sample_mse_improvement_fraction_mean",
        "sample_mse_improvement_fraction_median",
        "delta_rmsd_method_minus_raw",
        "method_rmsd",
        "raw_af2_rmsd",
        "target_state",
        "predicted_state",
        "state_correct",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_row({column: row.get(column) for column in columns}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _paired_delta_summary(
    sample_results: list[dict[str, Any]],
    *,
    bootstrap_iter: int,
    bootstrap_seed: int,
) -> tuple[dict[str, dict[str, float | int | str]], list[dict[str, Any]]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for sample in sample_results:
        sample_id = str(sample.get("sample_id", ""))
        for region_name, metrics in (sample.get("regions") or {}).items():
            raw = _safe_float(metrics.get("zero_error_rms"))
            method = _safe_float(metrics.get("prediction_error_rms"))
            if not math.isfinite(raw) or not math.isfinite(method):
                continue
            delta = method - raw
            row = {
                "sample_id": sample_id,
                "region": region_name,
                "raw_af2_rmsd": raw,
                "method_rmsd": method,
                "delta_rmsd_method_minus_raw": delta,
                "improved": delta < 0.0,
                "worsened": delta > 0.0,
                "n_residues": metrics.get("n_residues", ""),
            }
            rows.append(row)
            by_region[region_name].append(row)

    summary: dict[str, dict[str, float | int | str]] = {}
    for region_name, region_rows in sorted(by_region.items()):
        deltas = [float(row["delta_rmsd_method_minus_raw"]) for row in region_rows]
        mean_low, mean_high = _bootstrap_ci(
            deltas,
            n_iter=int(bootstrap_iter),
            seed=int(bootstrap_seed),
            statistic="mean",
        )
        med_low, med_high = _bootstrap_ci(
            deltas,
            n_iter=int(bootstrap_iter),
            seed=int(bootstrap_seed) + 101,
            statistic="median",
        )
        summary[region_name] = {
            "n_targets": len(deltas),
            "n_improved": sum(1 for value in deltas if value < 0.0),
            "n_worsened": sum(1 for value in deltas if value > 0.0),
            "n_tied": sum(1 for value in deltas if value == 0.0),
            "improved_fraction": _mean([1.0 if value < 0.0 else 0.0 for value in deltas]),
            "median_delta_rmsd_method_minus_raw": _median(deltas),
            "mean_delta_rmsd_method_minus_raw": _mean(deltas),
            "mean_delta_ci95_low": mean_low,
            "mean_delta_ci95_high": mean_high,
            "median_delta_ci95_low": med_low,
            "median_delta_ci95_high": med_high,
            **_wilcoxon_stats(deltas),
        }
    return summary, rows


def _tonb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_targets": 0}

    numeric_fields = [
        "tonb_target_centroid_displacement",
        "tonb_predicted_centroid_displacement",
        "tonb_centroid_displacement_error",
        "tonb_centroid_displacement_cosine",
        "tonb_target_exposure_delta",
        "tonb_predicted_exposure_delta",
        "tonb_exposure_delta_error",
        "tonb_predicted_axis_cosine_vs_target",
        "nt_plug_extension_prediction_error_rms",
        "nt_plug_extension_mse_improvement_vs_zero_fraction",
    ]
    summary: dict[str, Any] = {"n_targets": len(rows)}
    for key in numeric_fields:
        values = [_safe_float(row.get(key)) for row in rows]
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_median"] = _median(values)

    correct = [1.0 if bool(row.get("tonb_state_correct")) else 0.0 for row in rows]
    summary["tonb_state_accuracy"] = _mean(correct)
    direction_compatible = [1.0 if bool(row.get("tonb_direction_compatible")) else 0.0 for row in rows]
    summary["tonb_direction_compatible_rate"] = _mean(direction_compatible)
    confusion: dict[str, int] = defaultdict(int)
    for row in rows:
        confusion[f"{row.get('tonb_target_state', 'missing')}->{row.get('tonb_predicted_state', 'missing')}"] += 1
    summary["tonb_state_confusion"] = dict(sorted(confusion.items()))
    return summary


def _write_generic_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
            writer.writerow(_flatten_row({column: row.get(column) for column in columns}))


def _sample_region_summary(sample_results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in sample_results:
        for region_name, metrics in sample.get("regions", {}).items():
            by_region[region_name].append(metrics)

    summaries: dict[str, dict[str, float | int]] = {}
    for region_name, rows in by_region.items():
        improvements = [
            float(row["zero_error_rms"]) - float(row["prediction_error_rms"])
            for row in rows
            if math.isfinite(float(row.get("zero_error_rms", float("nan"))))
            and math.isfinite(float(row.get("prediction_error_rms", float("nan"))))
        ]
        prediction_error = [float(row["prediction_error_rms"]) for row in rows]
        zero_error = [float(row["zero_error_rms"]) for row in rows]
        mse_improve = [float(row["mse_improvement_vs_zero_fraction"]) for row in rows]
        summaries[region_name] = {
            "sample_count": len(rows),
            "sample_improvement_rate": _mean([1.0 if value > 0.0 else 0.0 for value in improvements]),
            "sample_worse_rate": _mean([1.0 if value < 0.0 else 0.0 for value in improvements]),
            "sample_improvement_mean": _mean(improvements),
            "sample_improvement_median": _median(improvements),
            "sample_prediction_error_rms_mean": _mean(prediction_error),
            "sample_prediction_error_rms_median": _median(prediction_error),
            "sample_zero_error_rms_mean": _mean(zero_error),
            "sample_zero_error_rms_median": _median(zero_error),
            "sample_mse_improvement_fraction_mean": _mean(mse_improve),
            "sample_mse_improvement_fraction_median": _median(mse_improve),
        }
    return summaries
