"""Conformal calibration evaluation pipeline.

This module hosts the executable logic originally kept in scripts/eval_run.py
so the training/inference pipeline code lives under src/ for easier reuse/import.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointLitModule
from evopoint_da.utils.binning import parse_float_edges


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute conformal q-hat from calibration set.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/processed_graphs")
    p.add_argument("--data_cfg", default="configs/data/protein_displacement.yaml")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--output", default="artifacts/conformal_stats.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--calib_batch_size", type=int, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--fallback_num_features", type=int, default=None)
    p.add_argument("--dump_scores", action="store_true", help="Also dump raw calibration scores in output JSON")
    p.add_argument(
        "--plddt-bins",
        default="0,50,70,90,100",
        help="Raw pLDDT bin edges for stratified coverage and tail-risk reporting.",
    )
    p.add_argument(
        "--disp-bins",
        default="0,1,2,3,5,10",
        help="Å displacement-magnitude bin edges for stratified coverage and tail-risk reporting.",
    )
    p.add_argument(
        "--seed-sweep",
        default="",
        help="Comma-separated split seeds (e.g. '11,22,33'). If set, run multi-seed stats (mean/std).",
    )
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--ence-bins", type=int, default=10)
    p.add_argument(
        "--cost-json-out",
        default=None,
        help="Optional output path for runtime/cost/drift report JSON.",
    )
    p.add_argument(
        "--reference-scores-json",
        default=None,
        help="Optional historical score JSON path for drift checks (expects key 'scores' or 'score_quantiles').",
    )
    return p.parse_args()


def _parse_float_edges(raw: str) -> list[float]:
    return parse_float_edges(raw)


def _extract_plddt_from_batch(batch: torch.Tensor, fallback_index: int = 128) -> torch.Tensor:
    if hasattr(batch, "plddt") and batch.plddt is not None:
        plddt = batch.plddt
        if plddt.dim() > 1:
            plddt = plddt.squeeze(-1)
        # normalized [0,1] -> raw [0,100]
        if float(plddt.max().item()) <= 1.5:
            plddt = plddt * 100.0
        return plddt.float()
    if hasattr(batch, "x") and batch.x is not None and batch.x.dim() == 2 and batch.x.size(1) > fallback_index:
        return (batch.x[:, fallback_index].float().clamp(0.0, 1.0) * 100.0).float()
    return torch.full((batch.y.size(0),), float("nan"), device=batch.y.device, dtype=torch.float32)


def _compute_ece(errors: np.ndarray, confidences: np.ndarray, n_bins: int) -> float:
    if errors.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(errors[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _compute_ence(pred_uncertainty: np.ndarray, realized_error: np.ndarray, n_bins: int) -> float:
    if pred_uncertainty.size == 0:
        return float("nan")
    order = np.argsort(pred_uncertainty)
    splits = np.array_split(order, n_bins)
    parts: list[float] = []
    for idx in splits:
        if idx.size == 0:
            continue
        rmse = float(np.sqrt(np.mean(np.square(realized_error[idx]))))
        rmu = float(np.sqrt(np.mean(np.square(pred_uncertainty[idx]))))
        denom = max(rmu, 1e-8)
        parts.append(abs(rmu - rmse) / denom)
    return float(np.mean(parts)) if parts else float("nan")


def _bootstrap_ci(values: np.ndarray, fn, n_iter: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.size
    boots = []
    for _ in range(int(n_iter)):
        sample = values[rng.integers(0, n, n)]
        boots.append(float(fn(sample)))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _tail_risk_block(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for th in thresholds:
        out[f"error_gt_{th:.1f}A_rate"] = float(np.mean(errors > th)) if errors.size else float("nan")
    return out


def _stratified_stats(
    values: np.ndarray,
    strata: np.ndarray,
    edges: list[float],
    qhat: float,
    alpha: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (strata >= lo) & (strata <= hi)
        else:
            mask = (strata >= lo) & (strata < hi)
        key = f"{lo:.1f}to{hi:.1f}"
        if not np.any(mask):
            result[key] = {
                "n": 0,
                "coverage": float("nan"),
                "target_coverage": float(1.0 - alpha),
                "mean_error": float("nan"),
                "median_error": float("nan"),
                "error_gt_2.0A_rate": float("nan"),
                "error_gt_3.0A_rate": float("nan"),
            }
            continue
        vals = values[mask]
        result[key] = {
            "n": int(mask.sum()),
            "coverage": float(np.mean(vals <= qhat)),
            "target_coverage": float(1.0 - alpha),
            "mean_error": float(np.mean(vals)),
            "median_error": float(np.median(vals)),
            **_tail_risk_block(vals, [2.0, 3.0]),
        }
    return result


def _parse_seed_sweep(raw: str) -> list[int]:
    if not str(raw).strip():
        return []
    seeds = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted(set(seeds))


def _safe_reference_scores(path: str | None) -> np.ndarray:
    if not path or not os.path.exists(path):
        return np.array([], dtype=np.float32)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "scores" in payload and isinstance(payload["scores"], list):
        return np.array(payload["scores"], dtype=np.float32)
    return np.array([], dtype=np.float32)


def _compute_ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    xs = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), xs, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), xs, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _evaluate_once(args: argparse.Namespace, split_seed_override: int | None = None) -> dict[str, object]:
    cfg = OmegaConf.load(args.data_cfg) if os.path.exists(args.data_cfg) else OmegaConf.create({})
    dm_kwargs = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "calib_batch_size": args.calib_batch_size if args.calib_batch_size is not None else cfg.get("calib_batch_size", 1),
        "split_seed": split_seed_override if split_seed_override is not None else (args.split_seed if args.split_seed is not None else cfg.get("split_seed", 42)),
        "fallback_num_features": (
            args.fallback_num_features if args.fallback_num_features is not None else cfg.get("fallback_num_features", 144)
        ),
        "split_ranges": cfg.get("split_ranges", None),
    }
    dm = EvoPointDataModule(**dm_kwargs)
    dm.setup("fit")
    loader = dm.calib_dataloader()

    model = EvoPointLitModule.load_from_checkpoint(args.ckpt, map_location=args.device, weights_only=False)
    model.eval().to(args.device)

    plddt_bins = _parse_float_edges(args.plddt_bins)
    disp_bins = _parse_float_edges(args.disp_bins)

    start = time.perf_counter()
    scores: list[float] = []
    plddt_all: list[float] = []
    disp_all: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            pred = model.predict_displacement(batch)
            err = torch.norm(pred - batch.y, dim=-1)
            gt_disp = torch.norm(batch.y, dim=-1)
            plddt = _extract_plddt_from_batch(batch)
            scores.extend(err.detach().cpu().numpy().tolist())
            disp_all.extend(gt_disp.detach().cpu().numpy().tolist())
            plddt_all.extend(plddt.detach().cpu().numpy().tolist())
    elapsed = float(time.perf_counter() - start)

    scores_np = np.array(scores, dtype=np.float32)
    plddt_np = np.array(plddt_all, dtype=np.float32)
    disp_np = np.array(disp_all, dtype=np.float32)
    n = len(scores_np)

    q = np.quantile(scores_np, min(1.0, np.ceil((n + 1) * (1 - args.alpha)) / n), method="higher") if n > 0 else float("nan")
    quantiles = {}
    for pctl in (50, 90, 95, 99):
        quantiles[f"p{pctl}"] = float(np.percentile(scores_np, pctl)) if n > 0 else float("nan")

    cov = float(np.mean(scores_np <= q)) if n > 0 else float("nan")
    cov_ci = _bootstrap_ci((scores_np <= q).astype(np.float32), lambda x: np.mean(x), args.bootstrap_iter, args.bootstrap_seed)
    mean_ci = _bootstrap_ci(scores_np, lambda x: np.mean(x), args.bootstrap_iter, args.bootstrap_seed)
    tail2_ci = _bootstrap_ci((scores_np > 2.0).astype(np.float32), lambda x: np.mean(x), args.bootstrap_iter, args.bootstrap_seed)
    tail3_ci = _bootstrap_ci((scores_np > 3.0).astype(np.float32), lambda x: np.mean(x), args.bootstrap_iter, args.bootstrap_seed)

    conf = np.clip(1.0 - (scores_np / max(float(q), 1e-8)), 0.0, 1.0) if n > 0 else np.array([], dtype=np.float32)
    binary_correct = (scores_np <= q).astype(np.float32) if n > 0 else np.array([], dtype=np.float32)
    ece = _compute_ece(binary_correct, conf, n_bins=args.ece_bins)
    ence = _compute_ence(np.full_like(scores_np, float(q)), scores_np, n_bins=args.ence_bins) if n > 0 else float("nan")

    payload: dict[str, object] = {
        "alpha": args.alpha,
        "target_coverage": float(1.0 - args.alpha),
        "num_calibration_nodes": int(n),
        "qhat": float(q),
        "score_quantiles": quantiles,
        "score_mean": float(scores_np.mean()) if n > 0 else float("nan"),
        "score_std": float(scores_np.std()) if n > 0 else float("nan"),
        "estimated_empirical_coverage": cov,
        "calibration_quality": {
            "ece": float(ece),
            "ence": float(ence),
            "coverage_gap_abs": float(abs(cov - (1.0 - args.alpha))) if n > 0 else float("nan"),
        },
        "tail_risk": _tail_risk_block(scores_np, [2.0, 3.0]),
        "bootstrap_ci95": {
            "coverage": {"low": cov_ci[0], "high": cov_ci[1]},
            "score_mean": {"low": mean_ci[0], "high": mean_ci[1]},
            "error_gt_2.0A_rate": {"low": tail2_ci[0], "high": tail2_ci[1]},
            "error_gt_3.0A_rate": {"low": tail3_ci[0], "high": tail3_ci[1]},
        },
        "stratified_by_plddt": _stratified_stats(scores_np, plddt_np, plddt_bins, float(q), args.alpha),
        "stratified_by_displacement": _stratified_stats(scores_np, disp_np, disp_bins, float(q), args.alpha),
        "runtime_cost": {
            "elapsed_seconds": elapsed,
            "nodes_per_second": float(n / elapsed) if elapsed > 0 else float("nan"),
            "estimated_gpu_hours": float(elapsed / 3600.0) if str(args.device).startswith("cuda") else 0.0,
        },
    }
    if args.dump_scores:
        payload["scores"] = scores_np.tolist()
    return payload


def main() -> None:
    args = get_args()
    payload = _evaluate_once(args, split_seed_override=args.split_seed)
    seed_sweep = _parse_seed_sweep(args.seed_sweep)
    if seed_sweep:
        runs: list[dict[str, object]] = []
        for seed in seed_sweep:
            run = _evaluate_once(args, split_seed_override=seed)
            run["split_seed"] = seed
            runs.append(run)
        key_extractors = {
            "qhat": lambda x: float(x["qhat"]),
            "estimated_empirical_coverage": lambda x: float(x["estimated_empirical_coverage"]),
            "ece": lambda x: float(x["calibration_quality"]["ece"]),
            "ence": lambda x: float(x["calibration_quality"]["ence"]),
            "error_gt_2.0A_rate": lambda x: float(x["tail_risk"]["error_gt_2.0A_rate"]),
            "error_gt_3.0A_rate": lambda x: float(x["tail_risk"]["error_gt_3.0A_rate"]),
        }
        summary = {}
        for name, fn in key_extractors.items():
            vals = np.array([fn(r) for r in runs], dtype=np.float64)
            summary[name] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "n_runs": int(vals.size),
            }
        payload["multi_seed"] = {"seeds": seed_sweep, "summary": summary, "runs": runs}

    reference_scores = _safe_reference_scores(args.reference_scores_json)
    if reference_scores.size > 0 and args.dump_scores and "scores" in payload:
        current_scores = np.array(payload["scores"], dtype=np.float32)
        ks = _compute_ks_stat(current_scores, reference_scores)
        payload["drift_check"] = {
            "reference_path": args.reference_scores_json,
            "reference_n": int(reference_scores.size),
            "current_n": int(current_scores.size),
            "ks_statistic": float(ks),
            "mean_shift": float(current_scores.mean() - reference_scores.mean()),
            "std_shift": float(current_scores.std() - reference_scores.std()),
        }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.cost_json_out:
        cost_dir = os.path.dirname(args.cost_json_out)
        if cost_dir:
            os.makedirs(cost_dir, exist_ok=True)
        with open(args.cost_json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "runtime_cost": payload.get("runtime_cost", {}),
                    "multi_seed": payload.get("multi_seed", {}),
                    "drift_check": payload.get("drift_check", {}),
                },
                f,
                indent=2,
            )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
