"""Build a compact mechanistic TBDT evaluation report.

This report is intentionally downstream of ``eval_tbdt_state``. It does not
recompute metrics; it formats per-target paired deltas, fine region readouts,
and TonB-box state metrics from detailed evaluation JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon


REPORT_REGIONS = (
    "eval",
    "plug",
    "plug_core",
    "plug_apical_loop",
    "plug_extension_nt",
    "tonb_box",
    "barrel_core",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize detailed TBDT mechanistic metric JSON files.")
    parser.add_argument(
        "--metric-json",
        action="append",
        required=True,
        help="Label/path pair such as scaffold_blend=artifacts/.../metrics.json. May repeat.",
    )
    parser.add_argument(
        "--seed-family-dir",
        action="append",
        default=[],
        help=(
            "Label/path pair for a directory of seed_*_best-selection_test.json files. "
            "The report averages method RMSD across seeds per target before paired testing."
        ),
    )
    parser.add_argument("--bootstrap-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-md", default="artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_report.md")
    parser.add_argument("--out-csv", default="artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_summary.csv")
    return parser.parse_args()


def _load_specs(
    raw_specs: list[str],
    seed_family_specs: list[str],
    *,
    bootstrap_iter: int,
    bootstrap_seed: int,
) -> list[tuple[str, Path, dict[str, Any]]]:
    specs: list[tuple[str, Path, dict[str, Any]]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"--metric-json must be label=path, got: {raw}")
        label, path_text = raw.split("=", 1)
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            specs.append((label, path, json.load(handle)))
    for raw in seed_family_specs:
        if "=" not in raw:
            raise ValueError(f"--seed-family-dir must be label=path, got: {raw}")
        label, path_text = raw.split("=", 1)
        path = Path(path_text)
        specs.append(
            (
                label,
                path,
                _build_seed_family_report(
                    path,
                    bootstrap_iter=bootstrap_iter,
                    bootstrap_seed=bootstrap_seed,
                ),
            )
        )
    return specs


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: Any, digits: int = 4) -> str:
    val = _f(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _pct(value: Any) -> str:
    val = _f(value)
    if not math.isfinite(val):
        return "NA"
    return f"{100.0 * val:.1f}%"


def _mean(values: list[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _bootstrap_ci(values: list[float], *, n_iter: int, seed: int, statistic: str) -> tuple[float, float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_iter), dtype=float)
    for idx in range(int(n_iter)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[idx] = float(np.median(sample) if statistic == "median" else np.mean(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _signed_rank_biserial_effect(deltas: list[float]) -> float:
    values = [float(value) for value in deltas if math.isfinite(float(value)) and abs(float(value)) > 1e-12]
    if not values:
        return float("nan")
    ranks = rankdata([abs(value) for value in values], method="average")
    improved_rank_sum = float(sum(rank for rank, value in zip(ranks, values) if value < 0.0))
    worsened_rank_sum = float(sum(rank for rank, value in zip(ranks, values) if value > 0.0))
    rank_total = improved_rank_sum + worsened_rank_sum
    return (improved_rank_sum - worsened_rank_sum) / rank_total if rank_total else float("nan")


def _wilcoxon_stats(deltas: list[float]) -> dict[str, Any]:
    values = [float(value) for value in deltas if math.isfinite(float(value)) and abs(float(value)) > 1e-12]
    effect = _signed_rank_biserial_effect(values)
    if not values:
        return {
            "wilcoxon_n_nonzero": 0,
            "wilcoxon_statistic_less": float("nan"),
            "wilcoxon_p_less_method_lt_raw": float("nan"),
            "wilcoxon_statistic_two_sided": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
            "signed_rank_biserial_effect_method_lt_raw": effect,
            "wilcoxon_status": "no_nonzero_deltas",
        }
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


def _build_seed_family_report(
    metric_dir: Path,
    *,
    bootstrap_iter: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    target_regions: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(metric_dir.glob("seed_*_best-selection_test.json")):
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        seed_id = path.stem.split("_")[1] if "_" in path.stem else path.stem
        for sample in report.get("samples") or []:
            sample_id = str(sample.get("sample_id") or "")
            for region, metrics in (sample.get("regions") or {}).items():
                raw = _f(metrics.get("zero_error_rms"))
                method = _f(metrics.get("prediction_error_rms"))
                if not math.isfinite(raw) or not math.isfinite(method):
                    continue
                row = target_regions.setdefault(
                    (sample_id, region),
                    {
                        "sample_id": sample_id,
                        "region": region,
                        "raw": [],
                        "method": [],
                        "seed_ids": [],
                        "n_residues": metrics.get("n_residues", ""),
                    },
                )
                row["raw"].append(raw)
                row["method"].append(method)
                row["seed_ids"].append(seed_id)

    samples_by_id: dict[str, dict[str, Any]] = {}
    deltas_by_region: dict[str, list[float]] = {}
    for (_sample_id, _region), row in sorted(target_regions.items()):
        raw = _mean(row["raw"])
        method = _mean(row["method"])
        delta = method - raw
        deltas_by_region.setdefault(row["region"], []).append(delta)
        sample = samples_by_id.setdefault(row["sample_id"], {"sample_id": row["sample_id"], "regions": {}})
        sample["regions"][row["region"]] = {
            "zero_error_rms": raw,
            "prediction_error_rms": method,
            "n_residues": row["n_residues"],
            "n_seeds": len(set(row["seed_ids"])),
        }

    paired: dict[str, dict[str, Any]] = {}
    for region, deltas in sorted(deltas_by_region.items()):
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
        paired[region] = {
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
    return {
        "task": "seed_family_paired_delta",
        "prediction_source": str(metric_dir),
        "metric_contract": {
            "paired_delta_rmsd": "5-seed mean method RMSD minus raw AF2 RMSD per target; negative is better",
        },
        "paired_delta_by_region": paired,
        "samples": list(samples_by_id.values()),
    }


def _paired_effect_by_region(report: dict[str, Any]) -> dict[str, float]:
    deltas_by_region: dict[str, list[float]] = {}
    for sample in report.get("samples") or []:
        for region, metrics in (sample.get("regions") or {}).items():
            raw = _f(metrics.get("zero_error_rms"))
            method = _f(metrics.get("prediction_error_rms"))
            if math.isfinite(raw) and math.isfinite(method):
                deltas_by_region.setdefault(region, []).append(method - raw)
    return {region: _signed_rank_biserial_effect(deltas) for region, deltas in deltas_by_region.items()}


def _summary_rows(specs: list[tuple[str, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path, report in specs:
        aggregate = report.get("aggregate_by_region") or {}
        paired = report.get("paired_delta_by_region") or {}
        effect_by_region = _paired_effect_by_region(report)
        for region in REPORT_REGIONS:
            agg = aggregate.get(region)
            pair = paired.get(region)
            if not agg and not pair:
                continue
            rows.append(
                {
                    "method": label,
                    "region": region,
                    "prediction_error_rms": (agg or {}).get("prediction_error_rms", ""),
                    "raw_af2_rmsd": (agg or {}).get("zero_error_rms", ""),
                    "mse_improvement_vs_zero_fraction": (agg or {}).get("mse_improvement_vs_zero_fraction", ""),
                    "predicted_displacement_mean": (agg or {}).get("predicted_displacement_mean", ""),
                    "n_targets": (pair or {}).get("n_targets", ""),
                    "n_improved": (pair or {}).get("n_improved", ""),
                    "n_worsened": (pair or {}).get("n_worsened", ""),
                    "improved_fraction": (pair or {}).get("improved_fraction", ""),
                    "median_delta_rmsd_method_minus_raw": (pair or {}).get(
                        "median_delta_rmsd_method_minus_raw",
                        "",
                    ),
                    "median_delta_ci95_low": (pair or {}).get("median_delta_ci95_low", ""),
                    "median_delta_ci95_high": (pair or {}).get("median_delta_ci95_high", ""),
                    "wilcoxon_p_less_method_lt_raw": (pair or {}).get("wilcoxon_p_less_method_lt_raw", ""),
                    "signed_rank_biserial_effect_method_lt_raw": (pair or {}).get(
                        "signed_rank_biserial_effect_method_lt_raw",
                        effect_by_region.get(region, ""),
                    ),
                    "source_json": str(path),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
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
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_markdown(specs: list[tuple[str, Path, dict[str, Any]]], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# TBDT Mechanistic Evaluation",
        "",
        "Delta RMSD is defined as `RMSD(method) - RMSD(raw AFDB/AF2)`, so negative values are improvements.",
        "All paired statistics are per target, not residue-pooled.",
        "",
        "## Per-Target Paired Delta",
        "",
    ]
    paired_rows = []
    method_order = {"primary_single_5seed_family": 0, "primary_single_seed404": 1, "scaffold_blend": 2}
    region_order = {region: idx for idx, region in enumerate(REPORT_REGIONS)}
    sorted_rows = sorted(
        rows,
        key=lambda item: (
            method_order.get(str(item.get("method")), 99),
            region_order.get(str(item.get("region")), 99),
        ),
    )
    for row in sorted_rows:
        if row["region"] not in {"eval", "plug", "plug_apical_loop", "tonb_box", "barrel_core", "all"}:
            continue
        paired_rows.append(
            [
                row["method"],
                row["region"],
                f"{row.get('n_improved', '')}/{row.get('n_targets', '')}",
                str(row.get("n_worsened", "")),
                _fmt(row.get("median_delta_rmsd_method_minus_raw")),
                f"[{_fmt(row.get('median_delta_ci95_low'))}, {_fmt(row.get('median_delta_ci95_high'))}]",
                _fmt(row.get("wilcoxon_p_less_method_lt_raw"), 3),
                _fmt(row.get("signed_rank_biserial_effect_method_lt_raw"), 3),
            ]
        )
    lines.append(
        _table(
            [
                "method",
                "region",
                "improved/targets",
                "worsened",
                "median Delta A",
                "bootstrap 95% CI",
                "Wilcoxon p",
                "signed-rank effect",
            ],
            paired_rows,
        )
    )
    lines.extend(["", "## Fine Region Aggregate", ""])
    fine_rows = []
    for row in sorted_rows:
        if row["region"] not in {"plug", "plug_core", "plug_apical_loop", "plug_extension_nt", "tonb_box"}:
            continue
        fine_rows.append(
            [
                row["method"],
                row["region"],
                _fmt(row.get("raw_af2_rmsd")),
                _fmt(row.get("prediction_error_rms")),
                _pct(row.get("mse_improvement_vs_zero_fraction")),
                _fmt(row.get("predicted_displacement_mean")),
            ]
        )
    lines.append(_table(["method", "region", "raw RMSD A", "method RMSD A", "MSE gain", "pred disp mean A"], fine_rows))
    lines.extend(["", "## TonB State Metrics", ""])
    tonb_rows = []
    for label, _path, report in specs:
        summary = report.get("tonb_state_summary") or {}
        if not summary:
            continue
        tonb_rows.append(
            [
                label,
                str(summary.get("n_targets", "")),
                _pct(summary.get("tonb_state_accuracy")),
                _pct(summary.get("tonb_direction_compatible_rate")),
                _fmt(summary.get("tonb_centroid_displacement_cosine_median")),
                _fmt(summary.get("tonb_exposure_delta_error_median")),
                json.dumps(summary.get("tonb_state_confusion", {}), sort_keys=True),
            ]
        )
    lines.append(
        _table(
            [
                "method",
                "n TonB targets",
                "exposure-state accuracy",
                "direction-compatible rate",
                "median disp cosine",
                "median exposure-delta error A",
                "state confusion",
            ],
            tonb_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `plug_apical_loop` is a sequence-order diagnostic split derived from the C-terminal fraction of the plug mask unless an explicit mask exists.",
            "- `plug_extension_nt` is the N-terminal plug window or the plug-overlapping TonB-box when available; current models do not improve this subregion.",
            "- TonB coordinate RMSD can improve while exposure-state classification stays weak. Report both; do not infer TonB compatibility from RMSD alone.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = _load_specs(
        args.metric_json,
        args.seed_family_dir,
        bootstrap_iter=int(args.bootstrap_iter),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    rows = _summary_rows(specs)
    _write_csv(rows, Path(args.out_csv))
    _write_markdown(specs, rows, Path(args.out_md))
    print(json.dumps({"out_md": args.out_md, "out_csv": args.out_csv, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
