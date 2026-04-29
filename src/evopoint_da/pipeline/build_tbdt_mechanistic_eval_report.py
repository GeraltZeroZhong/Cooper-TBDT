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
    parser.add_argument("--out-md", default="artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_report.md")
    parser.add_argument("--out-csv", default="artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_summary.csv")
    return parser.parse_args()


def _load_specs(raw_specs: list[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    specs: list[tuple[str, Path, dict[str, Any]]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"--metric-json must be label=path, got: {raw}")
        label, path_text = raw.split("=", 1)
        path = Path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            specs.append((label, path, json.load(handle)))
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


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _summary_rows(specs: list[tuple[str, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path, report in specs:
        aggregate = report.get("aggregate_by_region") or {}
        paired = report.get("paired_delta_by_region") or {}
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
        "Delta RMSD is defined as `RMSD(method) - RMSD(raw AF2)`, so negative values are improvements.",
        "All paired statistics are per target, not residue-pooled.",
        "",
        "## Per-Target Paired Delta",
        "",
    ]
    paired_rows = []
    for row in rows:
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
            ]
        )
    lines.append(
        _table(
            ["method", "region", "improved/targets", "worsened", "median Delta A", "bootstrap 95% CI", "Wilcoxon p"],
            paired_rows,
        )
    )
    lines.extend(["", "## Fine Region Aggregate", ""])
    fine_rows = []
    for row in rows:
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
    specs = _load_specs(args.metric_json)
    rows = _summary_rows(specs)
    _write_csv(rows, Path(args.out_csv))
    _write_markdown(specs, rows, Path(args.out_md))
    print(json.dumps({"out_md": args.out_md, "out_csv": args.out_csv, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
