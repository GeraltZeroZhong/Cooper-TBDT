"""Summarize TBDT scaffold-prior ablation sweeps.

The report intentionally selects checkpoints by validation score, then reports
the paired test metrics. This keeps hyperparameter selection separate from the
test-set readout used in the manuscript table.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY_COLUMNS = (
    "variant",
    "group",
    "description",
    "selector",
    "val_score",
    "test_score",
    "test_eval_prediction_error_rms",
    "test_eval_mse_improvement_vs_zero_fraction",
    "test_plug_prediction_error_rms",
    "test_plug_mse_improvement_vs_zero_fraction",
    "test_tonb_box_prediction_error_rms",
    "test_tonb_box_mse_improvement_vs_zero_fraction",
    "test_barrel_core_prediction_error_rms",
    "test_barrel_core_predicted_displacement_mean",
    "test_barrel_core_mse_improvement_vs_zero_fraction",
    "checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scaffold-prior sweep CSV files.")
    parser.add_argument("--summary-csv", action="append", required=True, help="Sweep summary CSV. May repeat.")
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/scaffold_prior_sweep_report")
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _read_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def _f(row: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _selected_by_val(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("variant", ""), row.get("selector", ""))
        if row.get("split") == "val":
            val_by_variant[row.get("variant", "")].append(row)
        elif row.get("split") == "test":
            test_by_key[key] = row

    selected: list[dict[str, Any]] = []
    for variant, val_rows in sorted(val_by_variant.items()):
        best_val = max(val_rows, key=lambda row: _f(row, "score", -float("inf")))
        test = test_by_key.get((variant, best_val.get("selector", "")))
        if test is None:
            continue
        out: dict[str, Any] = {
            "variant": variant,
            "group": best_val.get("group", ""),
            "description": best_val.get("description", ""),
            "selector": best_val.get("selector", ""),
            "val_score": _f(best_val, "score"),
            "test_score": _f(test, "score"),
            "checkpoint": best_val.get("checkpoint", ""),
            "source_csv": best_val.get("source_csv", ""),
        }
        for region in ("eval", "plug", "tonb_box", "barrel_core", "all"):
            for metric in (
                "prediction_error_rms",
                "zero_error_rms",
                "mse_improvement_vs_zero_fraction",
                "predicted_displacement_mean",
                "sample_improvement_rate",
                "direction_cosine_mean",
                "magnitude_mae",
            ):
                out[f"val_{region}_{metric}"] = _f(best_val, f"{region}_{metric}")
                out[f"test_{region}_{metric}"] = _f(test, f"{region}_{metric}")
        selected.append(out)
    return selected


def _group_best(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_group[row.get("group", "")].append(row)
    return [
        max(group_rows, key=lambda row: _f(row, "val_score", -float("inf")))
        for _, group_rows in sorted(by_group.items())
    ]


def _mechanism_contrast(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {row["variant"]: row for row in selected}
    baseline = by_variant.get("sp00_no_prior")
    rows: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: _f(item, "val_score", -float("inf")), reverse=True):
        out = {key: row.get(key, "") for key in PRIMARY_COLUMNS}
        for metric in (
            "test_eval_prediction_error_rms",
            "test_plug_prediction_error_rms",
            "test_tonb_box_prediction_error_rms",
            "test_barrel_core_predicted_displacement_mean",
            "test_eval_mse_improvement_vs_zero_fraction",
            "test_plug_mse_improvement_vs_zero_fraction",
            "test_tonb_box_mse_improvement_vs_zero_fraction",
        ):
            out[f"delta_vs_no_prior_{metric}"] = _f(row, metric) - _f(baseline, metric) if baseline else float("nan")
        rows.append(out)
    return rows


def _write_tradeoff_plot(path: Path, selected: list[dict[str, Any]], top_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    points = [
        row
        for row in selected
        if math.isfinite(_f(row, "val_score")) and math.isfinite(_f(row, "test_barrel_core_predicted_displacement_mean"))
    ]
    if not points:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = [_f(row, "test_barrel_core_predicted_displacement_mean") for row in points]
    y = [_f(row, "val_score") for row in points]
    groups = sorted({row.get("group", "") for row in points})
    colors = {group: f"C{i % 10}" for i, group in enumerate(groups)}
    for group in groups:
        group_rows = [row for row in points if row.get("group", "") == group]
        ax.scatter(
            [_f(row, "test_barrel_core_predicted_displacement_mean") for row in group_rows],
            [_f(row, "val_score") for row in group_rows],
            label=group,
            s=34,
            alpha=0.82,
            color=colors[group],
        )
    for row in sorted(points, key=lambda item: _f(item, "val_score"), reverse=True)[:top_k]:
        ax.annotate(
            row["variant"].replace("sp", ""),
            (_f(row, "test_barrel_core_predicted_displacement_mean"), _f(row, "val_score")),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axvline(0.05, color="0.35", lw=1.0, ls="--", label="core pred 0.05 A")
    ax.set_xlabel("Test barrel-core predicted displacement mean (A)")
    ax.set_ylabel("Validation selection score")
    ax.set_title("TBDT Scaffold Prior Tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(path: Path, selected: list[dict[str, Any]], group_best: list[dict[str, Any]], top_k: int) -> None:
    ranked = sorted(selected, key=lambda row: _f(row, "val_score", -float("inf")), reverse=True)
    best = ranked[0] if ranked else None
    full_region_focus = [
        row
        for row in ranked
        if row.get("group") == "winner_focus" and "region_anchor" in row.get("variant", "")
    ]
    default_candidate = full_region_focus[0] if full_region_focus else None

    lines = [
        "# TBDT Region-Aware Scaffold Prior Sweep",
        "",
        "Selection rule: choose each variant/checkpoint selector by validation score, then report the paired test metrics.",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Recommended Variant",
                "",
                f"- Variant: `{best['variant']}`",
                f"- Selector: `{best['selector']}`",
                f"- Mechanism: {best.get('description', '')}",
                f"- Validation score: {_fmt(_f(best, 'val_score'), 6)}",
                f"- Test eval RMS error: {_fmt(_f(best, 'test_eval_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(best, 'test_eval_mse_improvement_vs_zero_fraction'), 2)}% MSE improvement vs raw AF2)",
                f"- Test plug RMS error: {_fmt(_f(best, 'test_plug_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(best, 'test_plug_mse_improvement_vs_zero_fraction'), 2)}%)",
                f"- Test TonB-box RMS error: {_fmt(_f(best, 'test_tonb_box_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(best, 'test_tonb_box_mse_improvement_vs_zero_fraction'), 2)}%)",
                f"- Test barrel-core predicted displacement mean: "
                f"{_fmt(_f(best, 'test_barrel_core_predicted_displacement_mean'))} A",
                "",
            ]
        )
    if default_candidate and default_candidate is not best:
        lines.extend(
            [
                "## Balanced Default Candidate",
                "",
                "This is the top validation-selected winner-focused variant that keeps the full plug/eval/TonB/"
                "substrate-contact region prior active.",
                "",
                f"- Variant: `{default_candidate['variant']}`",
                f"- Selector: `{default_candidate['selector']}`",
                f"- Mechanism: {default_candidate.get('description', '')}",
                f"- Validation score: {_fmt(_f(default_candidate, 'val_score'), 6)}",
                f"- Test eval RMS error: {_fmt(_f(default_candidate, 'test_eval_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(default_candidate, 'test_eval_mse_improvement_vs_zero_fraction'), 2)}% MSE improvement vs raw AF2)",
                f"- Test plug RMS error: {_fmt(_f(default_candidate, 'test_plug_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(default_candidate, 'test_plug_mse_improvement_vs_zero_fraction'), 2)}%)",
                f"- Test TonB-box RMS error: {_fmt(_f(default_candidate, 'test_tonb_box_prediction_error_rms'))} A "
                f"({_fmt(100.0 * _f(default_candidate, 'test_tonb_box_mse_improvement_vs_zero_fraction'), 2)}%)",
                f"- Test barrel-core predicted displacement mean: "
                f"{_fmt(_f(default_candidate, 'test_barrel_core_predicted_displacement_mean'))} A",
                "",
            ]
        )

    lines.extend(["## Top Validation-Selected Variants", ""])
    lines.append(
        "| rank | variant | selector | val score | test eval RMS | test plug RMS | test TonB RMS | core pred mean |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row['variant']}`",
                    f"`{row['selector']}`",
                    _fmt(_f(row, "val_score"), 6),
                    _fmt(_f(row, "test_eval_prediction_error_rms")),
                    _fmt(_f(row, "test_plug_prediction_error_rms")),
                    _fmt(_f(row, "test_tonb_box_prediction_error_rms")),
                    _fmt(_f(row, "test_barrel_core_predicted_displacement_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.extend(["## Best By Ablation Group", ""])
    lines.append("| group | variant | val score | test eval MSE gain | test plug MSE gain | test TonB MSE gain | core pred mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(group_best, key=lambda item: item.get("group", "")):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("group", ""),
                    f"`{row['variant']}`",
                    _fmt(_f(row, "val_score"), 6),
                    _fmt(100.0 * _f(row, "test_eval_mse_improvement_vs_zero_fraction"), 2) + "%",
                    _fmt(100.0 * _f(row, "test_plug_mse_improvement_vs_zero_fraction"), 2) + "%",
                    _fmt(100.0 * _f(row, "test_tonb_box_mse_improvement_vs_zero_fraction"), 2) + "%",
                    _fmt(_f(row, "test_barrel_core_predicted_displacement_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.extend(
        [
            "## Mechanistic Readout",
            "",
            "- The scaffold prior should be reported as a tradeoff: functional-region MSE improvement versus "
            "barrel-core predicted displacement. The latter is the direct readout of whether the model is "
            "preserving the reliable beta-barrel scaffold instead of spending capacity on core motion.",
            "- Region weights are not automatically beneficial. Compare the component variants before claiming "
            "plug/TonB/substrate-contact specificity.",
            "- The preferred manuscript metric remains region-specific displacement recovery; docking and "
            "full-chain RMSD should stay secondary.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = _read_rows(args.summary_csv)
    selected = _selected_by_val(rows)
    selected.sort(key=lambda row: _f(row, "val_score", -float("inf")), reverse=True)
    group_best = _group_best(selected)
    contrast = _mechanism_contrast(selected)

    _write_csv(out_dir / "scaffold_prior_best_by_variant.csv", selected)
    _write_csv(out_dir / "scaffold_prior_group_best.csv", group_best, columns=PRIMARY_COLUMNS)
    _write_csv(out_dir / "scaffold_prior_mechanism_contrast.csv", contrast)
    _write_markdown(out_dir / "scaffold_prior_sweep_report.md", selected, group_best, int(args.top_k))
    _write_tradeoff_plot(out_dir / "scaffold_prior_tradeoff.png", selected, int(args.top_k))

    print(f"Wrote {len(selected)} validation-selected variants to {out_dir}")


if __name__ == "__main__":
    main()
