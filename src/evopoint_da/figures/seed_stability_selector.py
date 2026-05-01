"""Build the seed-stability and selector-sensitivity figure for Cooper-TBDT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evopoint_da.figures.io import finite_float, read_csv_rows, write_csv_rows
from evopoint_da.figures.style import (
    add_note_box,
    add_panel_label,
    apply_style,
    clean_axis,
    figure_output_dir,
    get_style,
    parse_formats,
    save_figure,
)


REGIONS = ("eval", "plug", "tonb_box")
REGION_LABELS = {"eval": "Evaluation", "plug": "Plug", "tonb_box": "TonB box"}
SELECTORS = ("best-selection", "best-disp1to2", "best-disp1to5", "best-flex")
SELECTOR_LABELS = {
    "best-selection": "selection",
    "best-disp1to2": "disp 1-2 Å",
    "best-disp1to5": "disp 1-5 Å",
    "best-flex": "flex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT seed-stability/selector figure")
    parser.add_argument(
        "--seed-summary-csv",
        default="artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_summary.csv",
        help="Best-selection seed-stability summary CSV.",
    )
    parser.add_argument(
        "--selector-summary-csv",
        default="artifacts/tbdt_v1/publication_report/selector_sensitivity_summary.csv",
        help="Selector-sensitivity summary CSV from publication report.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="seed_stability_selector_sensitivity", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _region_color(region: str, style: Any) -> str:
    return {
        "eval": style.palette["primary"],
        "plug": style.palette["blend"],
        "tonb_box": style.palette["worsened"],
    }[region]


def _selector_color(selector: str, style: Any) -> str:
    return {
        "best-selection": style.palette["primary"],
        "best-disp1to2": style.palette["baseline"],
        "best-disp1to5": style.palette["blend"],
        "best-flex": style.palette["worsened"],
    }[selector]


def _load_seed_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(path):
        if row.get("split") != "test":
            continue
        parsed: dict[str, Any] = {
            "model": row.get("model", ""),
            "seed": int(finite_float(row.get("seed"), field="seed", source=str(path))),
            "selector": row.get("selector", ""),
            "selected_epoch": int(finite_float(row.get("selected_epoch"), field="selected_epoch", source=str(path))),
            "score": finite_float(row.get("score"), field="score", source=str(path)),
            "source": str(path),
        }
        for region in REGIONS:
            parsed[f"{region}_prediction_error_rms"] = finite_float(
                row.get(f"{region}_prediction_error_rms"),
                field=f"{region}_prediction_error_rms",
                source=str(path),
            )
            parsed[f"{region}_zero_error_rms"] = finite_float(
                row.get(f"{region}_zero_error_rms"),
                field=f"{region}_zero_error_rms",
                source=str(path),
            )
            parsed[f"{region}_mse_improvement_vs_zero_fraction"] = finite_float(
                row.get(f"{region}_mse_improvement_vs_zero_fraction"),
                field=f"{region}_mse_improvement_vs_zero_fraction",
                source=str(path),
            )
        parsed["barrel_core_predicted_displacement_mean"] = finite_float(
            row.get("barrel_core_predicted_displacement_mean"),
            field="barrel_core_predicted_displacement_mean",
            source=str(path),
        )
        rows.append(parsed)
    if not rows:
        raise ValueError(f"No test rows found in {path}")
    return sorted(rows, key=lambda item: int(item["seed"]))


def _load_selector_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(path):
        selector = row.get("selector", "")
        region = row.get("region", "")
        metric = row.get("metric", "")
        if selector not in SELECTORS:
            continue
        if not ((region in REGIONS and metric == "prediction_error_rms") or (region == "barrel_core" and metric == "predicted_displacement_mean")):
            continue
        rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABELS[selector],
                "region": region,
                "metric": metric,
                "mean": finite_float(row.get("mean"), field="mean", source=str(path)),
                "std": finite_float(row.get("std"), field="std", source=str(path)),
                "n_seeds": int(finite_float(row.get("n_seeds"), field="n_seeds", source=str(path))),
                "source": str(path),
            }
        )
    return rows


def _selector_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(str(row["selector"]), str(row["region"]), str(row["metric"])): row for row in rows}


def _plot_seed_improvements(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    seeds = np.array([int(row["seed"]) for row in rows], dtype=int)
    for region in REGIONS:
        values = np.array([float(row[f"{region}_mse_improvement_vs_zero_fraction"]) * 100.0 for row in rows])
        ax.plot(
            seeds,
            values,
            marker="o",
            markersize=4.2,
            linewidth=1.25,
            color=_region_color(region, style),
            label=REGION_LABELS[region],
        )
    ax.axhline(0.0, color=style.palette["reference"], linewidth=0.9, linestyle="--")
    ax.set_xticks(seeds)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("MSE improvement vs raw AFDB (%)")
    ax.set_title("Best-selection seed stability")
    ax.legend(loc="upper right")
    clean_axis(ax, style)


def _plot_selected_epochs(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    seeds = np.array([int(row["seed"]) for row in rows], dtype=int)
    epochs = np.array([int(row["selected_epoch"]) for row in rows], dtype=float)
    scores = np.array([float(row["score"]) for row in rows], dtype=float) * 100.0
    x = np.arange(len(rows), dtype=float)
    ax.bar(x, epochs, color=style.palette["primary"], edgecolor="white", linewidth=0.8, width=0.62)
    for xi, epoch in zip(x, epochs):
        ax.text(xi, epoch + 0.25, f"{int(epoch)}", ha="center", va="bottom", fontsize=6.8)
    ax.plot(x, scores, color=style.palette["blend"], marker="o", linewidth=1.1, markersize=4.0, label="selector score x100")
    ax.axhline(0.0, color=style.palette["reference"], linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Selected epoch / score x100")
    ax.set_title("Validation-selected checkpoints")
    ax.legend(loc="upper right")
    ax.set_ylim(min(-1.0, float(scores.min()) - 0.7), max(float(epochs.max()) + 1.3, float(scores.max()) + 0.7))
    clean_axis(ax, style)


def _plot_selector_rmsd(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    lookup = _selector_lookup(rows)
    y = np.arange(len(SELECTORS), dtype=float)
    offsets = {"eval": -0.17, "plug": 0.17}
    for region in ("eval", "plug"):
        means = np.array([float(lookup[(selector, region, "prediction_error_rms")]["mean"]) for selector in SELECTORS])
        stds = np.array([float(lookup[(selector, region, "prediction_error_rms")]["std"]) for selector in SELECTORS])
        ax.errorbar(
            means,
            y + offsets[region],
            xerr=stds,
            fmt="o",
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.9,
            color=_region_color(region, style),
            label=REGION_LABELS[region],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([SELECTOR_LABELS[selector] for selector in SELECTORS])
    ax.invert_yaxis()
    ax.set_xlabel("RMSD (Å), mean +/- SD")
    ax.set_title("Selector sensitivity: eval/plug")
    ax.legend(loc="lower right")
    clean_axis(ax, style, grid_axis="x")


def _plot_tonb_tradeoff(ax: object, rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], style: Any) -> None:
    lookup = _selector_lookup(rows)
    raw_tonb = float(seed_rows[0]["tonb_box_zero_error_rms"])
    offsets = {
        "best-selection": (0.006, 0.003),
        "best-disp1to2": (0.006, -0.004),
        "best-disp1to5": (0.006, 0.004),
        "best-flex": (0.006, 0.004),
    }
    for selector in SELECTORS:
        tonb = lookup[(selector, "tonb_box", "prediction_error_rms")]
        barrel = lookup[(selector, "barrel_core", "predicted_displacement_mean")]
        x = float(tonb["mean"])
        y = float(barrel["mean"])
        ax.errorbar(
            [x],
            [y],
            xerr=[float(tonb["std"])],
            yerr=[float(barrel["std"])],
            fmt="o",
            markersize=4.8,
            capsize=2.0,
            color=_selector_color(selector, style),
            elinewidth=0.9,
        )
        dx, dy = offsets[selector]
        ax.text(x + dx, y + dy, SELECTOR_LABELS[selector], fontsize=6.7, va="center", ha="left")
    ax.axvline(raw_tonb, color=style.palette["reference"], linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color=style.palette["grid"], linestyle="--", linewidth=0.9)
    ax.text(raw_tonb - 0.004, 0.052, "raw TonB RMSD", ha="right", va="bottom", fontsize=6.4, color=style.palette["reference"])
    ax.set_xlabel("TonB-box RMSD (Å)")
    ax.set_ylabel("Barrel-core predicted mean (Å)")
    ax.set_title("TonB-selector tradeoff")
    ax.set_ylim(0.0, max(0.07, ax.get_ylim()[1]))
    clean_axis(ax, style)
    add_note_box(
        ax,
        "Lower TonB RMSD from flex-biased selection\nis a sensitivity result, not the primary claim.",
        style,
        x=0.03,
        y=0.97,
    )


def _write_outputs(out_dir: Path, seed_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]]) -> None:
    seed_fields = [
        "model",
        "seed",
        "selector",
        "selected_epoch",
        "score",
        "eval_prediction_error_rms",
        "eval_zero_error_rms",
        "eval_mse_improvement_vs_zero_fraction",
        "plug_prediction_error_rms",
        "plug_zero_error_rms",
        "plug_mse_improvement_vs_zero_fraction",
        "tonb_box_prediction_error_rms",
        "tonb_box_zero_error_rms",
        "tonb_box_mse_improvement_vs_zero_fraction",
        "barrel_core_predicted_displacement_mean",
        "source",
    ]
    write_csv_rows(out_dir / "seed_stability_by_seed_values.csv", seed_rows, seed_fields)
    write_csv_rows(
        out_dir / "seed_stability_selector_values.csv",
        selector_rows,
        ["selector", "selector_label", "region", "metric", "mean", "std", "n_seeds", "source"],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)
    out_dir = figure_output_dir(args.out_dir, args.out_name)

    seed_rows = _load_seed_rows(args.seed_summary_csv)
    selector_rows = _load_selector_rows(args.selector_summary_csv)
    _write_outputs(out_dir, seed_rows, selector_rows)

    fig = plt.figure(figsize=(8.8, 6.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    _plot_seed_improvements(ax_a, seed_rows, style)
    _plot_selected_epochs(ax_b, seed_rows, style)
    _plot_selector_rmsd(ax_c, selector_rows, style)
    _plot_tonb_tradeoff(ax_d, selector_rows, seed_rows, style)

    for label, ax in zip(("A", "B", "C", "D"), (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label, style)

    written = save_figure(fig, out_dir / args.out_name, formats=parse_formats(args.formats), dpi=int(args.dpi))
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    for path in build_figure(args):
        print(path)


if __name__ == "__main__":
    main()
