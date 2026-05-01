"""Build the baseline comparison figure for Cooper-TBDT."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evopoint_da.figures.io import as_float, finite_float, read_csv_rows, write_csv_rows
from evopoint_da.figures.style import (
    add_panel_label,
    apply_style,
    clean_axis,
    figure_output_dir,
    get_style,
    parse_formats,
    save_figure,
)


REGIONS = ("eval", "plug", "tonb_box")
REGION_TITLES = {
    "eval": "Evaluation-region RMSD",
    "plug": "Plug RMSD",
    "tonb_box": "TonB-box RMSD",
}
METHODS = (
    {
        "source": "coord",
        "method": "raw_af2_zero",
        "label": "Raw AFDB",
        "color_key": "raw",
    },
    {
        "source": "coord",
        "method": "foldseek_nearest_template",
        "label": "Foldseek transfer",
        "color_key": "sakura",
    },
    {
        "source": "coord",
        "method": "usalign_nearest_template",
        "label": "US-align transfer",
        "color_key": "blossom",
    },
    {
        "source": "coord",
        "method": "nearest_template",
        "label": "nearest-template",
        "color_key": "sakura",
    },
    {
        "source": "coord",
        "method": "family_state_average",
        "label": "family/state average",
        "color_key": "celadon",
    },
    {
        "source": "coord",
        "method": "barrel_frame_ridge",
        "label": "barrel-frame ridge",
        "color_key": "baseline",
    },
    {
        "source": "seed",
        "method": "cooper_tbdt_scaffold_single_5seed_mean",
        "label": "Cooper-TBDT single",
        "color_key": "primary",
    },
    {
        "source": "coord",
        "method": "cooper_tbdt_scaffold_blend",
        "label": "Cooper-TBDT blend",
        "color_key": "blend",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT baseline comparison figure")
    parser.add_argument(
        "--publication-dir",
        default="artifacts/tbdt_v1/publication_report",
        help="Directory produced by build_tbdt_publication_report.",
    )
    parser.add_argument(
        "--seed-aggregate-csv",
        default="artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv",
        help="Primary 5-seed aggregate CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="baseline_comparison", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _first_row(rows: list[dict[str, str]], *, source: str, **matches: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    match_text = ", ".join(f"{key}={value}" for key, value in matches.items())
    raise ValueError(f"No row in {source} matching {match_text}")


def _seed_metric(seed_rows: list[dict[str, str]], path: str | Path, *, region: str, metric: str) -> tuple[float, float]:
    row = _first_row(seed_rows, source=str(path), split="test", region=region, metric=metric)
    return (
        finite_float(row.get("mean"), field="mean", source=str(path)),
        as_float(row.get("std"), 0.0),
    )


def _load_plot_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    coord_path = Path(args.publication_dir) / "coordinate_metrics_summary.csv"
    seed_path = Path(args.seed_aggregate_csv)
    coord_rows = read_csv_rows(coord_path)
    seed_rows = read_csv_rows(seed_path)

    plot_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for region in REGIONS:
            if method["source"] == "seed":
                rmsd, rmsd_sd = _seed_metric(seed_rows, seed_path, region=region, metric="prediction_error_rms")
                improvement, improvement_sd = _seed_metric(
                    seed_rows,
                    seed_path,
                    region=region,
                    metric="mse_improvement_vs_zero_fraction",
                )
                category = "primary_model_family"
                source_path = str(seed_path)
            else:
                row = _first_row(
                    coord_rows,
                    source=str(coord_path),
                    method=str(method["method"]),
                    region=region,
                )
                rmsd = finite_float(row.get("prediction_error_rms"), field="prediction_error_rms", source=str(coord_path))
                rmsd_sd = 0.0
                improvement = as_float(row.get("mse_improvement_vs_zero_fraction"))
                if not math.isfinite(improvement) and method["method"] == "raw_af2_zero":
                    improvement = 0.0
                if not math.isfinite(improvement):
                    raise ValueError(
                        "Missing mse_improvement_vs_zero_fraction "
                        f"for method={method['method']} region={region} in {coord_path}"
                    )
                improvement_sd = 0.0
                category = row.get("category", "")
                source_path = str(coord_path)

            plot_rows.append(
                {
                    "method": method["method"],
                    "method_label": method["label"],
                    "category": category,
                    "region": region,
                    "region_label": REGION_TITLES[region],
                    "prediction_error_rms": rmsd,
                    "prediction_error_rms_sd": rmsd_sd,
                    "eval_mse_improvement_vs_raw_afdb_fraction": improvement if region == "eval" else "",
                    "eval_mse_improvement_vs_raw_afdb_fraction_sd": improvement_sd if region == "eval" else "",
                    "source": source_path,
                }
            )
    return plot_rows


def _rows_for_region(rows: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    order = {str(method["method"]): idx for idx, method in enumerate(METHODS)}
    selected = [row for row in rows if row["region"] == region]
    return sorted(selected, key=lambda row: order[str(row["method"])])


def _method_color(row: dict[str, Any], style: Any) -> str:
    method_id = str(row["method"])
    for method in METHODS:
        if method["method"] == method_id:
            return style.palette[str(method["color_key"])]
    return style.palette["neutral"]


def _plot_rmsd_panel(ax: object, rows: list[dict[str, Any]], region: str, style: Any, *, show_ylabel: bool) -> None:
    selected = _rows_for_region(rows, region)
    labels = [str(row["method_label"]) for row in selected]
    values = np.array([float(row["prediction_error_rms"]) for row in selected], dtype=float)
    errors = np.array([float(row["prediction_error_rms_sd"] or 0.0) for row in selected], dtype=float)
    y = np.arange(len(selected), dtype=float)
    colors = [_method_color(row, style) for row in selected]

    ax.barh(y, values, height=0.66, color=colors, edgecolor="white", linewidth=0.75)
    has_error = errors > 0
    if np.any(has_error):
        ax.errorbar(
            values[has_error],
            y[has_error],
            xerr=errors[has_error],
            fmt="none",
            ecolor=style.palette["text"],
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
        )

    raw_value = float(selected[0]["prediction_error_rms"])
    ax.axvline(raw_value, color=style.palette["reference"], linestyle="--", linewidth=0.8, alpha=0.78)

    x_max = float(np.max(values + errors)) * 1.15
    x_pad = x_max * 0.014
    for yi, value in zip(y, values):
        ax.text(value + x_pad, yi, f"{value:.2f}", ha="left", va="center", fontsize=6.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_ylabel else [])
    ax.invert_yaxis()
    ax.set_xlabel("RMSD (Å)")
    ax.set_title(REGION_TITLES[region])
    ax.set_xlim(0.0, x_max)
    clean_axis(ax, style, grid_axis="x")


def _plot_improvement_panel(ax: object, rows: list[dict[str, Any]], style: Any, *, show_ylabel: bool) -> None:
    selected = _rows_for_region(rows, "eval")
    labels = [str(row["method_label"]) for row in selected]
    values = np.array(
        [float(row["eval_mse_improvement_vs_raw_afdb_fraction"]) * 100.0 for row in selected],
        dtype=float,
    )
    errors = np.array(
        [float(row["eval_mse_improvement_vs_raw_afdb_fraction_sd"] or 0.0) * 100.0 for row in selected],
        dtype=float,
    )
    y = np.arange(len(selected), dtype=float)

    colors: list[str] = []
    for row, value in zip(selected, values):
        method_id = str(row["method"])
        if method_id == "cooper_tbdt_scaffold_single_5seed_mean":
            colors.append(style.palette["primary"])
        elif method_id == "cooper_tbdt_scaffold_blend":
            colors.append(style.palette["blend"])
        elif method_id == "raw_af2_zero":
            colors.append(style.palette["raw"])
        elif value < 0:
            colors.append(style.palette["worsened"])
        else:
            colors.append(style.palette["improved"])

    ax.barh(y, values, height=0.66, color=colors, edgecolor="white", linewidth=0.75)
    has_error = errors > 0
    if np.any(has_error):
        ax.errorbar(
            values[has_error],
            y[has_error],
            xerr=errors[has_error],
            fmt="none",
            ecolor=style.palette["text"],
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
        )
    ax.axvline(0.0, color=style.palette["reference"], linewidth=0.9)

    x_min = float(np.min(values - errors))
    x_max = float(np.max(values + errors))
    x_range = max(x_max - x_min, 1.0)
    left = x_min - x_range * 0.18
    right = max(5.2, x_max + x_range * 0.18)
    x_pad = x_range * 0.035
    for yi, value in zip(y, values):
        if value < 0:
            ax.text(value - x_pad, yi, f"{value:+.1f}%", ha="right", va="center", fontsize=6.6)
        else:
            ax.text(value + x_pad, yi, f"{value:+.1f}%", ha="left", va="center", fontsize=6.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_ylabel else [])
    ax.invert_yaxis()
    ax.set_xlabel("MSE improvement vs raw AFDB (%)")
    ax.set_title("Evaluation-region MSE improvement")
    ax.set_xlim(left, right)
    clean_axis(ax, style, grid_axis="x")


def _write_plotting_data(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        out_dir / "baseline_comparison_values.csv",
        rows,
        [
            "method",
            "method_label",
            "category",
            "region",
            "region_label",
            "prediction_error_rms",
            "prediction_error_rms_sd",
            "eval_mse_improvement_vs_raw_afdb_fraction",
            "eval_mse_improvement_vs_raw_afdb_fraction_sd",
            "source",
        ],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    rows = _load_plot_rows(args)
    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_plotting_data(out_dir, rows)

    fig = plt.figure(figsize=(10.4, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_rmsd_panel(ax_a, rows, "eval", style, show_ylabel=True)
    _plot_rmsd_panel(ax_b, rows, "plug", style, show_ylabel=True)
    _plot_rmsd_panel(ax_c, rows, "tonb_box", style, show_ylabel=True)
    _plot_improvement_panel(ax_d, rows, style, show_ylabel=True)

    for label, ax in zip(("A", "B", "C", "D"), (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label, style)

    formats = parse_formats(args.formats)
    written = save_figure(fig, out_dir / args.out_name, formats=formats, dpi=int(args.dpi))
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    written = build_figure(args)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
