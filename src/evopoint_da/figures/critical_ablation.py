"""Build the critical neural ablation figure for Cooper-TBDT."""

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
    "eval": "Evaluation",
    "plug": "Plug",
    "tonb_box": "TonB box",
}
VARIANTS = (
    ("raw_afdb_zero", "Raw AFDB"),
    ("full_scaffold_prior", "Full scaffold prior"),
    ("no_state_conditioning", "No state conditioning"),
    ("no_tbdt_conditioning", "No TBDT conditioning"),
    ("no_afdb_confidence_features", "No AFDB confidence features"),
    ("no_region_loss_weights", "No region loss weights"),
    ("no_scaffold_anchor", "No scaffold anchor"),
)
MODEL_VARIANTS = tuple(variant for variant in VARIANTS if variant[0] != "raw_afdb_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT critical ablation figure")
    parser.add_argument(
        "--ablation-dir",
        default="artifacts/tbdt_v1/critical_neural_ablation",
        help="Directory produced by the critical neural ablation runner.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="critical_ablation", help="Output filename stem.")
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


def _non_nan(value: Any, default: float = 0.0) -> float:
    parsed = as_float(value)
    return parsed if math.isfinite(parsed) else default


def _variant_color(variant: str, style: Any) -> str:
    if variant == "raw_afdb_zero":
        return style.palette["raw"]
    if variant == "full_scaffold_prior":
        return style.palette["primary"]
    if variant == "no_scaffold_anchor":
        return style.palette["instagram"]
    if variant == "no_region_loss_weights":
        return style.palette["blossom"]
    if variant == "no_afdb_confidence_features":
        return style.palette["celadon"]
    if variant == "no_state_conditioning":
        return style.palette["peacock"]
    if variant == "no_tbdt_conditioning":
        return style.palette["aurora"]
    return style.palette["neutral"]


def _load_region_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.ablation_dir) / "critical_region_seed_summary.csv"
    rows = read_csv_rows(path)
    test_by_variant = {
        row["variant"]: row
        for row in rows
        if row.get("split") == "test" and row.get("variant") in {variant for variant, _ in VARIANTS}
    }
    missing = [variant for variant, _ in VARIANTS if variant not in test_by_variant]
    if missing:
        raise ValueError(f"Missing test rows in {path}: {', '.join(missing)}")

    out: list[dict[str, Any]] = []
    for variant, label in VARIANTS:
        row = test_by_variant[variant]
        for region in REGIONS:
            out.append(
                {
                    "variant": variant,
                    "variant_label": label,
                    "region": region,
                    "region_label": REGION_TITLES[region],
                    "prediction_error_rms_mean": finite_float(
                        row.get(f"{region}_prediction_error_rms_mean"),
                        field=f"{region}_prediction_error_rms_mean",
                        source=str(path),
                    ),
                    "prediction_error_rms_sd": _non_nan(row.get(f"{region}_prediction_error_rms_sd")),
                    "barrel_core_predicted_displacement_mean": finite_float(
                        row.get("barrel_core_predicted_displacement_mean_mean"),
                        field="barrel_core_predicted_displacement_mean_mean",
                        source=str(path),
                    ),
                    "barrel_core_predicted_displacement_sd": _non_nan(
                        row.get("barrel_core_predicted_displacement_mean_sd")
                    ),
                    "source": str(path),
                }
            )
    return out


def _load_paired_count_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.ablation_dir) / "critical_seed_mean_paired_summary.csv"
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for variant, label in MODEL_VARIANTS:
        for region in ("eval", "plug"):
            row = _first_row(rows, source=str(path), variant=variant, split="test", region=region)
            out.append(
                {
                    "variant": variant,
                    "variant_label": label,
                    "region": region,
                    "region_label": REGION_TITLES[region],
                    "n_targets": int(finite_float(row.get("n_targets"), field="n_targets", source=str(path))),
                    "n_improved": int(finite_float(row.get("n_improved"), field="n_improved", source=str(path))),
                    "n_worsened": int(finite_float(row.get("n_worsened"), field="n_worsened", source=str(path))),
                    "n_tied": int(finite_float(row.get("n_tied"), field="n_tied", source=str(path))),
                    "improved_fraction": finite_float(row.get("improved_fraction"), field="improved_fraction", source=str(path)),
                    "median_delta_rmsd_method_minus_raw": finite_float(
                        row.get("median_delta_rmsd_method_minus_raw"),
                        field="median_delta_rmsd_method_minus_raw",
                        source=str(path),
                    ),
                    "source": str(path),
                }
            )
    return out


def _region_rows_for(region_rows: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    order = {variant: idx for idx, (variant, _) in enumerate(VARIANTS)}
    return sorted([row for row in region_rows if row["region"] == region], key=lambda row: order[str(row["variant"])])


def _barrel_rows(region_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {variant: idx for idx, (variant, _) in enumerate(VARIANTS)}
    by_variant: dict[str, dict[str, Any]] = {}
    for row in region_rows:
        by_variant.setdefault(str(row["variant"]), row)
    return [by_variant[variant] for variant, _ in sorted(VARIANTS, key=lambda item: order[item[0]])]


def _region_axis_limits(region: str, values: np.ndarray, errors: np.ndarray) -> tuple[float, float, bool]:
    if region not in set(REGIONS):
        return 0.0, float(np.max(values + errors)) * 1.14, False
    low = float(np.min(values - errors))
    high = float(np.max(values + errors))
    span = max(high - low, 1e-6)
    pad = max(span * 0.18, 0.004)
    return max(0.0, low - pad), high + pad, True


def _add_x_axis_break(ax: object, style: Any) -> None:
    kwargs = {
        "transform": ax.transAxes,
        "color": style.palette["text"],
        "clip_on": False,
        "linewidth": 1.0,
        "solid_capstyle": "round",
    }
    ax.plot((-0.010, 0.014), (-0.018, 0.022), **kwargs)
    ax.plot((0.010, 0.034), (-0.018, 0.022), **kwargs)


def _plot_region_panel(ax: object, rows: list[dict[str, Any]], region: str, style: Any, *, show_labels: bool) -> None:
    selected = _region_rows_for(rows, region)
    values = np.array([float(row["prediction_error_rms_mean"]) for row in selected], dtype=float)
    errors = np.array([float(row["prediction_error_rms_sd"]) for row in selected], dtype=float)
    y = np.arange(len(selected), dtype=float)
    colors = [_variant_color(str(row["variant"]), style) for row in selected]

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
    ax.axvline(values[0], color=style.palette["reference"], linestyle="--", linewidth=0.8, alpha=0.78)

    x_min, x_max, truncated = _region_axis_limits(region, values, errors)
    x_range = max(x_max - x_min, 1e-6)
    x_pad = x_range * 0.025
    value_fmt = "{:.3f}" if region in {"eval", "plug"} else "{:.2f}"
    for yi, value, error in zip(y, values, errors):
        label_x = value + max(float(error), 0.0) + x_pad
        ax.text(label_x, yi, value_fmt.format(value), ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels([str(row["variant_label"]) for row in selected] if show_labels else [])
    ax.invert_yaxis()
    ax.set_xlabel("RMSD (Å; truncated axis)" if truncated else "RMSD (Å)")
    ax.set_title(REGION_TITLES[region])
    ax.set_xlim(x_min, x_max)
    clean_axis(ax, style, grid_axis="x")
    if truncated:
        _add_x_axis_break(ax, style)


def _plot_barrel_panel(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    selected = _barrel_rows(rows)
    values = np.array([float(row["barrel_core_predicted_displacement_mean"]) for row in selected], dtype=float)
    errors = np.array([float(row["barrel_core_predicted_displacement_sd"]) for row in selected], dtype=float)
    y = np.arange(len(selected), dtype=float)
    colors = [_variant_color(str(row["variant"]), style) for row in selected]

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
    ax.axvline(0.05, color=style.palette["reference"], linestyle="--", linewidth=0.9)
    ax.text(0.052, -0.48, "0.05 Å reference", ha="left", va="center", fontsize=6.5, color=style.palette["reference"])

    x_max = max(0.3, float(np.max(values + errors)) * 1.18)
    for yi, value in zip(y, values):
        ax.text(value + x_max * 0.018, yi, f"{value:.3f}", ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels([str(row["variant_label"]) for row in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Predicted displacement mean (Å)")
    ax.set_title("Barrel-core predicted displacement")
    ax.set_xlim(0.0, x_max)
    clean_axis(ax, style, grid_axis="x")


def _paired_rows_for(rows: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    order = {variant: idx for idx, (variant, _) in enumerate(MODEL_VARIANTS)}
    return sorted([row for row in rows if row["region"] == region], key=lambda row: order[str(row["variant"])])


def _plot_paired_counts(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    eval_rows = _paired_rows_for(rows, "eval")
    plug_rows = _paired_rows_for(rows, "plug")
    labels = [str(row["variant_label"]) for row in eval_rows]
    y = np.arange(len(eval_rows), dtype=float)
    height = 0.28
    offsets = {"eval": -0.17, "plug": 0.17}
    region_colors = {"eval": style.palette["primary"], "plug": style.palette["blend"]}

    for region, selected in (("eval", eval_rows), ("plug", plug_rows)):
        improved = np.array([float(row["n_improved"]) for row in selected], dtype=float)
        total = np.array([float(row["n_targets"]) for row in selected], dtype=float)
        remaining = total - improved
        ypos = y + offsets[region]
        ax.barh(
            ypos,
            improved,
            height=height,
            color=region_colors[region],
            edgecolor="white",
            linewidth=0.7,
            label=REGION_TITLES[region],
        )
        ax.barh(
            ypos,
            remaining,
            height=height,
            left=improved,
            color=style.palette["glacier"],
            edgecolor="white",
            linewidth=0.7,
        )
        for yi, imp, tot in zip(ypos, improved, total):
            ax.text(tot + 0.35, yi, f"{int(imp)}/{int(tot)}", ha="left", va="center", fontsize=6.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Improved targets vs raw AFDB")
    ax.set_title("Paired improvement counts")
    ax.set_xlim(0.0, 22.5)
    ax.text(
        0.02,
        0.98,
        "colored = improved; pale = not improved",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=style.palette["reference"],
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), ncol=2)
    clean_axis(ax, style, grid_axis="x")


def _write_plotting_data(out_dir: Path, region_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        out_dir / "critical_ablation_region_values.csv",
        region_rows,
        [
            "variant",
            "variant_label",
            "region",
            "region_label",
            "prediction_error_rms_mean",
            "prediction_error_rms_sd",
            "barrel_core_predicted_displacement_mean",
            "barrel_core_predicted_displacement_sd",
            "source",
        ],
    )
    write_csv_rows(
        out_dir / "critical_ablation_paired_counts.csv",
        paired_rows,
        [
            "variant",
            "variant_label",
            "region",
            "region_label",
            "n_targets",
            "n_improved",
            "n_worsened",
            "n_tied",
            "improved_fraction",
            "median_delta_rmsd_method_minus_raw",
            "source",
        ],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    region_rows = _load_region_rows(args)
    paired_rows = _load_paired_count_rows(args)
    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_plotting_data(out_dir, region_rows, paired_rows)

    fig = plt.figure(figsize=(10.4, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0, 1.0])
    ax_a1 = fig.add_subplot(grid[0, 0])
    ax_a2 = fig.add_subplot(grid[0, 1])
    ax_a3 = fig.add_subplot(grid[0, 2])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1:])

    _plot_region_panel(ax_a1, region_rows, "eval", style, show_labels=True)
    _plot_region_panel(ax_a2, region_rows, "plug", style, show_labels=False)
    _plot_region_panel(ax_a3, region_rows, "tonb_box", style, show_labels=False)
    _plot_barrel_panel(ax_b, region_rows, style)
    _plot_paired_counts(ax_c, paired_rows, style)

    add_panel_label(ax_a1, "A", style, x=-0.36, y=1.08)
    add_panel_label(ax_b, "B", style, x=-0.36, y=1.08)
    add_panel_label(ax_c, "C", style, x=-0.18, y=1.08)

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
