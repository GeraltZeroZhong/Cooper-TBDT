"""Build the residue-shift ROC/PR localization figure for Cooper-TBDT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evopoint_da.figures.io import write_csv_rows
from evopoint_da.figures.localization import (
    BLEND_METHOD_ID,
    COOPER_METHOD_ID,
    LOCALIZATION_METHOD_ORDER,
    collect_cooper_seed_localization,
    load_external_localization_curves,
    load_external_localization_summary,
    localization_method_color,
    localization_method_label,
    localization_method_linewidth,
    localization_method_linestyle,
    parse_seed_list,
)
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


REGIONS = ("eval", "plug")
REGION_LABELS = {
    "eval": "Evaluation region",
    "plug": "Plug",
    "tonb_box": "TonB box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT residue-shift localization figure")
    parser.add_argument(
        "--curve-summary-csv",
        default="artifacts/tbdt_v1/external_baseline_curves/classification_curve_summary.csv",
        help="Classification summary CSV produced by eval_tbdt_classification_curves.",
    )
    parser.add_argument(
        "--curve-points-csv",
        default="artifacts/tbdt_v1/external_baseline_curves/classification_curve_points.csv",
        help="Classification ROC/PR curve-points CSV.",
    )
    parser.add_argument(
        "--sample-list",
        default="artifacts/tbdt_v1/test_graph_files.txt",
        help="Held-out graph file list used to export Cooper-TBDT seed-level localization audit data.",
    )
    parser.add_argument(
        "--seed-prediction-root",
        default="artifacts/tbdt_v1/seed_stability_best_selection/predictions",
        help="Directory containing seed_*_best-selection_test prediction folders.",
    )
    parser.add_argument("--seeds", default="42,101,202,303,404", help="Comma-separated primary training seeds.")
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=1.0,
        help="Residue-shift positive threshold in Angstrom for localization panels.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="residue_shift_localization", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _summary_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["region"]), str(row["method"])): row for row in rows}


def _plot_curve(
    ax: object,
    curve_rows: list[dict[str, Any]],
    summary: dict[tuple[str, str], dict[str, Any]],
    *,
    region: str,
    curve: str,
    style: Any,
    show_legend: bool = False,
) -> None:
    for method in LOCALIZATION_METHOD_ORDER:
        selected = [row for row in curve_rows if row["region"] == region and row["curve"] == curve and row["method"] == method]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: (float(row["x"]), float(row["y"])))
        metric = summary[(region, method)]["auroc"] if curve == "roc" else summary[(region, method)]["average_precision"]
        label = f"{localization_method_label(method)} ({float(metric):.2f})"
        x_values = np.array([float(row["x"]) for row in selected], dtype=float)
        y_values = np.array([float(row["y"]) for row in selected], dtype=float)
        ax.plot(
            x_values,
            y_values,
            color=localization_method_color(method, style),
            linestyle=localization_method_linestyle(method),
            linewidth=localization_method_linewidth(method),
            alpha=0.96 if method in {BLEND_METHOD_ID, COOPER_METHOD_ID, "af2_low_plddt", "protcross_pocket_score"} else 0.82,
            label=label,
            zorder=3 if method == COOPER_METHOD_ID else 2,
        )

    if curve == "roc":
        ax.plot([0, 1], [0, 1], color=style.palette["grid"], linestyle="--", linewidth=0.9)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{REGION_LABELS[region]} ROC")
    else:
        positive_rate = float(summary[(region, BLEND_METHOD_ID)]["positive_rate"])
        ax.axhline(positive_rate, color=style.palette["grid"], linestyle="--", linewidth=0.9)
        ax.text(0.98, positive_rate + 0.018, "prevalence", ha="right", va="bottom", fontsize=6.5, color=style.palette["neutral"])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{REGION_LABELS[region]} PR")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    clean_axis(ax, style, grid_axis="both")
    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            handlelength=1.9,
            labelspacing=0.18,
            borderpad=0.35,
            fontsize=5.7,
        )


def _plot_metric_heatmap(ax: object, summary_rows: list[dict[str, Any]], style: Any) -> None:
    summary = _summary_lookup(summary_rows)
    columns = [("eval", "AUROC"), ("eval", "AP"), ("plug", "AUROC"), ("plug", "AP")]
    methods = list(LOCALIZATION_METHOD_ORDER)
    matrix = np.zeros((len(methods), len(columns)), dtype=float)
    for i, method in enumerate(methods):
        for j, (region, metric) in enumerate(columns):
            row = summary[(region, method)]
            matrix[i, j] = float(row["auroc"] if metric == "AUROC" else row["average_precision"])

    cmap = LinearSegmentedColormap.from_list(
        "cooper_blue_cyan",
        [
            style.palette["glacier"],
            style.palette["celadon"],
            style.palette["aurora"],
            style.palette["tech"],
            style.palette["navy"],
        ],
    )
    image = ax.imshow(matrix, vmin=0.2, vmax=0.85, cmap=cmap, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.63 else style.palette["text"]
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color=color)
    ax.set_xticks(np.arange(len(columns)))
    short_region_labels = {"eval": "Eval", "plug": "Plug"}
    ax.set_xticklabels([f"{short_region_labels[region]}\n{metric}" for region, metric in columns])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([localization_method_label(method) for method in methods])
    ax.set_title("Localization metrics")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.figure.colorbar(image, ax=ax, shrink=0.82, fraction=0.06, pad=0.02, label="Score")


def _plot_tonb_caveat(ax: object, summary_rows: list[dict[str, Any]], style: Any) -> None:
    summary = _summary_lookup(summary_rows)
    methods = list(LOCALIZATION_METHOD_ORDER)
    y = np.arange(len(methods), dtype=float)
    auroc = np.array([float(summary[("tonb_box", method)]["auroc"]) for method in methods])
    ap = np.array([float(summary[("tonb_box", method)]["average_precision"]) for method in methods])
    positive_rate = float(summary[("tonb_box", BLEND_METHOD_ID)]["positive_rate"])
    n_pos = int(summary[("tonb_box", BLEND_METHOD_ID)]["n_positive"])
    n_neg = int(summary[("tonb_box", BLEND_METHOD_ID)]["n_negative"])
    n_total = int(summary[("tonb_box", BLEND_METHOD_ID)]["n_residues"])

    colors = [localization_method_color(method, style) for method in methods]
    ax.scatter(auroc, y - 0.12, color=colors, s=34, marker="o", label="AUROC", zorder=3)
    ax.scatter(ap, y + 0.12, facecolor="white", edgecolor=colors, linewidth=1.15, s=34, marker="s", label="AP", zorder=3)
    ax.axvline(positive_rate, color=style.palette["grid"], linestyle="--", linewidth=0.9)
    ax.text(
        positive_rate - 0.01,
        -0.62,
        f"PR baseline {positive_rate:.3f}",
        ha="right",
        va="center",
        fontsize=6.4,
        color=style.palette["neutral"],
    )
    ax.set_yticks(y)
    ax.set_yticklabels([localization_method_label(method) for method in methods])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.03)
    ax.set_xlabel("TonB score")
    ax.set_title("TonB ranking caveat")
    add_note_box(
        ax,
        "filled circle = AUROC; open square = AP\n"
        f"TonB residues: {n_pos}/{n_total} positives, {n_neg} negative\n"
        "AP is inflated by class imbalance;\nAUROC is also unstable with one negative.",
        style,
        x=0.03,
        y=0.08,
        va="bottom",
    )
    clean_axis(ax, style, grid_axis="x")


def _write_outputs(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    seed_metric_rows: list[dict[str, Any]],
) -> None:
    write_csv_rows(
        out_dir / "residue_shift_localization_metric_values.csv",
        summary_rows,
        [
            "region",
            "region_label",
            "method",
            "method_label",
            "method_group",
            "n_residues",
            "n_positive",
            "n_negative",
            "positive_rate",
            "auroc",
            "auroc_sd",
            "average_precision",
            "average_precision_sd",
            "n_seeds",
            "source",
        ],
    )
    write_csv_rows(
        out_dir / "residue_shift_localization_seed_metric_values.csv",
        seed_metric_rows,
        [
            "region",
            "method",
            "method_label",
            "method_group",
            "seed",
            "n_residues",
            "n_positive",
            "n_negative",
            "positive_rate",
            "auroc",
            "average_precision",
            "source",
        ],
    )
    write_csv_rows(
        out_dir / "residue_shift_localization_curve_points.csv",
        curve_rows,
        ["region", "method", "method_label", "method_group", "curve", "x", "y", "y_low", "y_high", "y_sd", "source"],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)
    out_dir = figure_output_dir(args.out_dir, args.out_name)

    all_regions = ("eval", "plug", "tonb_box")
    _, _, seed_metric_rows = collect_cooper_seed_localization(
        sample_list=args.sample_list,
        prediction_root=args.seed_prediction_root,
        seeds=parse_seed_list(args.seeds),
        regions=all_regions,
        positive_threshold=float(args.positive_threshold),
    )
    summary_rows = load_external_localization_summary(args.curve_summary_csv, all_regions)
    curve_rows = load_external_localization_curves(args.curve_points_csv, REGIONS)
    for row in summary_rows:
        row.setdefault("region_label", REGION_LABELS.get(str(row.get("region", "")), str(row.get("region", ""))))
    _write_outputs(out_dir, summary_rows, curve_rows, seed_metric_rows)
    summary = _summary_lookup(summary_rows)

    fig = plt.figure(figsize=(10.3, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.13])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    _plot_curve(ax_a, curve_rows, summary, region="eval", curve="roc", style=style, show_legend=True)
    _plot_curve(ax_b, curve_rows, summary, region="eval", curve="pr", style=style)
    _plot_metric_heatmap(ax_c, summary_rows, style)
    _plot_curve(ax_d, curve_rows, summary, region="plug", curve="roc", style=style)
    _plot_curve(ax_e, curve_rows, summary, region="plug", curve="pr", style=style)
    _plot_tonb_caveat(ax_f, summary_rows, style)

    for label, ax in zip(("A", "B", "C", "D", "E", "F"), (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
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
