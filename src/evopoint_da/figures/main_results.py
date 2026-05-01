"""Build the main Cooper-TBDT result figure.

The figure is designed around the article-facing primary comparison:
raw AFDB/zero displacement versus the single scaffold-prior Cooper-TBDT
5-seed family, with the validation-calibrated blend retained as a secondary
aggregate reference in panel A. It also includes plug ROC/PR score-only
localization panels, which should be interpreted separately from coordinate
RMSD.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from evopoint_da.figures.io import as_float, finite_float, iter_sample_region_values, read_csv_rows, read_json, write_csv_rows
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


SUMMARY_REGIONS = ("eval", "plug", "tonb_box")
REGION_LABELS = {
    "eval": "Evaluation\nregion",
    "plug": "Plug",
    "tonb_box": "TonB box",
    "barrel_core": "Barrel core",
}
METHOD_LABELS = {
    "raw": "Raw AFDB",
    "primary": "Cooper-TBDT\nsingle 5-seed",
    "blend": "Validation-\ncalibrated blend",
}
LEGEND_LABELS = {
    "raw": "Raw AFDB",
    "primary": "Cooper-TBDT single 5-seed",
    "blend": "Validation-calibrated blend",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT main result figure")
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
        "--seed-metrics-dir",
        default="artifacts/tbdt_v1/seed_stability_best_selection/metrics",
        help="Directory containing per-seed best-selection metric JSON files.",
    )
    parser.add_argument(
        "--blend-metrics-json",
        default="artifacts/tbdt_v1/report_models/metrics/validation_calibrated_region_blend_test.json",
        help="Validation-calibrated blend metric JSON.",
    )
    parser.add_argument(
        "--curve-summary-csv",
        default="artifacts/tbdt_v1/external_baseline_curves/classification_curve_summary.csv",
        help="Residue-shift localization summary CSV for main plug ROC/PR panels.",
    )
    parser.add_argument(
        "--curve-points-csv",
        default="artifacts/tbdt_v1/external_baseline_curves/classification_curve_points.csv",
        help="Residue-shift localization curve-point CSV for main plug ROC/PR panels.",
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
        help="Output directory for figure files and exported plotting data.",
    )
    parser.add_argument("--out-name", default="main_result_primary_vs_raw_afdb", help="Output filename stem.")
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


def _load_panel_a_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    publication_dir = Path(args.publication_dir)
    coord_path = publication_dir / "coordinate_metrics_summary.csv"
    coord_rows = read_csv_rows(coord_path)
    seed_rows = read_csv_rows(args.seed_aggregate_csv)

    rows: list[dict[str, Any]] = []
    for region in SUMMARY_REGIONS:
        raw = _first_row(coord_rows, source=str(coord_path), method="raw_af2_zero", region=region)
        rows.append(
            {
                "method": "raw",
                "method_label": METHOD_LABELS["raw"],
                "region": region,
                "region_label": REGION_LABELS[region],
                "rmsd": finite_float(raw.get("prediction_error_rms"), field="prediction_error_rms", source=str(coord_path)),
                "rmsd_sd": 0.0,
            }
        )

        primary = _first_row(
            seed_rows,
            source=str(args.seed_aggregate_csv),
            split="test",
            region=region,
            metric="prediction_error_rms",
        )
        rows.append(
            {
                "method": "primary",
                "method_label": METHOD_LABELS["primary"],
                "region": region,
                "region_label": REGION_LABELS[region],
                "rmsd": finite_float(primary.get("mean"), field="mean", source=str(args.seed_aggregate_csv)),
                "rmsd_sd": as_float(primary.get("std"), 0.0),
            }
        )

        blend = _first_row(coord_rows, source=str(coord_path), method="cooper_tbdt_scaffold_blend", region=region)
        rows.append(
            {
                "method": "blend",
                "method_label": METHOD_LABELS["blend"],
                "region": region,
                "region_label": REGION_LABELS[region],
                "rmsd": finite_float(blend.get("prediction_error_rms"), field="prediction_error_rms", source=str(coord_path)),
                "rmsd_sd": 0.0,
            }
        )
    return rows


def _load_delta_rows(args: argparse.Namespace, region: str) -> list[dict[str, Any]]:
    path = Path(args.publication_dir) / "primary_model_paired_delta_samples.csv"
    rows = read_csv_rows(path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "cooper_tbdt_scaffold_single_5seed_mean" or row.get("region") != region:
            continue
        selected.append(
            {
                "sample_id": row["sample_id"],
                "region": region,
                "raw_af2_rmsd": finite_float(row.get("raw_af2_rmsd"), field="raw_af2_rmsd", source=str(path)),
                "method_rmsd": finite_float(row.get("method_rmsd"), field="method_rmsd", source=str(path)),
                "delta_rmsd_method_minus_raw": finite_float(
                    row.get("delta_rmsd_method_minus_raw"),
                    field="delta_rmsd_method_minus_raw",
                    source=str(path),
                ),
            }
        )
    if not selected:
        raise ValueError(f"No primary paired-delta sample rows found for region {region!r} in {path}")
    return selected


def _load_delta_summary(args: argparse.Namespace, region: str) -> dict[str, Any]:
    path = Path(args.publication_dir) / "primary_model_paired_delta_summary.csv"
    row = _first_row(
        read_csv_rows(path),
        source=str(path),
        method="cooper_tbdt_scaffold_single_5seed_mean",
        aggregation="per_target_seed_mean",
        region=region,
    )
    return {
        "n_targets": int(finite_float(row.get("n_targets"), field="n_targets", source=str(path))),
        "n_improved": int(finite_float(row.get("n_improved"), field="n_improved", source=str(path))),
        "n_worsened": int(finite_float(row.get("n_worsened"), field="n_worsened", source=str(path))),
        "median": finite_float(row.get("median_delta_rmsd_method_minus_raw"), field="median_delta", source=str(path)),
        "median_ci_low": finite_float(row.get("median_delta_ci95_low"), field="median_delta_ci95_low", source=str(path)),
        "median_ci_high": finite_float(row.get("median_delta_ci95_high"), field="median_delta_ci95_high", source=str(path)),
        "wilcoxon_p": as_float(row.get("wilcoxon_p_less_method_lt_raw")),
    }


def _short_target_labels(sample_ids: list[str]) -> list[str]:
    labels: list[str] = []
    for sample_id in sample_ids:
        parts = sample_id.split("_")
        labels.append(parts[-2].upper() if len(parts) >= 3 else sample_id)
    counts: dict[str, int] = defaultdict(int)
    unique: list[str] = []
    for label in labels:
        counts[label] += 1
        unique.append(label if counts[label] == 1 else f"{label}.{counts[label]}")
    return unique


def _load_barrel_core_values(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed_metrics_dir = Path(args.seed_metrics_dir)
    seed_paths = sorted(seed_metrics_dir.glob("seed_*_best-selection_test.json"))
    if not seed_paths:
        raise FileNotFoundError(f"No seed metric JSON files found in {seed_metrics_dir}")

    primary_by_sample: dict[str, list[float]] = defaultdict(list)
    for path in seed_paths:
        payload = read_json(path)
        for row in iter_sample_region_values(
            payload,
            region="barrel_core",
            metric="predicted_displacement_mean",
            source=str(path),
        ):
            primary_by_sample[row["sample_id"]].append(float(row["value"]))

    primary_rows: list[dict[str, Any]] = []
    for sample_id, values in sorted(primary_by_sample.items()):
        if len(values) != len(seed_paths):
            raise ValueError(f"Sample {sample_id!r} has {len(values)} seed values; expected {len(seed_paths)}")
        primary_rows.append(
            {
                "method": "primary",
                "method_label": METHOD_LABELS["primary"],
                "sample_id": sample_id,
                "barrel_core_predicted_displacement_mean": float(np.mean(values)),
                "n_seeds": len(values),
            }
        )

    blend_payload = read_json(args.blend_metrics_json)
    blend_by_sample = {
        row["sample_id"]: row["value"]
        for row in iter_sample_region_values(
            blend_payload,
            region="barrel_core",
            metric="predicted_displacement_mean",
            source=str(args.blend_metrics_json),
        )
    }

    rows: list[dict[str, Any]] = []
    for row in primary_rows:
        sample_id = row["sample_id"]
        rows.append(
            {
                "method": "raw",
                "method_label": METHOD_LABELS["raw"],
                "sample_id": sample_id,
                "barrel_core_predicted_displacement_mean": 0.0,
                "n_seeds": 0,
            }
        )
        rows.append(row)
        if sample_id in blend_by_sample:
            rows.append(
                {
                    "method": "blend",
                    "method_label": METHOD_LABELS["blend"],
                    "sample_id": sample_id,
                    "barrel_core_predicted_displacement_mean": float(blend_by_sample[sample_id]),
                    "n_seeds": 1,
                }
            )
    return rows


def _plot_panel_a(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    palette = style.palette
    methods = ("raw", "primary", "blend")
    x = np.arange(len(SUMMARY_REGIONS), dtype=float)
    width = 0.23
    offsets = {"raw": -width, "primary": 0.0, "blend": width}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        values = [float(row["rmsd"]) for row in method_rows]
        errors = [float(row["rmsd_sd"]) if method == "primary" else 0.0 for row in method_rows]
        bars = ax.bar(
            x + offsets[method],
            values,
            width=width,
            yerr=errors,
            capsize=2.0 if method == "primary" else 0.0,
            color=palette[method],
            edgecolor="white",
            linewidth=0.7,
            label=LEGEND_LABELS[method],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.08,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=palette["text"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS[region] for region in SUMMARY_REGIONS])
    ax.set_ylabel("Region RMSD (Å)")
    ax.set_title("Aggregate held-out Gold RMSD")
    ax.set_ylim(0.0, max(float(row["rmsd"]) for row in rows) + 0.75)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncols=1,
        handlelength=1.1,
        labelspacing=0.25,
        borderpad=0.25,
        fontsize=6.6,
    )
    clean_axis(ax, style)

    # TonB has a much larger RMSD scale; the inset keeps eval/plug differences visible.
    inset = inset_axes(ax, width="43%", height="38%", loc="upper center", borderpad=1.0)
    zoom_regions = ("eval", "plug")
    zoom_x = np.arange(len(zoom_regions), dtype=float)
    zoom_rows = {
        (str(row["region"]), str(row["method"])): row
        for row in rows
        if str(row["region"]) in zoom_regions
    }
    for method in methods:
        values = [float(zoom_rows[(region, method)]["rmsd"]) for region in zoom_regions]
        errors = [float(zoom_rows[(region, method)]["rmsd_sd"]) if method == "primary" else 0.0 for region in zoom_regions]
        inset.bar(
            zoom_x + offsets[method],
            values,
            width=width,
            yerr=errors,
            capsize=1.5 if method == "primary" else 0.0,
            color=palette[method],
            edgecolor="white",
            linewidth=0.5,
        )
    inset.set_title("Eval/plug zoom", fontsize=6.4, pad=2.0)
    inset.set_xticks(zoom_x)
    inset.set_xticklabels(["Eval", "Plug"], fontsize=5.8)
    inset.tick_params(axis="y", labelsize=5.8, length=2.0)
    inset.set_ylim(1.18, 1.84)
    clean_axis(inset, style)


def _plot_delta_panel(ax: object, rows: list[dict[str, Any]], summary: dict[str, Any], region: str, style: Any) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["delta_rmsd_method_minus_raw"])
    sample_ids = [str(row["sample_id"]) for row in sorted_rows]
    labels = _short_target_labels(sample_ids)
    deltas = np.array([float(row["delta_rmsd_method_minus_raw"]) for row in sorted_rows], dtype=float)
    x = np.arange(len(deltas), dtype=float)
    colors = [style.palette["improved"] if value < 0 else style.palette["worsened"] for value in deltas]

    ax.axhline(0.0, color=style.palette["reference"], linewidth=1.0)
    ax.axhline(float(summary["median"]), color=style.palette["primary"], linewidth=1.25)
    ax.vlines(x, 0.0, deltas, color=colors, alpha=0.62, linewidth=0.9)
    ax.scatter(x, deltas, s=style.point_size, c=colors, edgecolor="white", linewidth=0.45, zorder=3)

    max_abs = float(np.max(np.abs(deltas)))
    pad = max(max_abs * 0.18, 0.012)
    ax.set_ylim(-max_abs - pad, max_abs + pad)
    ax.set_xlim(-0.7, len(deltas) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Delta RMSD (Å)")
    ax.set_title(f"{REGION_LABELS[region].replace(chr(10), ' ')} paired target delta")
    clean_axis(ax, style)

    p_value = summary.get("wilcoxon_p")
    p_text = f", p={p_value:.3g}" if math.isfinite(float(p_value)) else ""
    text = (
        f"{summary['n_improved']}/{summary['n_targets']} improved\n"
        f"median {summary['median']:+.3f} Å{p_text}"
    )
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )


def _plot_barrel_core_panel(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    methods = ("raw", "primary", "blend")
    values_by_method = {
        method: [
            float(row["barrel_core_predicted_displacement_mean"])
            for row in rows
            if row["method"] == method
        ]
        for method in methods
    }
    positions = np.arange(1, len(methods) + 1, dtype=float)
    data = [values_by_method[method] for method in methods]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"color": style.palette["reference"], "linewidth": 0.9},
        capprops={"color": style.palette["reference"], "linewidth": 0.9},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(style.palette[method])
        patch.set_alpha(0.82)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(7)
    for pos, method in zip(positions, methods):
        values = np.array(values_by_method[method], dtype=float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=13,
            color=style.palette[method],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
            zorder=3,
        )
    ax.axhline(0.05, color=style.palette["reference"], linestyle="--", linewidth=0.95)
    ax.text(
        len(methods) + 0.28,
        0.05,
        "0.05 Å",
        va="center",
        ha="left",
        fontsize=6.8,
        color=style.palette["reference"],
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[method] for method in methods])
    ax.set_ylabel("Predicted displacement mean (Å)")
    ax.set_title("Barrel-core displacement remains small")
    ax.set_xlim(0.45, len(methods) + 0.65)
    ymax = max(max(values) if values else 0.0 for values in data)
    ax.set_ylim(-0.005, max(0.075, ymax * 1.22))
    clean_axis(ax, style)


def _plot_plug_localization_curve(
    ax: object,
    curve_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    curve: str,
    style: Any,
) -> None:
    summary = {str(row["method"]): row for row in summary_rows}
    for method in LOCALIZATION_METHOD_ORDER:
        selected = [row for row in curve_rows if row["curve"] == curve and row["method"] == method]
        if not selected:
            continue
        metric = float(summary[method]["auroc"] if curve == "roc" else summary[method]["average_precision"])
        label = f"{localization_method_label(method)} ({metric:.2f})"
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
        ax.set_title("Plug residue-shift ROC")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            handlelength=1.9,
            labelspacing=0.18,
            borderpad=0.35,
            fontsize=5.7,
        )
        add_note_box(
            ax,
            "Score-only localization\npositive: target displacement >=1 Å",
            style,
            x=0.03,
            y=0.97,
            size=6.0,
        )
    else:
        positive_rate = float(summary[BLEND_METHOD_ID]["positive_rate"])
        ax.axhline(positive_rate, color=style.palette["grid"], linestyle="--", linewidth=0.9)
        ax.text(0.98, positive_rate + 0.02, "prevalence", ha="right", va="bottom", fontsize=6.5, color=style.palette["neutral"])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Plug residue-shift PR")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            handlelength=1.9,
            labelspacing=0.18,
            borderpad=0.35,
            fontsize=5.7,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    clean_axis(ax, style, grid_axis="both")


def _write_plotting_data(
    out_dir: Path,
    panel_a_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    barrel_rows: list[dict[str, Any]],
    plug_localization_summary: list[dict[str, Any]],
    plug_localization_curves: list[dict[str, Any]],
    plug_localization_seed_metrics: list[dict[str, Any]],
) -> None:
    write_csv_rows(
        out_dir / "main_result_panel_a_values.csv",
        panel_a_rows,
        ["method", "method_label", "region", "region_label", "rmsd", "rmsd_sd"],
    )
    write_csv_rows(
        out_dir / "main_result_primary_paired_delta_values.csv",
        delta_rows,
        ["sample_id", "region", "raw_af2_rmsd", "method_rmsd", "delta_rmsd_method_minus_raw"],
    )
    write_csv_rows(
        out_dir / "main_result_barrel_core_values.csv",
        barrel_rows,
        ["method", "method_label", "sample_id", "barrel_core_predicted_displacement_mean", "n_seeds"],
    )
    write_csv_rows(
        out_dir / "main_result_plug_localization_metric_values.csv",
        plug_localization_summary,
        [
            "region",
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
        out_dir / "main_result_plug_localization_seed_metric_values.csv",
        plug_localization_seed_metrics,
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
        out_dir / "main_result_plug_localization_curve_points.csv",
        plug_localization_curves,
        ["region", "method", "method_label", "method_group", "curve", "x", "y", "y_low", "y_high", "y_sd", "source"],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    panel_a_rows = _load_panel_a_rows(args)
    eval_delta_rows = _load_delta_rows(args, "eval")
    plug_delta_rows = _load_delta_rows(args, "plug")
    eval_delta_summary = _load_delta_summary(args, "eval")
    plug_delta_summary = _load_delta_summary(args, "plug")
    barrel_rows = _load_barrel_core_values(args)
    _, _, cooper_seed_metrics = collect_cooper_seed_localization(
        sample_list=args.sample_list,
        prediction_root=args.seed_prediction_root,
        seeds=parse_seed_list(args.seeds),
        regions=("plug",),
        positive_threshold=float(args.positive_threshold),
    )
    plug_localization_summary = load_external_localization_summary(args.curve_summary_csv, ("plug",))
    plug_localization_curves = load_external_localization_curves(args.curve_points_csv, ("plug",))

    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_plotting_data(
        out_dir,
        panel_a_rows,
        eval_delta_rows + plug_delta_rows,
        barrel_rows,
        plug_localization_summary,
        plug_localization_curves,
        cooper_seed_metrics,
    )

    fig = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.08], width_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[0, 2])
    ax_f = fig.add_subplot(grid[1, 2])

    _plot_panel_a(ax_a, panel_a_rows, style)
    _plot_delta_panel(ax_b, eval_delta_rows, eval_delta_summary, "eval", style)
    _plot_delta_panel(ax_c, plug_delta_rows, plug_delta_summary, "plug", style)
    _plot_barrel_core_panel(ax_d, barrel_rows, style)
    _plot_plug_localization_curve(
        ax_e,
        plug_localization_curves,
        plug_localization_summary,
        curve="roc",
        style=style,
    )
    _plot_plug_localization_curve(
        ax_f,
        plug_localization_curves,
        plug_localization_summary,
        curve="pr",
        style=style,
    )

    for label, ax in zip(("A", "B", "C", "D", "E", "F"), (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
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
