"""Build the Gold held-out test displacement landscape figure."""

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
    add_panel_label,
    apply_style,
    clean_axis,
    figure_output_dir,
    get_style,
    parse_formats,
    save_figure,
)


REGIONS = ("eval", "plug", "tonb_box")
RMSD_REGIONS = ("eval", "plug", "tonb_box", "barrel_core")
REGION_LABELS = {
    "eval": "Evaluation\nregion",
    "plug": "Plug",
    "tonb_box": "TonB box",
    "barrel_core": "Barrel\ncore",
}
BIN_ORDER = ("lt_0p5", "0p5_to_1", "1_to_2", "2_to_5", "ge_5")
BIN_LABELS = {
    "lt_0p5": "0-0.5",
    "0p5_to_1": "0.5-1",
    "1_to_2": "1-2",
    "2_to_5": "2-5",
    "ge_5": ">=5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gold test raw-baseline and displacement-bin figure")
    parser.add_argument(
        "--publication-dir",
        default="artifacts/tbdt_v1/publication_report",
        help="Directory produced by build_tbdt_publication_report.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="gold_test_displacement_landscape", help="Output filename stem.")
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


def _load_raw_rmsd_rows(publication_dir: Path) -> list[dict[str, Any]]:
    path = publication_dir / "coordinate_metrics_summary.csv"
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for region in RMSD_REGIONS:
        row = _first_row(rows, source=str(path), method="raw_af2_zero", region=region)
        out.append(
            {
                "region": region,
                "region_label": REGION_LABELS[region].replace("\n", " "),
                "raw_afdb_rmsd": finite_float(
                    row.get("prediction_error_rms"),
                    field="prediction_error_rms",
                    source=str(path),
                ),
                "n_residues": int(finite_float(row.get("n_residues"), field="n_residues", source=str(path))),
            }
        )
    return out


def _load_bin_rows(publication_dir: Path) -> list[dict[str, Any]]:
    path = publication_dir / "displacement_bin_summary.csv"
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for region in REGIONS:
        for bin_name in BIN_ORDER:
            row = _first_row(
                rows,
                source=str(path),
                scope="gold_graphs",
                split="test",
                region=region,
                bin=bin_name,
            )
            out.append(
                {
                    "region": region,
                    "region_label": REGION_LABELS[region].replace("\n", " "),
                    "bin": bin_name,
                    "bin_label": BIN_LABELS[bin_name],
                    "fraction": finite_float(row.get("fraction"), field="fraction", source=str(path)),
                    "n_residues": int(finite_float(row.get("n_residues"), field="n_residues", source=str(path))),
                    "region_residue_total": int(
                        finite_float(row.get("region_residue_total"), field="region_residue_total", source=str(path))
                    ),
                    "n_samples_with_region": int(
                        finite_float(row.get("n_samples_with_region"), field="n_samples_with_region", source=str(path))
                    ),
                }
            )
    return out


def _plot_raw_rmsd(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    x = np.arange(len(rows), dtype=float)
    values = np.array([float(row["raw_afdb_rmsd"]) for row in rows], dtype=float)
    region_colors = {
        "eval": style.palette["aurora"],
        "plug": style.palette["tech"],
        "tonb_box": style.palette["instagram"],
        "barrel_core": style.palette["celadon"],
    }
    colors = [region_colors[str(row["region"])] for row in rows]
    bars = ax.bar(x, values, width=0.66, color=colors, edgecolor="white", linewidth=0.8)
    for bar, value, row in zip(bars, values, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.11,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.08,
            f"n={row['n_residues']}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=style.palette["text"] if str(row["region"]) == "barrel_core" else "white",
            rotation=90,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS[str(row["region"])] for row in rows])
    ax.set_ylabel("Raw AFDB RMSD (Å)")
    ax.set_title("Held-out Gold raw AFDB baseline RMSD")
    ax.set_ylim(0.0, max(values) + 0.9)
    clean_axis(ax, style)


def _plot_region_bins(ax: object, rows: list[dict[str, Any]], region: str, style: Any) -> None:
    selected = [row for row in rows if row["region"] == region]
    by_bin = {row["bin"]: row for row in selected}
    fractions = np.array([float(by_bin[bin_name]["fraction"]) for bin_name in BIN_ORDER], dtype=float)
    total = int(selected[0]["region_residue_total"]) if selected else 0
    n_samples = int(selected[0]["n_samples_with_region"]) if selected else 0
    x = np.arange(len(BIN_ORDER), dtype=float)
    colors = [style.palette[f"bin_{bin_name}"] for bin_name in BIN_ORDER]

    bars = ax.bar(x, fractions * 100.0, width=0.68, color=colors, edgecolor="white", linewidth=0.8)
    for bar, fraction in zip(bars, fractions):
        if fraction <= 0.08:
            label_y = max(fraction * 100.0 + 1.8, 1.8)
            va = "bottom"
            color = style.palette["text"]
        else:
            label_y = fraction * 100.0 - 2.0
            va = "top"
            color = "white" if fraction > 0.09 else style.palette["text"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{fraction * 100:.0f}%",
            ha="center",
            va=va,
            fontsize=6.7,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABELS[bin_name] for bin_name in BIN_ORDER])
    ax.set_ylabel("Residues (%)")
    ax.set_xlabel("Target displacement magnitude (Å)")
    ax.set_title(f"{REGION_LABELS[region].replace(chr(10), ' ')} displacement distribution")
    ax.set_ylim(0.0, 86.0)
    ax.text(
        0.02,
        0.96,
        f"{total} residues, {n_samples} targets",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.2},
    )
    clean_axis(ax, style)


def _write_plotting_data(out_dir: Path, rmsd_rows: list[dict[str, Any]], bin_rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        out_dir / "gold_test_raw_afdb_rmsd_values.csv",
        rmsd_rows,
        ["region", "region_label", "raw_afdb_rmsd", "n_residues"],
    )
    write_csv_rows(
        out_dir / "gold_test_displacement_bin_values.csv",
        bin_rows,
        [
            "region",
            "region_label",
            "bin",
            "bin_label",
            "fraction",
            "n_residues",
            "region_residue_total",
            "n_samples_with_region",
        ],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    publication_dir = Path(args.publication_dir)
    rmsd_rows = _load_raw_rmsd_rows(publication_dir)
    bin_rows = _load_bin_rows(publication_dir)

    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_plotting_data(out_dir, rmsd_rows, bin_rows)

    fig = plt.figure(figsize=(7.25, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_raw_rmsd(ax_a, rmsd_rows, style)
    _plot_region_bins(ax_b, bin_rows, "eval", style)
    _plot_region_bins(ax_c, bin_rows, "plug", style)
    _plot_region_bins(ax_d, bin_rows, "tonb_box", style)

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
