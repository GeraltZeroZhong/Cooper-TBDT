"""Build the Cooper-TBDT corpus workflow and composition figure."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from evopoint_da.figures.io import read_csv_rows, read_json, write_csv_rows
from evopoint_da.figures.style import (
    add_panel_label,
    apply_style,
    clean_axis,
    figure_output_dir,
    get_style,
    parse_formats,
    save_figure,
)


TIER_LABELS = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze"}
SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT corpus workflow figure")
    parser.add_argument(
        "--publication-dir",
        default="artifacts/tbdt_v1/publication_report",
        help="Directory produced by build_tbdt_publication_report.",
    )
    parser.add_argument(
        "--mixed-report-json",
        default="artifacts/tbdt_v1/tbdt_mixed_manifest_download_gold_report.json",
        help="Mixed-manifest construction report.",
    )
    parser.add_argument(
        "--gold-manifest",
        default="data/tbdt_gold_training_manifest.csv",
        help="Final Gold supervised training manifest.",
    )
    parser.add_argument(
        "--gold-prepare-report",
        default="artifacts/tbdt_v1/prepare_gold_training_manifest_report.json",
        help="Gold training-manifest preparation report.",
    )
    parser.add_argument(
        "--silver-clean-report",
        default="artifacts/tbdt_v1/build_silver_clean_real_graphs_report.json",
        help="Clean Silver graph build report.",
    )
    parser.add_argument(
        "--download-report",
        default="artifacts/tbdt_v1/download_silver_bronze_assets_report.json",
        help="Silver/Bronze asset download report.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="corpus_workflow", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _fmt_label(value: str) -> str:
    return (
        value.replace("_", " ")
        .replace("tbdt", "TBDT")
        .replace("btub", "BtuB")
        .replace("fepa", "FepA")
        .replace("fhua", "FhuA")
        .replace("feca", "FecA")
        .replace("fyua", "FyuA")
    )


def _load_gold_rows(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Gold manifest is empty: {path}")
    return rows


def _counter(rows: list[dict[str, str]], field: str) -> Counter[str]:
    return Counter(row.get(field, "") or "unknown" for row in rows)


def _load_summary(args: argparse.Namespace) -> dict[str, Any]:
    mixed = read_json(args.mixed_report_json)
    gold_prepare = read_json(args.gold_prepare_report)
    silver_clean = read_json(args.silver_clean_report)
    download = read_json(args.download_report)
    gold_rows = _load_gold_rows(args.gold_manifest)

    bronze_manifest_count = int((mixed.get("counts_by_evidence") or {}).get("bronze", 0))
    afdb_failures = len(download.get("afdb_failures", []))
    summary = {
        "mixed_rows": int(mixed.get("row_count", 0)),
        "mixed_unique_uniprot": int(mixed.get("unique_uniprots", 0)),
        "mixed_unique_pdb": int(mixed.get("unique_pdb_ids", 0)),
        "gold_manifest_rows": int((mixed.get("counts_by_evidence") or {}).get("gold", 0)),
        "silver_manifest_rows": int((mixed.get("counts_by_evidence") or {}).get("silver", 0)),
        "bronze_manifest_rows": bronze_manifest_count,
        "gold_pairs": len(gold_rows),
        "gold_prepared_rows": int(gold_prepare.get("prepared_rows", len(gold_rows))),
        "gold_skipped_uncertain": int(gold_prepare.get("skipped_rows", 0)),
        "silver_clean_graphs": int(silver_clean.get("processed_graphs", 0)),
        "bronze_usable_afdb": bronze_manifest_count - afdb_failures,
        "bronze_afdb_failures": afdb_failures,
        "gold_unique_uniprot": len({row.get("uniprot_id", "") for row in gold_rows if row.get("uniprot_id", "")}),
        "gold_unique_pdb": len({row.get("pdb_id", "") for row in gold_rows if row.get("pdb_id", "")}),
        "split_counts": dict(_counter(gold_rows, "split")),
        "family_counts": dict(_counter(gold_rows, "family")),
        "state_counts": dict(_counter(gold_rows, "state_label")),
        "substrate_counts": dict(_counter(gold_rows, "substrate_class")),
    }
    return summary


def _box(ax: object, xy: tuple[float, float], w: float, h: float, text: str, *, fc: str, ec: str, size: float = 7.2) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size)


def _tier_fill(key: str, style: Any) -> str:
    return {
        "gold": style.palette["glacier"],
        "silver": style.palette["celadon"],
        "bronze": style.palette["sakura"],
    }[key]


def _arrow(ax: object, start: tuple[float, float], end: tuple[float, float], style: Any) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=style.palette["reference"],
        )
    )


def _plot_workflow(ax: object, summary: dict[str, Any], style: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Corpus construction workflow", pad=8)
    text = style.palette["text"]
    _box(
        ax,
        (0.03, 0.58),
        0.24,
        0.24,
        "Candidate\ncollection\nRCSB + UniProt\nAFDB v6",
        fc=style.palette["muted_bg"],
        ec=style.palette["grid"],
        size=7.0,
    )
    _box(
        ax,
        (0.34, 0.58),
        0.25,
        0.24,
        f"Quality filters\n<=3.5 A X-ray/EM\nAFDB match\n{summary['mixed_rows']} rows",
        fc=style.palette["muted_bg"],
        ec=style.palette["grid"],
        size=7.0,
    )
    _box(
        ax,
        (0.68, 0.68),
        0.25,
        0.18,
        f"Gold\n{summary['gold_manifest_rows']} candidates\n{summary['gold_pairs']} pairs",
        fc=_tier_fill("gold", style),
        ec=style.palette["gold"],
        size=6.8,
    )
    _box(
        ax,
        (0.68, 0.44),
        0.25,
        0.18,
        f"Silver\n{summary['silver_manifest_rows']} rows",
        fc=_tier_fill("silver", style),
        ec=style.palette["silver"],
        size=7.0,
    )
    _box(
        ax,
        (0.68, 0.20),
        0.25,
        0.18,
        f"Bronze\n{summary['bronze_manifest_rows']} rows",
        fc=_tier_fill("bronze", style),
        ec=style.palette["bronze"],
        size=7.0,
    )
    _arrow(ax, (0.27, 0.70), (0.34, 0.70), style)
    _arrow(ax, (0.59, 0.70), (0.68, 0.77), style)
    _arrow(ax, (0.59, 0.68), (0.68, 0.53), style)
    _arrow(ax, (0.59, 0.66), (0.68, 0.29), style)
    ax.text(
        0.03,
        0.10,
        f"{summary['mixed_unique_uniprot']} UniProt accessions; {summary['mixed_unique_pdb']} PDB entries in mixed manifest",
        ha="left",
        va="center",
        fontsize=6.8,
        color=text,
    )


def _plot_roles(ax: object, summary: dict[str, Any], style: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Tier roles", pad=8)
    cards = [
        (
            "gold",
            "Gold",
            f"{summary['gold_pairs']} paired AFDB-v6 / experimental structures\nprimary training and held-out evaluation",
        ),
        (
            "silver",
            "Silver",
            f"{summary['silver_clean_graphs']} clean auxiliary graphs\nbeta-barrel geometry pretraining source",
        ),
        (
            "bronze",
            "Bronze",
            f"{summary['bronze_usable_afdb']}/{summary['bronze_manifest_rows']} AFDB-v6 homologs usable\ncoverage for weak scaffold/self-supervision",
        ),
    ]
    y_positions = [0.68, 0.40, 0.12]
    for (key, title, body), y in zip(cards, y_positions):
        _box(
            ax,
            (0.05, y),
            0.90,
            0.20,
            "",
            fc=_tier_fill(key, style),
            ec=style.palette[key],
        )
        ax.text(0.10, y + 0.13, title, ha="left", va="center", fontsize=8.8, fontweight="bold", color=style.palette[key])
        ax.text(0.10, y + 0.065, body, ha="left", va="center", fontsize=7.0, color=style.palette["text"])


def _plot_gold_split(ax: object, summary: dict[str, Any], style: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Gold split overview", pad=8)
    card_specs = [
        ("Pairs", summary["gold_pairs"]),
        ("UniProt groups", summary["gold_unique_uniprot"]),
        ("PDB entries", summary["gold_unique_pdb"]),
    ]
    for i, (label, value) in enumerate(card_specs):
        x = 0.05 + i * 0.30
        _box(ax, (x, 0.60), 0.25, 0.22, "", fc=style.palette["muted_bg"], ec=style.palette["grid"])
        ax.text(x + 0.125, 0.735, str(value), ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + 0.125, 0.645, label, ha="center", va="center", fontsize=7.0)

    split_counts = {split: int(summary["split_counts"].get(split, 0)) for split in SPLIT_ORDER}
    total = sum(split_counts.values())
    left = 0.05
    y = 0.34
    w_total = 0.86
    for split in SPLIT_ORDER:
        width = w_total * split_counts[split] / total
        ax.add_patch(
            FancyBboxPatch(
                (left, y),
                width,
                0.12,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor=style.palette[split],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(left + width / 2, y + 0.06, f"{split} {split_counts[split]}", ha="center", va="center", fontsize=7.0, color="white")
        left += width
    ax.text(0.05, 0.22, "Split unit: UniProt group", ha="left", va="center", fontsize=7.4)
    ax.text(0.05, 0.13, f"train/val/test = {split_counts['train']}/{split_counts['val']}/{split_counts['test']}", ha="left", va="center", fontsize=7.4)


def _segment_colors(keys: list[str], n: int, style: Any) -> list[str]:
    if n <= 0:
        return []
    colors = [style.palette[key] for key in keys]
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def _plot_distribution_stacks(ax: object, summary: dict[str, Any], style: Any) -> None:
    fields = [
        ("family", "Family", summary["family_counts"], ["glacier", "celadon", "aurora", "peacock", "tech", "navy"]),
        ("state", "State", summary["state_counts"], ["glacier", "aurora", "tech", "navy", "instagram"]),
        ("substrate", "Substrate", summary["substrate_counts"], ["glacier", "celadon", "aurora", "peacock", "tech", "sakura", "blossom", "instagram"]),
    ]
    total = int(summary["gold_pairs"])
    ax.set_title("Gold family / state / substrate distributions")
    ax.set_xlim(0, total)
    ax.set_ylim(-0.55, len(fields) - 0.45)
    ax.set_yticks(np.arange(len(fields)))
    ax.set_yticklabels([field[1] for field in fields])
    ax.invert_yaxis()
    ax.set_xlabel("Gold pairs")
    clean_axis(ax, style, grid_axis="x")

    for yi, (_, _, counts, color_keys) in enumerate(fields):
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        colors = _segment_colors(color_keys, len(items), style)
        left = 0
        for (label, count), color in zip(items, colors):
            ax.barh(yi, count, left=left, height=0.34, color=color, edgecolor="white", linewidth=0.7)
            if count >= 25:
                text = f"{_fmt_label(label)} {count}"
                ax.text(left + count / 2, yi, text, ha="center", va="center", fontsize=6.4, color=style.palette["text"])
            elif count >= 8:
                ax.text(left + count / 2, yi, f"{count}", ha="center", va="center", fontsize=6.0, color=style.palette["text"])
            left += count


def _write_plotting_data(out_dir: Path, summary: dict[str, Any]) -> None:
    summary_rows = [
        {"metric": key, "value": value}
        for key, value in summary.items()
        if not isinstance(value, dict)
    ]
    write_csv_rows(out_dir / "corpus_workflow_summary_values.csv", summary_rows, ["metric", "value"])

    dist_rows: list[dict[str, Any]] = []
    for field, counts in (
        ("family", summary["family_counts"]),
        ("state_label", summary["state_counts"]),
        ("substrate_class", summary["substrate_counts"]),
        ("split", summary["split_counts"]),
    ):
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            dist_rows.append({"field": field, "value": value, "count": count})
    write_csv_rows(out_dir / "corpus_workflow_gold_distribution_values.csv", dist_rows, ["field", "value", "count"])


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)
    summary = _load_summary(args)

    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_plotting_data(out_dir, summary)

    fig = plt.figure(figsize=(8.4, 6.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_workflow(ax_a, summary, style)
    _plot_roles(ax_b, summary, style)
    _plot_gold_split(ax_c, summary, style)
    _plot_distribution_stacks(ax_d, summary, style)

    for label, ax in zip(("A", "B", "C"), (ax_a, ax_b, ax_c)):
        add_panel_label(ax, label, style)
    add_panel_label(ax_d, "D", style)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.08, wspace=0.22, hspace=0.36)

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
