"""Build source-generated standalone panel figures for review.

This module intentionally does not crop combined figures. Each output panel is
rendered from the same source artifacts and plotting functions used by the
combined figure builders, but into its own matplotlib figure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evopoint_da.figures import (
    baseline_comparison,
    corpus_workflow,
    critical_ablation,
    gold_test_displacement_landscape,
    main_results,
    residue_shift_localization,
    seed_stability_selector,
    task_definition,
    tonb_mechanistic_boundary,
)
from evopoint_da.figures.io import write_csv_rows
from evopoint_da.figures.localization import load_external_localization_curves, load_external_localization_summary
from evopoint_da.figures.style import (
    add_panel_label,
    apply_style,
    figure_output_dir,
    get_style,
    parse_formats,
    save_figure,
)


PanelPlotter = Callable[[Any, Any], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone source-generated panel figures")
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figure_panel_review",
        help="Single review directory where all standalone panel files are written.",
    )
    parser.add_argument(
        "--figure-out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Canonical figure directory used for supporting data exports.",
    )
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg,pdf", help="Comma-separated panel output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument(
        "--only",
        default="",
        help="Optional comma-separated subset by figure id, e.g. figure_1,figure_s1.",
    )
    return parser.parse_args()


def _default_args(module: Any, **overrides: Any) -> argparse.Namespace:
    old_argv = sys.argv[:]
    sys.argv = [module.__name__]
    try:
        args = module.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _selected(figure_id: str, only: set[str]) -> bool:
    return not only or figure_id in only


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    return f"{float(value) * 100.0:.{digits}f}%"


def _first(rows: list[dict[str, Any]], **matches: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in matches.items()):
            return row
    match_text = ", ".join(f"{key}={value}" for key, value in matches.items())
    raise ValueError(f"Missing panel row matching {match_text}")


def _mean(rows: list[dict[str, Any]], key: str, *, method: str | None = None) -> float:
    values = [
        float(row[key])
        for row in rows
        if method is None or str(row.get("method")) == method
    ]
    if not values:
        raise ValueError(f"No values for {key!r}")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("Cannot take median of an empty sequence")
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _normalise_units(text: str) -> str:
    replacements = {
        " >=1 A": " >=1 Å",
        ">=1 A": ">=1 Å",
        " 1 A": " 1 Å",
        " 1-2 A": " 1-2 Å",
        " 1-5 A": " 1-5 Å",
        " 2-5 A": " 2-5 Å",
        " 5 A": " 5 Å",
        " A ": " Å ",
        " A,": " Å,",
        " A.": " Å.",
        " A;": " Å;",
        " A)": " Å)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _caption_context(stem: str) -> str:
    if stem.startswith("figure_s0"):
        return (
            "Reading guide: this panel defines the Cooper-TBDT task. The model predicts residue-level Cα displacement from "
            "a barrel-core-aligned AFDB-v6 starting structure to a paired experimental target; raw AFDB is the zero-displacement baseline. "
            "Region-resolved scoring is used because full-chain RMSD is dominated by the conserved β-barrel scaffold."
        )
    if stem.startswith("figure_3"):
        return (
            "Reading guide: Gold, Silver, and Bronze are evidence tiers with different scientific roles. Only Gold provides state-labeled "
            "TBDT AFDB-to-experimental supervision for primary training and held-out testing; Silver and Bronze are auxiliary resources and "
            "are not pooled into the primary test claims."
        )
    if stem.startswith("figure_2"):
        return (
            "Reading guide: these panels characterize the held-out Gold test target before modeling. RMSD values are raw AFDB zero-displacement "
            "errors after barrel-core alignment, and displacement bins summarize experimental target displacement magnitudes within each region."
        )
    if stem.startswith("figure_1"):
        return (
            "Reading guide: primary Cooper-TBDT results use a fixed single scaffold-prior GVP recipe evaluated over five training seeds. "
            "The validation-calibrated blend is a secondary coordinate candidate. ROC/PR panels are score-only residue-localization diagnostics "
            "and do not replace the coordinate-displacement endpoint."
        )
    if stem.startswith("figure_4"):
        return (
            "Reading guide: coordinate baselines must output explicit per-residue displacement vectors and are scored by the same regional RMSD "
            "contract as Cooper-TBDT. Positive MSE improvement means lower evaluation-region error than the raw AFDB zero-displacement baseline; "
            "negative values mean the baseline method is worse than leaving AFDB unchanged."
        )
    if stem.startswith("figure_5"):
        return (
            "Reading guide: critical ablations were run across the same five seeds, fixed Gold split, strict graph cache, and validation-only "
            "checkpoint selection as the primary model. The goal is to test methodological necessity and interpretation boundaries, not to claim "
            "that every component yields a large independent gain."
        )
    if stem.startswith("figure_6"):
        return (
            "Reading guide: TonB-box panels are negative mechanistic diagnostics. Coordinate RMSD can improve even when the model fails to move "
            "the TonB centroid in the correct direction or recover buried/exposed/unchanged exposure states."
        )
    if stem.startswith("figure_s1"):
        return (
            "Reading guide: residue-shift localization is a secondary score-only endpoint. A residue is positive when its experimental displacement "
            "magnitude is at least 1 Å; AUROC and average precision test ranking of moving residues, not prediction of a three-dimensional displacement vector."
        )
    if stem.startswith("figure_s2"):
        return (
            "Reading guide: seed-stability and selector-sensitivity panels audit whether the primary result depends on training randomness or checkpoint "
            "definition. Selectors are validation-only rules; test metrics are reported after the selector is fixed."
        )
    return (
        "Reading guide: this standalone panel is rendered from the same source artifacts as the manuscript figures. It should be interpreted with the "
        "Cooper-TBDT region-resolved displacement endpoint unless the caption states that the panel is a score-only diagnostic."
    )


def _write_caption(path: Path, title: str, caption: str) -> None:
    caption = _normalise_units(caption.rstrip())
    context = _caption_context(path.stem)
    text = f"{title}.\n\n{caption}\n\n{context}\n"
    path.write_text(text, encoding="utf-8")


def _legend(
    ax: Any,
    *,
    loc: str = "center left",
    bbox_to_anchor: tuple[float, float] | None = (1.02, 0.5),
    fontsize: float = 6.0,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            handlelength=1.9,
            labelspacing=0.18,
            borderpad=0.35,
            fontsize=fontsize,
        )


def _rmsd(rows: list[dict[str, Any]], method: str, region: str) -> float:
    return float(_first(rows, method=method, region=region)["prediction_error_rms"])


def _eval_improvement(rows: list[dict[str, Any]], method: str) -> float:
    row = _first(rows, method=method, region="eval")
    return float(row["eval_mse_improvement_vs_raw_afdb_fraction"]) * 100.0


def _bin_caption_values(rows: list[dict[str, Any]], region: str) -> dict[str, Any]:
    selected = [row for row in rows if row["region"] == region]
    if not selected:
        raise ValueError(f"Missing bin rows for region={region}")
    by_bin = {str(row["bin"]): row for row in selected}
    return {
        "total": int(selected[0]["region_residue_total"]),
        "targets": int(selected[0]["n_samples_with_region"]),
        "lt1": float(by_bin["lt_0p5"]["fraction"]) + float(by_bin["0p5_to_1"]["fraction"]),
        "one_to_two": float(by_bin["1_to_2"]["fraction"]),
        "ge2": float(by_bin["2_to_5"]["fraction"]) + float(by_bin["ge_5"]["fraction"]),
        "ge5": float(by_bin["ge_5"]["fraction"]),
    }


def _top_count(counts: dict[str, Any]) -> tuple[str, int]:
    if not counts:
        return "none", 0
    key, value = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0]
    return str(key), int(value)


def _variant_rmsd(rows: list[dict[str, Any]], variant: str, region: str) -> float:
    return float(_first(rows, variant=variant, region=region)["prediction_error_rms_mean"])


def _variant_sd(rows: list[dict[str, Any]], variant: str, region: str) -> float:
    return float(_first(rows, variant=variant, region=region)["prediction_error_rms_sd"])


def _variant_barrel(rows: list[dict[str, Any]], variant: str) -> float:
    return float(_first(rows, variant=variant, region="eval")["barrel_core_predicted_displacement_mean"])


def _paired_count(rows: list[dict[str, Any]], variant: str, region: str) -> dict[str, Any]:
    return _first(rows, variant=variant, region=region)


def _localization(summary: dict[tuple[str, str], dict[str, Any]], region: str, method: str) -> dict[str, Any]:
    return summary[(region, method)]


def _selector_value(rows: list[dict[str, Any]], selector: str, region: str, metric: str) -> tuple[float, float]:
    row = _first(rows, selector=selector, region=region, metric=metric)
    return float(row["mean"]), float(row["std"])


def _save_panel(
    *,
    out_dir: Path,
    figure_id: str,
    panel_id: str,
    slug: str,
    title: str,
    caption: str,
    plotter: PanelPlotter,
    style: Any,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float] = (4.8, 3.7),
    label_x: float = -0.12,
    label_y: float = 1.05,
) -> list[dict[str, Any]]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plotter(ax, style)
    add_panel_label(ax, panel_id, style, x=label_x, y=label_y)
    stem = out_dir / f"{figure_id}_{panel_id}_{slug}"
    written = save_figure(fig, stem, formats=formats, dpi=dpi)
    caption_path = stem.with_suffix(".txt")
    _write_caption(caption_path, title, caption)
    plt.close(fig)
    return [
        {
            "figure_id": figure_id,
            "panel_id": panel_id,
            "slug": slug,
            "title": title,
            "format": path.suffix.lstrip("."),
            "path": str(path),
            "caption_path": str(caption_path),
        }
        for path in written
    ]


def _build_figure_1(out_dir: Path, style: Any, formats: list[str], dpi: int, figure_out_dir: str) -> list[dict[str, Any]]:
    args = _default_args(main_results, out_dir=figure_out_dir)
    panel_a_rows = main_results._load_panel_a_rows(args)
    eval_delta_rows = main_results._load_delta_rows(args, "eval")
    plug_delta_rows = main_results._load_delta_rows(args, "plug")
    eval_delta_summary = main_results._load_delta_summary(args, "eval")
    plug_delta_summary = main_results._load_delta_summary(args, "plug")
    barrel_rows = main_results._load_barrel_core_values(args)
    plug_summary = load_external_localization_summary(args.curve_summary_csv, ("plug",))
    plug_curves = load_external_localization_curves(args.curve_points_csv, ("plug",))
    raw_eval = _first(panel_a_rows, method="raw", region="eval")
    primary_eval = _first(panel_a_rows, method="primary", region="eval")
    blend_eval = _first(panel_a_rows, method="blend", region="eval")
    raw_plug = _first(panel_a_rows, method="raw", region="plug")
    primary_plug = _first(panel_a_rows, method="primary", region="plug")
    blend_plug = _first(panel_a_rows, method="blend", region="plug")
    raw_tonb = _first(panel_a_rows, method="raw", region="tonb_box")
    primary_tonb = _first(panel_a_rows, method="primary", region="tonb_box")
    blend_tonb = _first(panel_a_rows, method="blend", region="tonb_box")
    blend_loc = _first(plug_summary, method="cooper_tbdt_scaffold_blend")
    af2_loc = _first(plug_summary, method="af2_low_plddt")
    anm_loc = _first(plug_summary, method="prody_anm_mobility")
    protcross_loc = _first(plug_summary, method="protcross_pocket_score")

    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="A",
            slug="aggregate_region_rmsd",
            title="Aggregate held-out Gold RMSD",
            caption=(
                "Held-out Gold region RMSD comparison. Raw AFDB, Cooper-TBDT single 5-seed, and "
                f"validation-calibrated blend are {float(raw_eval['rmsd']):.3f}, {float(primary_eval['rmsd']):.3f}, "
                f"and {float(blend_eval['rmsd']):.3f} A for the evaluation region; "
                f"{float(raw_plug['rmsd']):.3f}, {float(primary_plug['rmsd']):.3f}, and {float(blend_plug['rmsd']):.3f} A for plug; "
                f"and {float(raw_tonb['rmsd']):.3f}, {float(primary_tonb['rmsd']):.3f}, and {float(blend_tonb['rmsd']):.3f} A for TonB box."
            ),
            plotter=lambda ax, st: main_results._plot_panel_a(ax, panel_a_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.7),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="B",
            slug="evaluation_region_paired_delta",
            title="Evaluation-region paired target delta",
            caption=(
                "Evaluation-region paired target delta for the primary five-seed family. "
                f"{eval_delta_summary['n_improved']}/{eval_delta_summary['n_targets']} held-out targets improve versus raw AFDB; "
                f"the median Delta RMSD is {float(eval_delta_summary['median']):+.4f} A "
                f"(one-sided Wilcoxon p={float(eval_delta_summary['wilcoxon_p']):.3g})."
            ),
            plotter=lambda ax, st: main_results._plot_delta_panel(ax, eval_delta_rows, eval_delta_summary, "eval", st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.2, 3.7),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="C",
            slug="plug_paired_delta",
            title="Plug paired target delta",
            caption=(
                "Plug paired target delta for the primary five-seed family. "
                f"{plug_delta_summary['n_improved']}/{plug_delta_summary['n_targets']} held-out plug targets improve versus raw AFDB; "
                f"the median Delta RMSD is {float(plug_delta_summary['median']):+.4f} A "
                f"(one-sided Wilcoxon p={float(plug_delta_summary['wilcoxon_p']):.3g})."
            ),
            plotter=lambda ax, st: main_results._plot_delta_panel(ax, plug_delta_rows, plug_delta_summary, "plug", st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.2, 3.7),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="D",
            slug="barrel_core_displacement",
            title="Barrel-core predicted displacement",
            caption=(
                "Per-target barrel-core predicted displacement remains small. Mean predicted barrel-core displacement is "
                f"{_mean(barrel_rows, 'barrel_core_predicted_displacement_mean', method='raw'):.3f} A for raw AFDB, "
                f"{_mean(barrel_rows, 'barrel_core_predicted_displacement_mean', method='primary'):.3f} A for the primary five-seed family, "
                f"and {_mean(barrel_rows, 'barrel_core_predicted_displacement_mean', method='blend'):.3f} A for the validation-calibrated blend."
            ),
            plotter=lambda ax, st: main_results._plot_barrel_core_panel(ax, barrel_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.7),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="E",
            slug="plug_residue_shift_roc",
            title="Plug residue-shift ROC",
            caption=(
                "Plug residue-shift ROC for the score-only localization endpoint, where positives are residues with target displacement >=1 A. "
                f"The Cooper-TBDT blend has AUROC {float(blend_loc['auroc']):.3f}; external score baselines include "
                f"AF2 low pLDDT {float(af2_loc['auroc']):.3f}, ANM {float(anm_loc['auroc']):.3f}, and ProtCross {float(protcross_loc['auroc']):.3f}."
            ),
            plotter=lambda ax, st: main_results._plot_plug_localization_curve(
                ax,
                plug_curves,
                plug_summary,
                curve="roc",
                style=st,
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_1",
            panel_id="F",
            slug="plug_residue_shift_pr",
            title="Plug residue-shift PR",
            caption=(
                "Plug residue-shift precision-recall curve for the same score-only localization endpoint. "
                f"The Cooper-TBDT blend has AP {float(blend_loc['average_precision']):.3f}; AF2 low pLDDT, ANM, and ProtCross have AP "
                f"{float(af2_loc['average_precision']):.3f}, {float(anm_loc['average_precision']):.3f}, and {float(protcross_loc['average_precision']):.3f}, respectively. "
                f"The plug positive rate is {_pct(blend_loc['positive_rate'])}."
            ),
            plotter=lambda ax, st: (
                main_results._plot_plug_localization_curve(ax, plug_curves, plug_summary, curve="pr", style=st),
                _legend(ax),
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
    ]


def _build_figure_2(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(gold_test_displacement_landscape)
    publication_dir = Path(args.publication_dir)
    rmsd_rows = gold_test_displacement_landscape._load_raw_rmsd_rows(publication_dir)
    bin_rows = gold_test_displacement_landscape._load_bin_rows(publication_dir)
    raw_eval = _first(rmsd_rows, region="eval")
    raw_plug = _first(rmsd_rows, region="plug")
    raw_tonb = _first(rmsd_rows, region="tonb_box")
    raw_barrel = _first(rmsd_rows, region="barrel_core")
    eval_bins = _bin_caption_values(bin_rows, "eval")
    plug_bins = _bin_caption_values(bin_rows, "plug")
    tonb_bins = _bin_caption_values(bin_rows, "tonb_box")
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_2",
            panel_id="A",
            slug="raw_afdb_region_rmsd",
            title="Held-out Gold raw AFDB baseline RMSD",
            caption=(
                "Raw AFDB zero-displacement baseline on the held-out Gold test set. "
                f"RMSD is {float(raw_eval['raw_afdb_rmsd']):.3f} A for the evaluation region "
                f"(n={raw_eval['n_residues']} residues), {float(raw_plug['raw_afdb_rmsd']):.3f} A for plug "
                f"(n={raw_plug['n_residues']}), {float(raw_tonb['raw_afdb_rmsd']):.3f} A for TonB box "
                f"(n={raw_tonb['n_residues']}), and {float(raw_barrel['raw_afdb_rmsd']):.3f} A for barrel core "
                f"(n={raw_barrel['n_residues']})."
            ),
            plotter=lambda ax, st: gold_test_displacement_landscape._plot_raw_rmsd(ax, rmsd_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_2",
            panel_id="B",
            slug="evaluation_region_displacement_distribution",
            title="Evaluation-region displacement distribution",
            caption=(
                "Target displacement distribution for held-out evaluation-region residues. "
                f"The panel summarizes {eval_bins['total']} residues across {eval_bins['targets']} targets; "
                f"{_pct(eval_bins['lt1'])} are below 1 A, {_pct(eval_bins['one_to_two'])} are 1-2 A, "
                f"and {_pct(eval_bins['ge2'])} are >=2 A."
            ),
            plotter=lambda ax, st: gold_test_displacement_landscape._plot_region_bins(ax, bin_rows, "eval", st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_2",
            panel_id="C",
            slug="plug_displacement_distribution",
            title="Plug displacement distribution",
            caption=(
                "Target displacement distribution for held-out plug residues. "
                f"The panel summarizes {plug_bins['total']} residues across {plug_bins['targets']} targets; "
                f"{_pct(plug_bins['lt1'])} are below 1 A, {_pct(plug_bins['one_to_two'])} are 1-2 A, "
                f"and {_pct(plug_bins['ge2'])} are >=2 A."
            ),
            plotter=lambda ax, st: gold_test_displacement_landscape._plot_region_bins(ax, bin_rows, "plug", st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_2",
            panel_id="D",
            slug="tonb_box_displacement_distribution",
            title="TonB-box displacement distribution",
            caption=(
                "Target displacement distribution for held-out TonB-box residues. "
                f"The panel summarizes {tonb_bins['total']} residues across {tonb_bins['targets']} targets; "
                f"{_pct(tonb_bins['lt1'])} are below 1 A, {_pct(tonb_bins['one_to_two'])} are 1-2 A, "
                f"{_pct(tonb_bins['ge2'])} are >=2 A, and {_pct(tonb_bins['ge5'])} are >=5 A."
            ),
            plotter=lambda ax, st: gold_test_displacement_landscape._plot_region_bins(ax, bin_rows, "tonb_box", st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
    ]


def _build_figure_3(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(corpus_workflow)
    summary = corpus_workflow._load_summary(args)
    top_family = _top_count(summary["family_counts"])
    top_state = _top_count(summary["state_counts"])
    top_substrate = _top_count(summary["substrate_counts"])
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_3",
            panel_id="A",
            slug="corpus_construction_workflow",
            title="Corpus construction workflow",
            caption=(
                "Corpus construction workflow for Cooper-TBDT v1. The mixed manifest contains "
                f"{summary['mixed_rows']} rows spanning {summary['mixed_unique_uniprot']} UniProt accessions and "
                f"{summary['mixed_unique_pdb']} PDB entries, split into {summary['gold_manifest_rows']} Gold, "
                f"{summary['silver_manifest_rows']} Silver, and {summary['bronze_manifest_rows']} Bronze candidates."
            ),
            plotter=lambda ax, st: corpus_workflow._plot_workflow(ax, summary, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.8, 3.9),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_3",
            panel_id="B",
            slug="tier_roles",
            title="Tier roles",
            caption=(
                "Data-tier roles used for reporting and auxiliary experiments. Gold provides "
                f"{summary['gold_pairs']} paired supervised AFDB-v6 to experimental displacement examples; "
                f"Silver contributes {summary['silver_clean_graphs']} clean auxiliary beta-barrel graphs; "
                f"Bronze has {summary['bronze_usable_afdb']}/{summary['bronze_manifest_rows']} usable AFDB-v6 homologs for weak scaffold or self-supervised use only."
            ),
            plotter=lambda ax, st: corpus_workflow._plot_roles(ax, summary, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.8, 3.9),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_3",
            panel_id="C",
            slug="gold_split_overview",
            title="Gold split overview",
            caption=(
                "Gold supervised split overview. The final Gold set contains "
                f"{summary['gold_pairs']} paired structures from {summary['gold_unique_uniprot']} UniProt groups and "
                f"{summary['gold_unique_pdb']} PDB entries. The metadata split is train/val/test = "
                f"{summary['split_counts'].get('train', 0)}/{summary['split_counts'].get('val', 0)}/{summary['split_counts'].get('test', 0)} by UniProt group."
            ),
            plotter=lambda ax, st: corpus_workflow._plot_gold_split(ax, summary, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.8, 3.9),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_3",
            panel_id="D",
            slug="gold_family_state_substrate_distributions",
            title="Gold family / state / substrate distributions",
            caption=(
                "Gold family, state, and substrate composition. The largest family label is "
                f"{top_family[0]} ({top_family[1]} pairs), the dominant state label is {top_state[0]} ({top_state[1]} pairs), "
                f"and the most common substrate class is {top_substrate[0]} ({top_substrate[1]} pairs)."
            ),
            plotter=lambda ax, st: corpus_workflow._plot_distribution_stacks(ax, summary, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.8, 3.9),
        ),
    ]


def _build_figure_s0(out_dir: Path, style: Any, formats: list[str], dpi: int, figure_out_dir: str) -> list[dict[str, Any]]:
    args = _default_args(task_definition, out_dir=figure_out_dir)
    support_dir = figure_output_dir(args.out_dir, args.out_name)
    case_metrics = task_definition._load_case_metrics(args.paired_delta_csv, args.case_sample_id)
    exports = task_definition._export_case_coordinates(args, support_dir, case_metrics)
    case_summary = exports["summary"]
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s0",
            panel_id="A",
            slug="task_contract",
            title="Task: local C-alpha displacement",
            caption=(
                "Cooper-TBDT predicts local C-alpha displacement from an AFDB-v6 starting state to an experimental target state. "
                "Raw AFDB is the zero-displacement baseline, and model error is evaluated as regional RMSD of predicted versus target displacement rather than full-chain RMSD."
            ),
            plotter=lambda ax, st: task_definition._plot_task_contract(ax, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s0",
            panel_id="B",
            slug="region_mask_schematic",
            title="Region masks define endpoints",
            caption=(
                "Schematic of the region masks used as reporting endpoints: barrel core for scaffold/frame checks, plug and extracellular/evaluation regions for functional local motion, and TonB box as a sparse periplasmic coupling endpoint."
            ),
            plotter=lambda ax, st: task_definition._plot_region_schematic(ax, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s0",
            panel_id="C",
            slug="positive_coordinate_case_placeholder",
            title="Positive coordinate case placeholder",
            caption=(
                "Placeholder for a structural overlay case. The exported case is "
                f"{case_summary['sample_id']} ({case_summary['family']}, PDB {case_summary['pdb_id']}{case_summary['pdb_chain']}), "
                f"state/substrate {case_summary['state_label']}/{case_summary['substrate_class']}. "
                f"Evaluation-region RMSD changes from {float(case_summary['eval_raw_afdb_rmsd']):.3f} to "
                f"{float(case_summary['eval_method_rmsd']):.3f} A, and plug RMSD changes from "
                f"{float(case_summary['plug_raw_afdb_rmsd']):.3f} to {float(case_summary['plug_method_rmsd']):.3f} A."
            ),
            plotter=lambda ax, st: task_definition._plot_case_placeholder(ax, exports, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s0",
            panel_id="D",
            slug="case_level_coordinate_improvement",
            title="Case-level coordinate improvement",
            caption=(
                "Case-level coordinate RMSD for the same structural placeholder. Raw AFDB to Cooper-TBDT seed404 RMSD changes are "
                f"{case_metrics['eval']['raw_afdb_rmsd']:.3f}->{case_metrics['eval']['method_rmsd']:.3f} A for evaluation region, "
                f"{case_metrics['plug']['raw_afdb_rmsd']:.3f}->{case_metrics['plug']['method_rmsd']:.3f} A for plug, and "
                f"{case_metrics['barrel_core']['raw_afdb_rmsd']:.3f}->{case_metrics['barrel_core']['method_rmsd']:.3f} A for barrel core."
            ),
            plotter=lambda ax, st: task_definition._plot_case_bars(ax, case_metrics, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
    ]


def _build_figure_4(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(baseline_comparison)
    rows = baseline_comparison._load_plot_rows(args)
    methods_for_caption = (
        ("raw_af2_zero", "Raw AFDB"),
        ("foldseek_nearest_template", "Foldseek"),
        ("usalign_nearest_template", "US-align"),
        ("barrel_frame_ridge", "barrel-frame ridge"),
        ("cooper_tbdt_scaffold_single_5seed_mean", "Cooper-TBDT single"),
        ("cooper_tbdt_scaffold_blend", "Cooper-TBDT blend"),
    )

    def rmsd_sentence(region: str) -> str:
        values = [f"{label} {_rmsd(rows, method, region):.3f} A" for method, label in methods_for_caption]
        return "; ".join(values) + "."

    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_4",
            panel_id="A",
            slug="evaluation_region_rmsd_baselines",
            title="Evaluation-region RMSD",
            caption=(
                "Evaluation-region coordinate RMSD for raw AFDB, template/linear baselines, and Cooper-TBDT methods. "
                + rmsd_sentence("eval")
            ),
            plotter=lambda ax, st: baseline_comparison._plot_rmsd_panel(ax, rows, "eval", st, show_ylabel=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.7, 4.1),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_4",
            panel_id="B",
            slug="plug_rmsd_baselines",
            title="Plug RMSD",
            caption=(
                "Plug coordinate RMSD for the same baseline comparison. "
                + rmsd_sentence("plug")
            ),
            plotter=lambda ax, st: baseline_comparison._plot_rmsd_panel(ax, rows, "plug", st, show_ylabel=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.7, 4.1),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_4",
            panel_id="C",
            slug="tonb_box_rmsd_baselines",
            title="TonB-box RMSD",
            caption=(
                "TonB-box coordinate RMSD for the same baseline comparison. "
                + rmsd_sentence("tonb_box")
            ),
            plotter=lambda ax, st: baseline_comparison._plot_rmsd_panel(ax, rows, "tonb_box", st, show_ylabel=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.7, 4.1),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_4",
            panel_id="D",
            slug="evaluation_region_mse_improvement",
            title="Evaluation-region MSE improvement",
            caption=(
                "Evaluation-region MSE improvement versus raw AFDB. Positive values mean lower error than the zero-displacement baseline. "
                f"Foldseek and US-align are {_eval_improvement(rows, 'foldseek_nearest_template'):+.1f}% and "
                f"{_eval_improvement(rows, 'usalign_nearest_template'):+.1f}%; barrel-frame ridge is "
                f"{_eval_improvement(rows, 'barrel_frame_ridge'):+.1f}%; Cooper-TBDT single and blend are "
                f"{_eval_improvement(rows, 'cooper_tbdt_scaffold_single_5seed_mean'):+.1f}% and "
                f"{_eval_improvement(rows, 'cooper_tbdt_scaffold_blend'):+.1f}%."
            ),
            plotter=lambda ax, st: baseline_comparison._plot_improvement_panel(ax, rows, st, show_ylabel=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.7, 4.1),
        ),
    ]


def _build_figure_5(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(critical_ablation)
    region_rows = critical_ablation._load_region_rows(args)
    paired_rows = critical_ablation._load_paired_count_rows(args)
    full_eval = _variant_rmsd(region_rows, "full_scaffold_prior", "eval")
    no_state_eval = _variant_rmsd(region_rows, "no_state_conditioning", "eval")
    no_conf_eval = _variant_rmsd(region_rows, "no_afdb_confidence_features", "eval")
    no_weights_eval = _variant_rmsd(region_rows, "no_region_loss_weights", "eval")
    no_anchor_eval = _variant_rmsd(region_rows, "no_scaffold_anchor", "eval")
    raw_eval = _variant_rmsd(region_rows, "raw_afdb_zero", "eval")
    full_plug = _variant_rmsd(region_rows, "full_scaffold_prior", "plug")
    no_state_plug = _variant_rmsd(region_rows, "no_state_conditioning", "plug")
    no_conf_plug = _variant_rmsd(region_rows, "no_afdb_confidence_features", "plug")
    no_weights_plug = _variant_rmsd(region_rows, "no_region_loss_weights", "plug")
    no_anchor_plug = _variant_rmsd(region_rows, "no_scaffold_anchor", "plug")
    raw_plug = _variant_rmsd(region_rows, "raw_afdb_zero", "plug")
    full_tonb = _variant_rmsd(region_rows, "full_scaffold_prior", "tonb_box")
    no_state_tonb = _variant_rmsd(region_rows, "no_state_conditioning", "tonb_box")
    no_conf_tonb = _variant_rmsd(region_rows, "no_afdb_confidence_features", "tonb_box")
    no_weights_tonb = _variant_rmsd(region_rows, "no_region_loss_weights", "tonb_box")
    no_anchor_tonb = _variant_rmsd(region_rows, "no_scaffold_anchor", "tonb_box")
    raw_tonb = _variant_rmsd(region_rows, "raw_afdb_zero", "tonb_box")
    full_barrel = _variant_barrel(region_rows, "full_scaffold_prior")
    no_anchor_barrel = _variant_barrel(region_rows, "no_scaffold_anchor")
    no_tbdt_barrel = _variant_barrel(region_rows, "no_tbdt_conditioning")
    no_weights_barrel = _variant_barrel(region_rows, "no_region_loss_weights")
    full_eval_pair = _paired_count(paired_rows, "full_scaffold_prior", "eval")
    full_plug_pair = _paired_count(paired_rows, "full_scaffold_prior", "plug")
    no_state_plug_pair = _paired_count(paired_rows, "no_state_conditioning", "plug")
    no_anchor_plug_pair = _paired_count(paired_rows, "no_scaffold_anchor", "plug")
    no_weights_plug_pair = _paired_count(paired_rows, "no_region_loss_weights", "plug")
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_5",
            panel_id="A1",
            slug="evaluation_region_rmsd_ablation",
            title="Evaluation-region RMSD",
            caption=(
                "Critical five-seed ablation, evaluation-region RMSD. Raw AFDB is "
                f"{raw_eval:.3f} A; the full scaffold-prior model is {full_eval:.3f} A. "
                f"No state conditioning is {no_state_eval:.3f} A, no AFDB confidence features is {no_conf_eval:.3f} A, "
                f"no region loss weights is {no_weights_eval:.3f} A, and no scaffold anchor is {no_anchor_eval:.3f} A. "
                "The x-axis is truncated to make the small but methodologically important RMSD differences visible."
            ),
            plotter=lambda ax, st: critical_ablation._plot_region_panel(ax, region_rows, "eval", st, show_labels=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.5, 4.0),
            label_x=-0.2,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_5",
            panel_id="A2",
            slug="plug_rmsd_ablation",
            title="Plug RMSD",
            caption=(
                "Critical five-seed ablation, plug RMSD. Raw AFDB is "
                f"{raw_plug:.3f} A; the full scaffold-prior model is {full_plug:.3f} A. "
                f"No state conditioning is {no_state_plug:.3f} A, no AFDB confidence features is {no_conf_plug:.3f} A, "
                f"no region loss weights is {no_weights_plug:.3f} A, and no scaffold anchor is {no_anchor_plug:.3f} A. "
                "The x-axis is truncated to make the small but methodologically important RMSD differences visible."
            ),
            plotter=lambda ax, st: critical_ablation._plot_region_panel(ax, region_rows, "plug", st, show_labels=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.5, 4.0),
            label_x=-0.2,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_5",
            panel_id="A3",
            slug="tonb_box_rmsd_ablation",
            title="TonB-box RMSD",
            caption=(
                "Critical five-seed ablation, TonB-box RMSD. Raw AFDB is "
                f"{raw_tonb:.3f} A; the full scaffold-prior model is {full_tonb:.3f} A. "
                f"No state conditioning is {no_state_tonb:.3f} A, no AFDB confidence features is {no_conf_tonb:.3f} A, "
                f"no region loss weights is {no_weights_tonb:.3f} A, and no scaffold anchor is {no_anchor_tonb:.3f} A. "
                "The x-axis is truncated to compare ablation variants within this high-RMSD TonB endpoint; absolute TonB scale is shown in the raw-baseline and TonB mechanism-boundary figures."
            ),
            plotter=lambda ax, st: critical_ablation._plot_region_panel(ax, region_rows, "tonb_box", st, show_labels=True),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.5, 4.0),
            label_x=-0.2,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_5",
            panel_id="B",
            slug="barrel_core_predicted_displacement_ablation",
            title="Barrel-core predicted displacement",
            caption=(
                "Barrel-core predicted displacement in the critical five-seed ablation. "
                f"The full scaffold-prior model predicts {full_barrel:.3f} A on average, whereas removing the scaffold anchor raises this to "
                f"{no_anchor_barrel:.3f} A. No TBDT conditioning is {no_tbdt_barrel:.3f} A and no region loss weights is {no_weights_barrel:.3f} A."
            ),
            plotter=lambda ax, st: critical_ablation._plot_barrel_panel(ax, region_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.5, 4.0),
            label_x=-0.2,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_5",
            panel_id="C",
            slug="paired_improvement_counts_ablation",
            title="Paired improvement counts",
            caption=(
                "Paired target-level improvement counts for evaluation and plug regions after averaging the five seeds per target. "
                f"The full model improves {full_eval_pair['n_improved']}/{full_eval_pair['n_targets']} evaluation targets "
                f"and {full_plug_pair['n_improved']}/{full_plug_pair['n_targets']} plug targets. "
                f"Plug improvement drops to {no_state_plug_pair['n_improved']}/{no_state_plug_pair['n_targets']} without state conditioning, "
                f"{no_anchor_plug_pair['n_improved']}/{no_anchor_plug_pair['n_targets']} without the scaffold anchor, and "
                f"{no_weights_plug_pair['n_improved']}/{no_weights_plug_pair['n_targets']} without region loss weights."
            ),
            plotter=lambda ax, st: critical_ablation._plot_paired_counts(ax, paired_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(7.0, 4.0),
            label_x=-0.1,
        ),
    ]


def _build_figure_6(out_dir: Path, style: Any, formats: list[str], dpi: int, figure_out_dir: str) -> list[dict[str, Any]]:
    args = _default_args(tonb_mechanistic_boundary, out_dir=figure_out_dir)
    tonb_rows = tonb_mechanistic_boundary._load_tonb_rows(args.tonb_state_csv)
    case = tonb_mechanistic_boundary._case_row(tonb_rows, args.case_sample_id)
    support_dir = figure_output_dir(args.out_dir, args.out_name)
    exports = tonb_mechanistic_boundary._export_case_data(args, support_dir, case)
    case_id = str(case["sample_id"])
    target_centroid_median = _median([float(row["target_centroid_displacement"]) for row in tonb_rows])
    predicted_centroid_median = _median([float(row["predicted_centroid_displacement"]) for row in tonb_rows])
    target_counts = Counter(str(row["target_state"]) for row in tonb_rows)
    predicted_counts = Counter(str(row["predicted_state"]) for row in tonb_rows)
    compatible_count = sum(1 for row in tonb_rows if row["direction_compatible"])
    cosine_median = _median([float(row["centroid_displacement_cosine"]) for row in tonb_rows])
    cosine_mean = sum(float(row["centroid_displacement_cosine"]) for row in tonb_rows) / len(tonb_rows)
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_6",
            panel_id="A",
            slug="tonb_case_study_placeholder",
            title="TonB case-study placeholder",
            caption=(
                "TonB case-study placeholder with exported coordinates for manual structural drawing. "
                f"The selected case is {case_id}; the experimental target is "
                f"{tonb_mechanistic_boundary.STATE_LABELS.get(str(case['target_state']), case['target_state'])} and the prediction is "
                f"{tonb_mechanistic_boundary.STATE_LABELS.get(str(case['predicted_state']), case['predicted_state'])}. "
                f"Target and predicted TonB centroid shifts from AFDB are {float(case['target_centroid_displacement']):.2f} A and "
                f"{float(case['predicted_centroid_displacement']):.2f} A, respectively. Exported CA PDB and CSV files are listed in the panel."
            ),
            plotter=lambda ax, st: tonb_mechanistic_boundary._plot_case_placeholder(ax, case, exports, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_6",
            panel_id="B",
            slug="target_vs_predicted_centroid_displacement",
            title="Target vs predicted centroid displacement",
            caption=(
                "TonB centroid displacement from the AFDB starting state for each held-out target with a TonB mask. "
                f"Across {len(tonb_rows)} targets, median target centroid displacement is {target_centroid_median:.2f} A, whereas "
                f"median predicted centroid displacement is {predicted_centroid_median:.2f} A. "
                f"The highlighted case {case_id} has target and predicted shifts of {float(case['target_centroid_displacement']):.2f} A and "
                f"{float(case['predicted_centroid_displacement']):.2f} A."
            ),
            plotter=lambda ax, st: tonb_mechanistic_boundary._plot_centroid_comparison(ax, tonb_rows, case_id, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_6",
            panel_id="C",
            slug="exposure_state_confusion",
            title="Exposure-state confusion",
            caption=(
                "TonB exposure-state confusion for the scaffold blend. Experimental target states are "
                f"{target_counts.get('buried_like', 0)} buried-like, {target_counts.get('exposed_like', 0)} exposed-like, "
                f"and {target_counts.get('unchanged', 0)} unchanged. Predicted states are "
                f"{predicted_counts.get('buried_like', 0)} buried-like, {predicted_counts.get('exposed_like', 0)} exposed-like, "
                f"and {predicted_counts.get('unchanged', 0)} unchanged, showing that the model mostly leaves TonB unchanged."
            ),
            plotter=lambda ax, st: tonb_mechanistic_boundary._plot_state_stacks(ax, tonb_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_6",
            panel_id="D",
            slug="direction_compatibility",
            title="Direction compatibility",
            caption=(
                "TonB centroid direction compatibility and cosine statistics. "
                f"{compatible_count}/{len(tonb_rows)} targets are direction-compatible ({compatible_count / len(tonb_rows):.0%}); "
                f"the median centroid displacement cosine is {cosine_median:.3f} and the mean cosine is {cosine_mean:.3f}. "
                "This supports reporting TonB as a hard, unresolved mechanistic endpoint despite weak coordinate-RMSD gains."
            ),
            plotter=lambda ax, st: tonb_mechanistic_boundary._plot_direction_stats(ax, tonb_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 3.8),
        ),
    ]


def _build_figure_s1(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(residue_shift_localization)
    all_regions = ("eval", "plug", "tonb_box")
    summary_rows = load_external_localization_summary(args.curve_summary_csv, all_regions)
    curve_rows = load_external_localization_curves(args.curve_points_csv, residue_shift_localization.REGIONS)
    summary = residue_shift_localization._summary_lookup(summary_rows)

    def metric(region: str, method: str, key: str) -> float:
        return float(_localization(summary, region, method)[key])

    def best(region: str, key: str) -> dict[str, Any]:
        selected = [row for row in summary_rows if str(row["region"]) == region]
        return max(selected, key=lambda row: float(row[key]))

    eval_best_roc = best("eval", "auroc")
    eval_best_pr = best("eval", "average_precision")
    plug_best_roc = best("plug", "auroc")
    plug_best_pr = best("plug", "average_precision")
    tonb_blend = _localization(summary, "tonb_box", "cooper_tbdt_scaffold_blend")
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="A",
            slug="evaluation_region_roc",
            title="Evaluation-region ROC",
            caption=(
                "Evaluation-region ROC for residue-shift localization, with positives defined as target displacement >=1 A. "
                f"The Cooper-TBDT blend AUROC is {metric('eval', 'cooper_tbdt_scaffold_blend', 'auroc'):.3f}. "
                f"The best AUROC in this panel is {eval_best_roc['method_label']} at {float(eval_best_roc['auroc']):.3f}; "
                f"AF2 low pLDDT and ANM are {metric('eval', 'af2_low_plddt', 'auroc'):.3f} and {metric('eval', 'prody_anm_mobility', 'auroc'):.3f}."
            ),
            plotter=lambda ax, st: residue_shift_localization._plot_curve(
                ax, curve_rows, summary, region="eval", curve="roc", style=st, show_legend=True
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="B",
            slug="evaluation_region_pr",
            title="Evaluation-region PR",
            caption=(
                "Evaluation-region precision-recall curve for residue-shift localization. "
                f"The Cooper-TBDT blend AP is {metric('eval', 'cooper_tbdt_scaffold_blend', 'average_precision'):.3f}. "
                f"The best AP is {eval_best_pr['method_label']} at {float(eval_best_pr['average_precision']):.3f}; "
                f"AF2 low pLDDT and ProtCross are {metric('eval', 'af2_low_plddt', 'average_precision'):.3f} and "
                f"{metric('eval', 'protcross_pocket_score', 'average_precision'):.3f}."
            ),
            plotter=lambda ax, st: residue_shift_localization._plot_curve(
                ax, curve_rows, summary, region="eval", curve="pr", style=st, show_legend=True
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="C",
            slug="localization_metric_heatmap",
            title="Localization metrics",
            caption=(
                "Summary heatmap of AUROC and average precision for evaluation-region and plug residue-shift localization. "
                f"Best evaluation-region AUROC/AP are {eval_best_roc['method_label']} {float(eval_best_roc['auroc']):.3f} and "
                f"{eval_best_pr['method_label']} {float(eval_best_pr['average_precision']):.3f}. "
                f"Best plug AUROC/AP are {plug_best_roc['method_label']} {float(plug_best_roc['auroc']):.3f} and "
                f"{plug_best_pr['method_label']} {float(plug_best_pr['average_precision']):.3f}."
            ),
            plotter=lambda ax, st: residue_shift_localization._plot_metric_heatmap(ax, summary_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.6, 4.2),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="D",
            slug="plug_roc",
            title="Plug ROC",
            caption=(
                "Plug ROC for residue-shift localization. "
                f"The Cooper-TBDT blend AUROC is {metric('plug', 'cooper_tbdt_scaffold_blend', 'auroc'):.3f}; "
                f"AF2 low pLDDT, ANM, and ProtCross are {metric('plug', 'af2_low_plddt', 'auroc'):.3f}, "
                f"{metric('plug', 'prody_anm_mobility', 'auroc'):.3f}, and {metric('plug', 'protcross_pocket_score', 'auroc'):.3f}, respectively."
            ),
            plotter=lambda ax, st: residue_shift_localization._plot_curve(
                ax, curve_rows, summary, region="plug", curve="roc", style=st, show_legend=True
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="E",
            slug="plug_pr",
            title="Plug PR",
            caption=(
                "Plug precision-recall curve for residue-shift localization. "
                f"The Cooper-TBDT blend AP is {metric('plug', 'cooper_tbdt_scaffold_blend', 'average_precision'):.3f}; "
                f"AF2 low pLDDT, ANM, and ProtCross are {metric('plug', 'af2_low_plddt', 'average_precision'):.3f}, "
                f"{metric('plug', 'prody_anm_mobility', 'average_precision'):.3f}, and "
                f"{metric('plug', 'protcross_pocket_score', 'average_precision'):.3f}."
            ),
            plotter=lambda ax, st: residue_shift_localization._plot_curve(
                ax, curve_rows, summary, region="plug", curve="pr", style=st, show_legend=True
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(4.9, 3.8),
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s1",
            panel_id="F",
            slug="tonb_ranking_caveat",
            title="TonB ranking caveat",
            caption=(
                "TonB-box residue-ranking caveat. The held-out TonB slice has "
                f"{int(float(tonb_blend['n_positive']))}/{int(float(tonb_blend['n_residues']))} positive residues and "
                f"{int(float(tonb_blend['n_negative']))} negative residue, so AP is inflated by prevalence. "
                f"The Cooper-TBDT blend AUROC/AP are {float(tonb_blend['auroc']):.3f}/{float(tonb_blend['average_precision']):.3f}; "
                "AUROC is shown but is also unstable because it depends on the ranking of a single negative residue."
            ),
            plotter=lambda ax, st: (
                residue_shift_localization._plot_tonb_caveat(ax, summary_rows, st),
                _legend(ax, loc="upper right", bbox_to_anchor=None),
            ),
            style=style,
            formats=formats,
            dpi=dpi,
            figsize=(5.4, 4.2),
        ),
    ]


def _build_figure_s2(out_dir: Path, style: Any, formats: list[str], dpi: int) -> list[dict[str, Any]]:
    args = _default_args(seed_stability_selector)
    seed_rows = seed_stability_selector._load_seed_rows(args.seed_summary_csv)
    selector_rows = seed_stability_selector._load_selector_rows(args.selector_summary_csv)
    seeds = [int(row["seed"]) for row in seed_rows]
    eval_imp = [float(row["eval_mse_improvement_vs_zero_fraction"]) * 100.0 for row in seed_rows]
    plug_imp = [float(row["plug_mse_improvement_vs_zero_fraction"]) * 100.0 for row in seed_rows]
    tonb_imp = [float(row["tonb_box_mse_improvement_vs_zero_fraction"]) * 100.0 for row in seed_rows]
    epochs = [int(row["selected_epoch"]) for row in seed_rows]
    selection_eval = _selector_value(selector_rows, "best-selection", "eval", "prediction_error_rms")
    selection_plug = _selector_value(selector_rows, "best-selection", "plug", "prediction_error_rms")
    disp15_eval = _selector_value(selector_rows, "best-disp1to5", "eval", "prediction_error_rms")
    flex_eval = _selector_value(selector_rows, "best-flex", "eval", "prediction_error_rms")
    raw_tonb = float(seed_rows[0]["tonb_box_zero_error_rms"])
    selection_tonb = _selector_value(selector_rows, "best-selection", "tonb_box", "prediction_error_rms")
    flex_tonb = _selector_value(selector_rows, "best-flex", "tonb_box", "prediction_error_rms")
    selection_barrel = _selector_value(selector_rows, "best-selection", "barrel_core", "predicted_displacement_mean")
    flex_barrel = _selector_value(selector_rows, "best-flex", "barrel_core", "predicted_displacement_mean")
    return [
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s2",
            panel_id="A",
            slug="best_selection_seed_stability",
            title="Best-selection seed stability",
            caption=(
                "Seed stability of the primary best-selection checkpoint rule across five fixed training seeds "
                f"({', '.join(str(seed) for seed in seeds)}). Evaluation-region MSE improvement ranges from "
                f"{min(eval_imp):.2f}% to {max(eval_imp):.2f}%, plug from {min(plug_imp):.2f}% to {max(plug_imp):.2f}%, "
                f"and TonB box from {min(tonb_imp):.2f}% to {max(tonb_imp):.2f}% versus raw AFDB."
            ),
            plotter=lambda ax, st: seed_stability_selector._plot_seed_improvements(ax, seed_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s2",
            panel_id="B",
            slug="validation_selected_checkpoints",
            title="Validation-selected checkpoints",
            caption=(
                "Validation-selected checkpoint epochs for the same five primary training seeds. "
                f"The selected epochs are {', '.join(str(epoch) for epoch in epochs)}, spanning epochs "
                f"{min(epochs)}-{max(epochs)}, consistent with early validation selection under the small Gold-only training set."
            ),
            plotter=lambda ax, st: seed_stability_selector._plot_selected_epochs(ax, seed_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s2",
            panel_id="C",
            slug="selector_sensitivity_eval_plug",
            title="Selector sensitivity: eval/plug",
            caption=(
                "Checkpoint-selector sensitivity for evaluation and plug RMSD. The primary best-selection selector gives "
                f"evaluation RMSD {selection_eval[0]:.3f} +/- {selection_eval[1]:.3f} A and plug RMSD "
                f"{selection_plug[0]:.3f} +/- {selection_plug[1]:.3f} A. Broad-band best-disp1to5 gives evaluation RMSD "
                f"{disp15_eval[0]:.3f} +/- {disp15_eval[1]:.3f} A, while the flex-biased selector gives "
                f"{flex_eval[0]:.3f} +/- {flex_eval[1]:.3f} A."
            ),
            plotter=lambda ax, st: seed_stability_selector._plot_selector_rmsd(ax, selector_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
        *_save_panel(
            out_dir=out_dir,
            figure_id="figure_s2",
            panel_id="D",
            slug="tonb_selector_tradeoff",
            title="TonB-selector tradeoff",
            caption=(
                "TonB selector tradeoff between TonB-box RMSD and barrel-core predicted displacement. Raw AFDB TonB RMSD is "
                f"{raw_tonb:.3f} A. The primary best-selection selector gives TonB RMSD {selection_tonb[0]:.3f} +/- "
                f"{selection_tonb[1]:.3f} A with barrel-core predicted displacement {selection_barrel[0]:.3f} +/- "
                f"{selection_barrel[1]:.3f} A. The flex-biased selector gives TonB RMSD {flex_tonb[0]:.3f} +/- "
                f"{flex_tonb[1]:.3f} A and barrel-core predicted displacement {flex_barrel[0]:.3f} +/- {flex_barrel[1]:.3f} A."
            ),
            plotter=lambda ax, st: seed_stability_selector._plot_tonb_tradeoff(ax, selector_rows, seed_rows, st),
            style=style,
            formats=formats,
            dpi=dpi,
        ),
    ]


def build_archive(args: argparse.Namespace) -> list[dict[str, Any]]:
    style = get_style(args.style)
    apply_style(style)
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {item.strip() for item in str(args.only).split(",") if item.strip()}

    records: list[dict[str, Any]] = []
    builders: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = [
        ("figure_1", lambda: _build_figure_1(out_dir, style, formats, int(args.dpi), str(args.figure_out_dir))),
        ("figure_2", lambda: _build_figure_2(out_dir, style, formats, int(args.dpi))),
        ("figure_3", lambda: _build_figure_3(out_dir, style, formats, int(args.dpi))),
        ("figure_s0", lambda: _build_figure_s0(out_dir, style, formats, int(args.dpi), str(args.figure_out_dir))),
        ("figure_4", lambda: _build_figure_4(out_dir, style, formats, int(args.dpi))),
        ("figure_5", lambda: _build_figure_5(out_dir, style, formats, int(args.dpi))),
        ("figure_6", lambda: _build_figure_6(out_dir, style, formats, int(args.dpi), str(args.figure_out_dir))),
        ("figure_s1", lambda: _build_figure_s1(out_dir, style, formats, int(args.dpi))),
        ("figure_s2", lambda: _build_figure_s2(out_dir, style, formats, int(args.dpi))),
    ]
    known = {figure_id for figure_id, _ in builders}
    unknown = sorted(only - known)
    if unknown:
        raise ValueError(f"Unknown figure selector(s): {', '.join(unknown)}")
    for figure_id, builder in builders:
        if _selected(figure_id, only):
            records.extend(builder())

    write_csv_rows(
        out_dir / "panel_archive_manifest.csv",
        records,
        ["figure_id", "panel_id", "slug", "title", "format", "path", "caption_path"],
    )
    (out_dir / "README.md").write_text(
        "# Cooper-TBDT Standalone Panel Review Archive\n\n"
        "Each file in this directory is rendered directly from source artifacts and panel plotting code. "
        "These are not crops of combined figure PNGs. Each panel has a same-stem `.txt` caption generated from "
        "the underlying plotting data.\n\n"
        "See `panel_archive_manifest.csv` for the complete file list and caption paths.\n",
        encoding="utf-8",
    )
    return records


def main() -> None:
    args = parse_args()
    records = build_archive(args)
    for path in sorted({row["path"] for row in records}):
        print(path)
    print(Path(args.out_dir) / "panel_archive_manifest.csv")


if __name__ == "__main__":
    main()
