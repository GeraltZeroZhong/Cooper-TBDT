"""Build the Cooper-TBDT task-definition and positive-case placeholder figure."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
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


REGION_LABELS = {
    "eval": "Evaluation region",
    "plug": "Plug",
    "tonb_box": "TonB box",
    "barrel_core": "Barrel core",
}
CASE_REGIONS = ("eval", "plug", "tonb_box", "barrel_core")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT task-definition figure")
    parser.add_argument(
        "--paired-delta-csv",
        default="artifacts/tbdt_v1/publication_report/primary_model_paired_delta_samples.csv",
        help="Primary 5-seed paired-delta sample CSV.",
    )
    parser.add_argument(
        "--graph-dir",
        default="data/processed_tbdt_gold_graphs",
        help="Processed graph directory for case-study coordinate export.",
    )
    parser.add_argument(
        "--prediction-dir",
        default="artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_404_best-selection_test",
        help="Representative primary-model prediction directory for case-study coordinate export.",
    )
    parser.add_argument(
        "--case-sample-id",
        default="btub_p06129_3m8d_a",
        help="Positive held-out example used as a structural placeholder.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="task_definition_and_positive_case", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _load_case_metrics(path: str | Path, sample_id: str) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(path)
    by_region: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("sample_id") != sample_id
            or row.get("method") != "cooper_tbdt_scaffold_single_5seed_mean"
            or row.get("aggregation") != "per_target_seed_mean"
        ):
            continue
        region = row.get("region", "")
        if region not in CASE_REGIONS:
            continue
        by_region[region] = {
            "sample_id": sample_id,
            "region": region,
            "raw_afdb_rmsd": finite_float(row.get("raw_af2_rmsd"), field="raw_af2_rmsd", source=str(path)),
            "method_rmsd": finite_float(row.get("method_rmsd"), field="method_rmsd", source=str(path)),
            "delta_rmsd_method_minus_raw": finite_float(
                row.get("delta_rmsd_method_minus_raw"),
                field="delta_rmsd_method_minus_raw",
                source=str(path),
            ),
            "improved": row.get("improved", "").lower() == "true",
            "source": str(path),
        }
    missing = [region for region in ("eval", "plug", "barrel_core") if region not in by_region]
    if missing:
        raise ValueError(f"Missing case metrics for {sample_id}: {', '.join(missing)}")
    return by_region


def _parse_residue_id(residue_id: str) -> tuple[str, int]:
    if "_" in residue_id:
        chain, residue = residue_id.split("_", 1)
    else:
        chain, residue = "A", residue_id
    try:
        return chain[:1] or "A", int(residue)
    except ValueError:
        digits = "".join(ch for ch in residue if ch.isdigit() or ch == "-")
        return chain[:1] or "A", int(digits or 0)


def _region_code(index: int, masks: dict[str, np.ndarray]) -> tuple[str, str, float]:
    if bool(masks.get("tonb_box", np.zeros(0, dtype=bool))[index]):
        return "TON", "tonb_box", 100.0
    if bool(masks.get("plug", np.zeros(0, dtype=bool))[index]):
        return "PLG", "plug", 80.0
    if bool(masks.get("barrel_core", np.zeros(0, dtype=bool))[index]):
        return "BAR", "barrel_core", 60.0
    if bool(masks.get("eval", np.zeros(0, dtype=bool))[index]):
        return "EVL", "eval", 40.0
    return "GLY", "other", 20.0


def _write_case_pdb(path: Path, residue_ids: list[str], coords: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    lines: list[str] = []
    for serial, (residue_id, xyz) in enumerate(zip(residue_ids, coords), start=1):
        chain, resseq = _parse_residue_id(str(residue_id))
        resname, _, bfactor = _region_code(serial - 1, masks)
        lines.append(
            f"ATOM  {serial:5d} {'CA':^4s} {resname:>3s} {chain:1s}{resseq:4d}    "
            f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
            f"  1.00{bfactor:6.2f}           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _export_case_coordinates(args: argparse.Namespace, out_dir: Path, case_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import torch

    sample_id = str(args.case_sample_id)
    graph_path = Path(args.graph_dir) / f"{sample_id}.pt"
    prediction_path = Path(args.prediction_dir) / f"{sample_id}.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Case graph not found: {graph_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Case prediction not found: {prediction_path}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    pred = torch.load(prediction_path, map_location="cpu", weights_only=False)
    pos = graph["pos"].detach().cpu().numpy()
    target = pos + graph["y_delta"].detach().cpu().numpy()
    prediction = pos + pred["pred_delta"].detach().cpu().numpy()
    residue_ids = [str(item) for item in graph["residue_ids"]]
    masks = {
        "eval": graph["eval_mask"].detach().cpu().numpy().astype(bool),
        "plug": graph["plug_mask"].detach().cpu().numpy().astype(bool),
        "tonb_box": graph["tonb_box_mask"].detach().cpu().numpy().astype(bool),
        "barrel_core": graph["barrel_core_mask"].detach().cpu().numpy().astype(bool),
    }

    prefix = f"case_{sample_id}"
    afdb_pdb = out_dir / f"{prefix}_afdb_aligned_ca.pdb"
    target_pdb = out_dir / f"{prefix}_experimental_target_aligned_ca.pdb"
    pred_pdb = out_dir / f"{prefix}_cooper_seed404_prediction_aligned_ca.pdb"
    _write_case_pdb(afdb_pdb, residue_ids, pos, masks)
    _write_case_pdb(target_pdb, residue_ids, target, masks)
    _write_case_pdb(pred_pdb, residue_ids, prediction, masks)

    centroid_rows: list[dict[str, Any]] = []
    for region in CASE_REGIONS:
        if region not in masks or not bool(masks[region].any()):
            continue
        afdb_centroid = pos[masks[region]].mean(axis=0)
        target_centroid = target[masks[region]].mean(axis=0)
        pred_centroid = prediction[masks[region]].mean(axis=0)
        for structure, centroid in (
            ("afdb", afdb_centroid),
            ("experimental_target", target_centroid),
            ("cooper_seed404_prediction", pred_centroid),
        ):
            vector = centroid - afdb_centroid
            centroid_rows.append(
                {
                    "sample_id": sample_id,
                    "region": region,
                    "structure": structure,
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "delta_from_afdb_x": float(vector[0]),
                    "delta_from_afdb_y": float(vector[1]),
                    "delta_from_afdb_z": float(vector[2]),
                    "centroid_displacement_from_afdb": float(np.linalg.norm(vector)),
                    "n_residues": int(masks[region].sum()),
                }
            )
    centroid_csv = out_dir / f"{prefix}_region_centroids.csv"
    write_csv_rows(
        centroid_csv,
        centroid_rows,
        [
            "sample_id",
            "region",
            "structure",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "delta_from_afdb_x",
            "delta_from_afdb_y",
            "delta_from_afdb_z",
            "centroid_displacement_from_afdb",
            "n_residues",
        ],
    )

    metadata = graph.get("metadata", {})
    manifest_row = metadata.get("manifest_row", {}) if isinstance(metadata, dict) else {}
    summary = {
        "sample_id": sample_id,
        "family": manifest_row.get("family", ""),
        "uniprot_id": manifest_row.get("uniprot_id", ""),
        "pdb_id": manifest_row.get("pdb_id", ""),
        "pdb_chain": manifest_row.get("pdb_chain", ""),
        "state_label": manifest_row.get("state_label", ""),
        "substrate_class": manifest_row.get("substrate_class", ""),
        "eval_raw_afdb_rmsd": case_metrics["eval"]["raw_afdb_rmsd"],
        "eval_method_rmsd": case_metrics["eval"]["method_rmsd"],
        "eval_delta_rmsd": case_metrics["eval"]["delta_rmsd_method_minus_raw"],
        "plug_raw_afdb_rmsd": case_metrics["plug"]["raw_afdb_rmsd"],
        "plug_method_rmsd": case_metrics["plug"]["method_rmsd"],
        "plug_delta_rmsd": case_metrics["plug"]["delta_rmsd_method_minus_raw"],
        "barrel_core_delta_rmsd": case_metrics["barrel_core"]["delta_rmsd_method_minus_raw"],
        "graph_path": str(graph_path),
        "prediction_path": str(prediction_path),
        "afdb_aligned_ca_pdb": str(afdb_pdb),
        "experimental_target_aligned_ca_pdb": str(target_pdb),
        "cooper_seed404_prediction_aligned_ca_pdb": str(pred_pdb),
        "region_centroids_csv": str(centroid_csv),
    }
    summary_csv = out_dir / f"{prefix}_summary.csv"
    write_csv_rows(summary_csv, [summary], list(summary.keys()))

    region_rows = []
    for region, mask in masks.items():
        region_rows.append({"sample_id": sample_id, "region": region, "n_residues": int(mask.sum())})
    write_csv_rows(out_dir / f"{prefix}_region_counts.csv", region_rows, ["sample_id", "region", "n_residues"])

    return {
        "summary": summary,
        "summary_csv": summary_csv,
        "centroid_csv": centroid_csv,
        "afdb_pdb": afdb_pdb,
        "target_pdb": target_pdb,
        "prediction_pdb": pred_pdb,
    }


def _box(ax: object, xy: tuple[float, float], w: float, h: float, text: str, *, fc: str, ec: str, size: float = 7.4) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size)


def _arrow(ax: object, start: tuple[float, float], end: tuple[float, float], color: str, *, lw: float = 1.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color))


def _plot_task_contract(ax: object, style: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Task: local C-alpha displacement")
    _box(ax, (0.05, 0.56), 0.25, 0.22, "AFDB-v6\nstarting state\nx_AFDB", fc=style.palette["muted_bg"], ec=style.palette["raw"])
    _box(
        ax,
        (0.39, 0.68),
        0.25,
        0.18,
        "Experimental\ntarget state\nx_target",
        fc=style.palette["glacier"],
        ec=style.palette["gold"],
    )
    _box(
        ax,
        (0.39, 0.38),
        0.25,
        0.18,
        "Cooper-TBDT\nprediction\nx_AFDB + y_hat",
        fc=style.palette["celadon"],
        ec=style.palette["primary"],
    )
    _box(
        ax,
        (0.72, 0.53),
        0.23,
        0.20,
        "Evaluate by region\nRMSD(y_hat, y_delta)\nnot full-chain RMSD",
        fc="#FFFFFF",
        ec=style.palette["grid"],
        size=7.0,
    )
    _arrow(ax, (0.30, 0.67), (0.39, 0.76), style.palette["gold"], lw=1.2)
    _arrow(ax, (0.30, 0.62), (0.39, 0.47), style.palette["primary"], lw=1.2)
    _arrow(ax, (0.64, 0.76), (0.72, 0.64), style.palette["reference"])
    _arrow(ax, (0.64, 0.47), (0.72, 0.59), style.palette["reference"])
    ax.text(
        0.06,
        0.94,
        "target displacement:\ny_delta = x_target - x_AFDB",
        ha="left",
        va="top",
        fontsize=6.9,
    )
    ax.text(0.20, 0.25, "Raw AFDB baseline: y_hat = 0", ha="center", va="center", fontsize=7.6)


def _plot_region_schematic(ax: object, style: Any) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Region masks define endpoints")
    barrel = FancyBboxPatch(
        (0.27, 0.20),
        0.46,
        0.62,
        boxstyle="round,pad=0.03,rounding_size=0.20",
        facecolor=style.palette["muted_bg"],
        edgecolor=style.palette["baseline"],
        linewidth=2.0,
    )
    ax.add_patch(barrel)
    ax.add_patch(Rectangle((0.32, 0.30), 0.36, 0.42, facecolor="white", edgecolor=style.palette["grid"], linewidth=1.0))
    ax.add_patch(Circle((0.50, 0.52), 0.12, facecolor=style.palette["primary"], edgecolor="white", linewidth=1.0, alpha=0.9))
    ax.add_patch(Rectangle((0.46, 0.12), 0.08, 0.14, facecolor=style.palette["worsened"], edgecolor="white", linewidth=1.0))
    for x in (0.34, 0.43, 0.57, 0.66):
        ax.add_patch(FancyArrowPatch((x, 0.82), (x + 0.02, 0.93), arrowstyle="-", linewidth=4.0, color=style.palette["blend"]))
    ax.scatter([0.39, 0.61, 0.50], [0.70, 0.70, 0.75], s=48, color=style.palette["gold"], edgecolor="white", linewidth=0.8)
    ax.text(0.50, 0.92, "extracellular loops", ha="center", va="bottom", fontsize=7.0, color=style.palette["blend"])
    ax.text(0.50, 0.52, "plug", ha="center", va="center", fontsize=7.2, color="white")
    ax.text(0.50, 0.09, "TonB box", ha="center", va="top", fontsize=7.0, color=style.palette["worsened"])
    ax.text(0.08, 0.54, "barrel core\nscaffold frame", ha="left", va="center", fontsize=7.2, color=style.palette["baseline"])
    ax.text(0.78, 0.72, "substrate-contact /\nevaluation region", ha="left", va="center", fontsize=7.0, color=style.palette["gold"])
    ax.plot([0.18, 0.30], [0.54, 0.54], color=style.palette["baseline"], linewidth=1.0)
    ax.plot([0.74, 0.61], [0.72, 0.70], color=style.palette["gold"], linewidth=1.0)


def _plot_case_placeholder(ax: object, exports: dict[str, Any], style: Any) -> None:
    summary = exports["summary"]
    ax.axis("off")
    ax.set_title("Positive coordinate case placeholder")
    ax.add_patch(Rectangle((0.04, 0.08), 0.92, 0.80, fill=False, edgecolor=style.palette["grid"], linewidth=1.1))
    text = (
        "Draw AFDB / experimental / prediction overlay here\n\n"
        f"case: {summary['sample_id']} ({summary['family']}, PDB {summary['pdb_id']}{summary['pdb_chain']})\n"
        f"state/substrate: {summary['state_label']} / {summary['substrate_class']}\n"
        f"evaluation RMSD: {summary['eval_raw_afdb_rmsd']:.3f} -> {summary['eval_method_rmsd']:.3f} A "
        f"({summary['eval_delta_rmsd']:+.3f})\n"
        f"plug RMSD: {summary['plug_raw_afdb_rmsd']:.3f} -> {summary['plug_method_rmsd']:.3f} A "
        f"({summary['plug_delta_rmsd']:+.3f})\n"
        f"barrel-core Delta RMSD: {summary['barrel_core_delta_rmsd']:+.4f} A\n\n"
        "Exported CA overlays:\n"
        f"{Path(summary['afdb_aligned_ca_pdb']).name}\n"
        f"{Path(summary['experimental_target_aligned_ca_pdb']).name}\n"
        f"{Path(summary['cooper_seed404_prediction_aligned_ca_pdb']).name}"
    )
    ax.text(0.08, 0.82, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=style.palette["text"])


def _plot_case_bars(ax: object, case_metrics: dict[str, dict[str, Any]], style: Any) -> None:
    regions = ("eval", "plug", "barrel_core")
    labels = [REGION_LABELS[region] for region in regions]
    raw = np.array([float(case_metrics[region]["raw_afdb_rmsd"]) for region in regions], dtype=float)
    method = np.array([float(case_metrics[region]["method_rmsd"]) for region in regions], dtype=float)
    x = np.arange(len(regions), dtype=float)
    width = 0.34
    ax.bar(x - width / 2, raw, width=width, color=style.palette["raw"], edgecolor="white", linewidth=0.8, label="Raw AFDB")
    ax.bar(
        x + width / 2,
        method,
        width=width,
        color=style.palette["primary"],
        edgecolor="white",
        linewidth=0.8,
        label="Cooper-TBDT seed404",
    )
    for xi, value in zip(x - width / 2, raw):
        ax.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.6)
    for xi, value in zip(x + width / 2, method):
        ax.text(xi, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Case-level coordinate improvement")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(raw.max(), method.max()) * 1.25)
    clean_axis(ax, style)


def _write_case_metrics(out_dir: Path, case_metrics: dict[str, dict[str, Any]]) -> None:
    write_csv_rows(
        out_dir / "task_definition_positive_case_metric_values.csv",
        case_metrics.values(),
        ["sample_id", "region", "raw_afdb_rmsd", "method_rmsd", "delta_rmsd_method_minus_raw", "improved", "source"],
    )


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    out_dir = figure_output_dir(args.out_dir, args.out_name)
    case_metrics = _load_case_metrics(args.paired_delta_csv, args.case_sample_id)
    _write_case_metrics(out_dir, case_metrics)
    exports = _export_case_coordinates(args, out_dir, case_metrics)

    fig = plt.figure(figsize=(8.6, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    _plot_task_contract(ax_a, style)
    _plot_region_schematic(ax_b, style)
    _plot_case_placeholder(ax_c, exports, style)
    _plot_case_bars(ax_d, case_metrics, style)
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
