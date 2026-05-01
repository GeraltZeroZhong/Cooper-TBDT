"""Build the TonB-box mechanism-boundary figure and case-study exports."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


STATE_ORDER = ("buried_like", "exposed_like", "unchanged")
STATE_LABELS = {
    "buried_like": "buried-like",
    "exposed_like": "exposed-like",
    "unchanged": "unchanged",
}
STATE_COLORS = {
    "buried_like": "peacock",
    "exposed_like": "instagram",
    "unchanged": "celadon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Cooper-TBDT TonB mechanism-boundary figure")
    parser.add_argument(
        "--tonb-state-csv",
        default="artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_tonb_state_metrics.csv",
        help="TonB state metrics CSV from eval_tbdt_state.",
    )
    parser.add_argument(
        "--graph-dir",
        default="data/processed_tbdt_gold_graphs",
        help="Processed graph directory used for case-study coordinate export.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test",
        help="Prediction directory used for case-study coordinate export.",
    )
    parser.add_argument(
        "--case-sample-id",
        default="btub_p06129_2gsk_a",
        help="Representative TonB case study sample ID.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory. The figure creates its own subdirectory.",
    )
    parser.add_argument("--out-name", default="tonb_mechanistic_boundary", help="Output filename stem.")
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument("--formats", default="png,svg", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    return parser.parse_args()


def _load_tonb_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            continue
        out.append(
            {
                "sample_id": sample_id,
                "tonb_n_residues": int(finite_float(row.get("tonb_n_residues"), field="tonb_n_residues", source=str(path))),
                "target_centroid_displacement": finite_float(
                    row.get("tonb_target_centroid_displacement"),
                    field="tonb_target_centroid_displacement",
                    source=str(path),
                ),
                "predicted_centroid_displacement": finite_float(
                    row.get("tonb_predicted_centroid_displacement"),
                    field="tonb_predicted_centroid_displacement",
                    source=str(path),
                ),
                "centroid_displacement_error": finite_float(
                    row.get("tonb_centroid_displacement_error"),
                    field="tonb_centroid_displacement_error",
                    source=str(path),
                ),
                "centroid_displacement_cosine": finite_float(
                    row.get("tonb_centroid_displacement_cosine"),
                    field="tonb_centroid_displacement_cosine",
                    source=str(path),
                ),
                "direction_compatible": str(row.get("tonb_direction_compatible", "")).lower() == "true",
                "target_exposure_delta": finite_float(
                    row.get("tonb_target_exposure_delta"),
                    field="tonb_target_exposure_delta",
                    source=str(path),
                ),
                "predicted_exposure_delta": finite_float(
                    row.get("tonb_predicted_exposure_delta"),
                    field="tonb_predicted_exposure_delta",
                    source=str(path),
                ),
                "target_state": row.get("tonb_target_state", "unknown") or "unknown",
                "predicted_state": row.get("tonb_predicted_state", "unknown") or "unknown",
                "state_correct": str(row.get("tonb_state_correct", "")).lower() == "true",
                "target_axis_cosine_vs_af2": finite_float(
                    row.get("tonb_target_axis_cosine_vs_af2"),
                    field="tonb_target_axis_cosine_vs_af2",
                    source=str(path),
                ),
                "predicted_axis_cosine_vs_target": finite_float(
                    row.get("tonb_predicted_axis_cosine_vs_target"),
                    field="tonb_predicted_axis_cosine_vs_target",
                    source=str(path),
                ),
                "source": str(path),
            }
        )
    if not out:
        raise ValueError(f"No TonB rows found in {path}")
    return out


def _case_row(rows: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    for row in rows:
        if row["sample_id"] == sample_id:
            return row
    raise ValueError(f"Case sample {sample_id!r} not found in TonB state rows")


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


def _write_ca_pdb(path: Path, residue_ids: list[str], coords: np.ndarray, *, tonb_mask: np.ndarray) -> None:
    lines: list[str] = []
    for serial, (residue_id, xyz, is_tonb) in enumerate(zip(residue_ids, coords, tonb_mask), start=1):
        chain, resseq = _parse_residue_id(str(residue_id))
        atom_name = "CA"
        residue_name = "TON" if bool(is_tonb) else "GLY"
        bfactor = 100.0 if bool(is_tonb) else 20.0
        lines.append(
            f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain:1s}{resseq:4d}    "
            f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}"
            f"  1.00{bfactor:6.2f}           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _export_case_data(args: argparse.Namespace, out_dir: Path, case: dict[str, Any]) -> dict[str, Path]:
    import torch

    sample_id = str(case["sample_id"])
    graph_path = Path(args.graph_dir) / f"{sample_id}.pt"
    pred_path = Path(args.predictions_dir) / f"{sample_id}.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Case graph not found: {graph_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Case prediction not found: {pred_path}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    pos = graph["pos"].detach().cpu().numpy()
    target = pos + graph["y_delta"].detach().cpu().numpy()
    pred_pos = pos + pred["pred_delta"].detach().cpu().numpy()
    pred_delta = pred["pred_delta"].detach().cpu().numpy()
    target_delta = graph["y_delta"].detach().cpu().numpy()
    residue_ids = [str(item) for item in graph["residue_ids"]]
    tonb_mask = graph["tonb_box_mask"].detach().cpu().numpy().astype(bool)
    plddt = graph.get("plddt")
    plddt_values = plddt.detach().cpu().numpy().reshape(-1) if plddt is not None else np.full(len(residue_ids), np.nan)

    case_prefix = f"case_{sample_id}"
    afdb_pdb = out_dir / f"{case_prefix}_afdb_aligned_ca.pdb"
    target_pdb = out_dir / f"{case_prefix}_experimental_target_aligned_ca.pdb"
    pred_pdb = out_dir / f"{case_prefix}_cooper_prediction_aligned_ca.pdb"
    _write_ca_pdb(afdb_pdb, residue_ids, pos, tonb_mask=tonb_mask)
    _write_ca_pdb(target_pdb, residue_ids, target, tonb_mask=tonb_mask)
    _write_ca_pdb(pred_pdb, residue_ids, pred_pos, tonb_mask=tonb_mask)

    tonb_indices = np.flatnonzero(tonb_mask)
    coord_rows: list[dict[str, Any]] = []
    for idx in tonb_indices:
        coord_rows.append(
            {
                "sample_id": sample_id,
                "node_index": int(idx),
                "residue_id": residue_ids[int(idx)],
                "plddt": float(plddt_values[int(idx)]),
                "afdb_x": float(pos[idx, 0]),
                "afdb_y": float(pos[idx, 1]),
                "afdb_z": float(pos[idx, 2]),
                "target_x": float(target[idx, 0]),
                "target_y": float(target[idx, 1]),
                "target_z": float(target[idx, 2]),
                "prediction_x": float(pred_pos[idx, 0]),
                "prediction_y": float(pred_pos[idx, 1]),
                "prediction_z": float(pred_pos[idx, 2]),
                "target_delta_x": float(target_delta[idx, 0]),
                "target_delta_y": float(target_delta[idx, 1]),
                "target_delta_z": float(target_delta[idx, 2]),
                "prediction_delta_x": float(pred_delta[idx, 0]),
                "prediction_delta_y": float(pred_delta[idx, 1]),
                "prediction_delta_z": float(pred_delta[idx, 2]),
            }
        )

    coord_csv = out_dir / f"{case_prefix}_tonb_ca_coordinates.csv"
    write_csv_rows(
        coord_csv,
        coord_rows,
        [
            "sample_id",
            "node_index",
            "residue_id",
            "plddt",
            "afdb_x",
            "afdb_y",
            "afdb_z",
            "target_x",
            "target_y",
            "target_z",
            "prediction_x",
            "prediction_y",
            "prediction_z",
            "target_delta_x",
            "target_delta_y",
            "target_delta_z",
            "prediction_delta_x",
            "prediction_delta_y",
            "prediction_delta_z",
        ],
    )

    centroids = {
        "afdb": pos[tonb_mask].mean(axis=0),
        "experimental_target": target[tonb_mask].mean(axis=0),
        "cooper_prediction": pred_pos[tonb_mask].mean(axis=0),
    }
    afdb_centroid = centroids["afdb"]
    centroid_rows = []
    for name, centroid in centroids.items():
        vector = centroid - afdb_centroid
        centroid_rows.append(
            {
                "sample_id": sample_id,
                "structure": name,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "delta_from_afdb_x": float(vector[0]),
                "delta_from_afdb_y": float(vector[1]),
                "delta_from_afdb_z": float(vector[2]),
                "centroid_displacement_from_afdb": float(np.linalg.norm(vector)),
            }
        )
    centroid_csv = out_dir / f"{case_prefix}_centroids.csv"
    write_csv_rows(
        centroid_csv,
        centroid_rows,
        [
            "sample_id",
            "structure",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "delta_from_afdb_x",
            "delta_from_afdb_y",
            "delta_from_afdb_z",
            "centroid_displacement_from_afdb",
        ],
    )

    summary_csv = out_dir / f"{case_prefix}_summary.csv"
    write_csv_rows(
        summary_csv,
        [
            {
                **case,
                "graph_path": str(graph_path),
                "prediction_path": str(pred_path),
                "afdb_aligned_ca_pdb": str(afdb_pdb),
                "experimental_target_aligned_ca_pdb": str(target_pdb),
                "cooper_prediction_aligned_ca_pdb": str(pred_pdb),
                "tonb_ca_coordinates_csv": str(coord_csv),
                "centroids_csv": str(centroid_csv),
            }
        ],
        [
            "sample_id",
            "tonb_n_residues",
            "target_state",
            "predicted_state",
            "target_centroid_displacement",
            "predicted_centroid_displacement",
            "centroid_displacement_error",
            "centroid_displacement_cosine",
            "direction_compatible",
            "target_exposure_delta",
            "predicted_exposure_delta",
            "state_correct",
            "graph_path",
            "prediction_path",
            "afdb_aligned_ca_pdb",
            "experimental_target_aligned_ca_pdb",
            "cooper_prediction_aligned_ca_pdb",
            "tonb_ca_coordinates_csv",
            "centroids_csv",
        ],
    )

    return {
        "afdb_pdb": afdb_pdb,
        "target_pdb": target_pdb,
        "prediction_pdb": pred_pdb,
        "coord_csv": coord_csv,
        "centroid_csv": centroid_csv,
        "summary_csv": summary_csv,
    }


def _plot_case_placeholder(ax: object, case: dict[str, Any], exports: dict[str, Path], style: Any) -> None:
    ax.axis("off")
    ax.set_title("TonB case-study placeholder")
    ax.add_patch(Rectangle((0.03, 0.07), 0.94, 0.82, fill=False, edgecolor=style.palette["grid"], linewidth=1.1))
    lines = [
        "Draw structural overlay here",
        "",
        f"case: {case['sample_id']}",
        f"target state: {STATE_LABELS.get(str(case['target_state']), case['target_state'])}",
        f"predicted state: {STATE_LABELS.get(str(case['predicted_state']), case['predicted_state'])}",
        f"target centroid shift: {case['target_centroid_displacement']:.2f} A",
        f"predicted centroid shift: {case['predicted_centroid_displacement']:.2f} A",
        "",
        "Exported data:",
        exports["coord_csv"].name,
        exports["centroid_csv"].name,
        exports["afdb_pdb"].name,
        exports["target_pdb"].name,
        exports["prediction_pdb"].name,
    ]
    ax.text(
        0.08,
        0.82,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=style.palette["text"],
    )


def _plot_centroid_comparison(ax: object, rows: list[dict[str, Any]], case_id: str, style: Any) -> None:
    selected = sorted(rows, key=lambda row: float(row["target_centroid_displacement"]), reverse=True)
    y = np.arange(len(selected), dtype=float)
    target = np.array([float(row["target_centroid_displacement"]) for row in selected], dtype=float)
    predicted = np.array([float(row["predicted_centroid_displacement"]) for row in selected], dtype=float)
    labels = [str(row["sample_id"]).replace("btub_p06129_", "BtuB ") for row in selected]

    for yi, x0, x1, row in zip(y, predicted, target, selected):
        color = style.palette["blossom"] if row["sample_id"] == case_id else style.palette["grid"]
        ax.plot([x0, x1], [yi, yi], color=color, linewidth=1.0, zorder=1)
    ax.scatter(target, y, s=25, color=style.palette["instagram"], label="target", zorder=3)
    ax.scatter(predicted, y, s=25, color=style.palette["primary"], label="prediction", zorder=3)
    for yi, row in zip(y, selected):
        if row["sample_id"] == case_id:
            ax.scatter(
                [float(row["target_centroid_displacement"])],
                [yi],
                s=55,
                facecolors="none",
                edgecolors=style.palette["instagram"],
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                float(row["target_centroid_displacement"]) + 0.55,
                yi,
                "case",
                ha="left",
                va="center",
                fontsize=6.7,
                color=style.palette["instagram"],
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("TonB centroid displacement from AFDB (Å)")
    ax.set_title("Target vs predicted centroid displacement")
    ax.set_xlim(0.0, max(target) + 2.3)
    ax.legend(loc="lower right")
    clean_axis(ax, style, grid_axis="x")


def _plot_state_stacks(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    target_counts = Counter(str(row["target_state"]) for row in rows)
    predicted_counts = Counter(str(row["predicted_state"]) for row in rows)
    totals = {"experimental": sum(target_counts.values()), "predicted": sum(predicted_counts.values())}
    y_positions = {"experimental": 0.0, "predicted": 1.0}
    for label, counts in (("experimental", target_counts), ("predicted", predicted_counts)):
        left = 0
        for state in STATE_ORDER:
            count = counts.get(state, 0)
            if count <= 0:
                continue
            ax.barh(
                y_positions[label],
                count,
                left=left,
                height=0.54,
                color=style.palette[STATE_COLORS[state]],
                edgecolor="white",
                linewidth=0.8,
                label=STATE_LABELS[state] if label == "experimental" else None,
            )
            ax.text(left + count / 2, y_positions[label], str(count), ha="center", va="center", fontsize=7.0, color="white")
            left += count
        ax.text(totals[label] + 0.35, y_positions[label], f"n={totals[label]}", ha="left", va="center", fontsize=7.0)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Experimental", "Predicted"])
    ax.invert_yaxis()
    ax.set_xlabel("Targets")
    ax.set_title("Exposure-state confusion")
    ax.set_xlim(0.0, max(totals.values()) + 2.2)
    ax.legend(loc="lower right", ncol=1)
    clean_axis(ax, style, grid_axis="x")


def _plot_direction_stats(ax: object, rows: list[dict[str, Any]], style: Any) -> None:
    cosines = np.array([float(row["centroid_displacement_cosine"]) for row in rows], dtype=float)
    compatible = np.array([bool(row["direction_compatible"]) for row in rows], dtype=bool)
    rng = np.random.default_rng(123)
    jitter = rng.normal(0.0, 0.025, size=len(cosines))
    ax.scatter(
        cosines[~compatible],
        np.full(int((~compatible).sum()), 0.0) + jitter[~compatible],
        s=24,
        color=style.palette["sakura"],
        alpha=0.88,
        label="not compatible",
    )
    ax.scatter(
        cosines[compatible],
        np.full(int(compatible.sum()), 0.0) + jitter[compatible],
        s=34,
        color=style.palette["primary"],
        label="direction-compatible",
        zorder=3,
    )
    median = float(np.median(cosines))
    mean = float(np.mean(cosines))
    ax.axvline(median, color=style.palette["reference"], linestyle="--", linewidth=0.9)
    ax.text(median + 0.03, 0.15, f"median {median:.2f}", ha="left", va="center", fontsize=7.0)
    rate = float(compatible.mean())
    ax.text(
        0.03,
        0.88,
        f"direction-compatible: {compatible.sum()}/{len(compatible)} ({rate * 100:.0f}%)\nmean cosine: {mean:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.4},
    )
    ax.axvline(0.0, color=style.palette["grid"], linewidth=0.9)
    ax.set_yticks([])
    ax.set_xlabel("TonB centroid displacement cosine")
    ax.set_title("Direction compatibility")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.35, 0.35)
    ax.legend(loc="lower right")
    clean_axis(ax, style, grid_axis="x")


def _write_tonb_plotting_data(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        out_dir / "tonb_state_values.csv",
        rows,
        [
            "sample_id",
            "tonb_n_residues",
            "target_centroid_displacement",
            "predicted_centroid_displacement",
            "centroid_displacement_error",
            "centroid_displacement_cosine",
            "direction_compatible",
            "target_exposure_delta",
            "predicted_exposure_delta",
            "target_state",
            "predicted_state",
            "state_correct",
            "target_axis_cosine_vs_af2",
            "predicted_axis_cosine_vs_target",
            "source",
        ],
    )
    target_counts = Counter(str(row["target_state"]) for row in rows)
    predicted_counts = Counter(str(row["predicted_state"]) for row in rows)
    summary_rows: list[dict[str, Any]] = []
    for source, counts in (("experimental", target_counts), ("predicted", predicted_counts)):
        for state in STATE_ORDER:
            summary_rows.append({"state_source": source, "state": state, "count": counts.get(state, 0)})
    summary_rows.append(
        {
            "state_source": "direction_compatible",
            "state": "true",
            "count": sum(1 for row in rows if row["direction_compatible"]),
        }
    )
    summary_rows.append(
        {
            "state_source": "direction_compatible",
            "state": "false",
            "count": sum(1 for row in rows if not row["direction_compatible"]),
        }
    )
    write_csv_rows(out_dir / "tonb_state_summary_values.csv", summary_rows, ["state_source", "state", "count"])


def build_figure(args: argparse.Namespace) -> list[Path]:
    style = get_style(args.style)
    apply_style(style)

    tonb_rows = _load_tonb_rows(args.tonb_state_csv)
    case = _case_row(tonb_rows, args.case_sample_id)
    out_dir = figure_output_dir(args.out_dir, args.out_name)
    _write_tonb_plotting_data(out_dir, tonb_rows)
    exports = _export_case_data(args, out_dir, case)

    fig = plt.figure(figsize=(10.2, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_case_placeholder(ax_a, case, exports, style)
    _plot_centroid_comparison(ax_b, tonb_rows, str(case["sample_id"]), style)
    _plot_state_stacks(ax_c, tonb_rows, style)
    _plot_direction_stats(ax_d, tonb_rows, style)

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
