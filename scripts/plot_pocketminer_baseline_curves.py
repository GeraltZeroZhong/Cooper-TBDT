#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PAPER_SCORES = {
    "PocketMiner": {
        "column_candidates": ["PocketMiner Predcitions", "PocketMiner Predictions"],
    },
    "CryptoSite": {
        "column_candidates": ["CryptoSite Predictions", "CryptoSite Prediction"],
    },
}

METHOD_STYLES = {
    "PocketMiner": {"color": "#d35400", "linewidth": 3.0, "linestyle": "-"},
    "CryptoSite": {"color": "#16a085", "linewidth": 2.4, "linestyle": "-"},
    "HoloShift |dCA|": {"color": "#8e44ad", "linewidth": 2.4, "linestyle": "-"},
    "HoloShift dRMSD": {"color": "#2c3e50", "linewidth": 2.4, "linestyle": "--"},
    "HoloShift 2x dRMSD": {"color": "#7f8c8d", "linewidth": 2.2, "linestyle": "-."},
}

HOLOSHIFT_SCORE_SPECS = [
    {
        "structure": "holoshift_unrelaxed",
        "score_column": "motion_norm_ca_A",
        "method": "HoloShift |dCA|",
        "score_kind": "deployable_motion_magnitude",
        "note": "CA displacement magnitude from apo to HoloShift; no holo coordinates are used in the score.",
        "requires_holo": False,
    },
    {
        "structure": "holoshift_unrelaxed",
        "score_column": "delta_distance_to_holo_A",
        "method": "HoloShift dRMSD",
        "score_kind": "holo_oracle_diagnostic",
        "note": "Apo-to-holo CA distance minus HoloShift-to-holo CA distance; positive means closer to holo.",
        "requires_holo": True,
    },
    {
        "structure": "holoshift_scale_2p00",
        "score_column": "delta_distance_to_holo_A",
        "method": "HoloShift 2x dRMSD",
        "score_kind": "holo_oracle_diagnostic",
        "note": "Same diagnostic score after applying the 2x HoloShift scale ensemble member.",
        "requires_holo": True,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot ROC/PR curves for PocketMiner baselines and aligned HoloShift residue scores."
    )
    p.add_argument(
        "--source-data-xlsx",
        type=Path,
        default=Path("outputs/pocketminer_holoshift/external/pocketminer_source_data.xlsx"),
        help="PocketMiner Nature Communications source-data workbook.",
    )
    p.add_argument("--sheet-name", default="Figure 5c,d")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift/curves"))
    p.add_argument("--manifest", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_cryptic_manifest.csv"))
    p.add_argument("--labels-csv", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_residue_labels.csv"))
    p.add_argument(
        "--holoshift-prior-dir",
        type=Path,
        default=Path("outputs/pocketminer_holoshift/holoshift_prior_full"),
        help="Directory produced by scripts/run_pocketminer_holoshift_prior.py.",
    )
    p.add_argument("--include-holoshift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--title-prefix",
        default="PocketMiner Cryptic-Pocket Benchmark",
        help="Prefix used in baseline-only figure titles.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns were found: {candidates}; columns={list(df.columns)}")


def load_paper_scores(path: Path, sheet_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PocketMiner source-data workbook: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    true_col = require_column(df, ["True Value", "true_value", "label"])
    score_cols = {method: require_column(df, cfg["column_candidates"]) for method, cfg in PAPER_SCORES.items()}
    rows = df[[true_col, *score_cols.values()]].dropna().copy()
    y_true = rows[true_col].astype(int).to_numpy()
    scores = {method: rows[col].astype(float).to_numpy() for method, col in score_cols.items()}
    return y_true, scores


def parse_atoms_for_chain(path: Path, chain_id: str | None):
    from run_binding_readiness_benchmark import parse_pdb_atoms

    atoms = parse_pdb_atoms(path, chain_id=chain_id or None, atom_records_only=True)
    if not atoms and chain_id:
        atoms = parse_pdb_atoms(path, chain_id=None, atom_records_only=True)
    return atoms


def eval_labels_by_target(labels_csv: Path) -> dict[str, list[dict[str, int]]]:
    labels: dict[str, list[dict[str, int]]] = {}
    for row in read_csv(labels_csv):
        if row.get("is_eval") != "1" or row.get("label") not in {"0", "1"}:
            continue
        labels.setdefault(row["target_id"], []).append(
            {"residue_index": int(row["residue_index"]), "label": int(row["label"])}
        )
    for target_rows in labels.values():
        target_rows.sort(key=lambda item: item["residue_index"])
    return labels


def holoshift_variant_path(prior_dir: Path, target_id: str, structure: str) -> Path:
    if structure == "holoshift_unrelaxed":
        return prior_dir / "targets" / target_id / f"{target_id}_holoshift_unrelaxed.pdb"
    return prior_dir / "targets" / target_id / f"{structure}.pdb"


def finite_or_blank(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def build_holoshift_residue_scores(
    *,
    manifest: Path,
    labels_csv: Path,
    prior_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from run_pocketminer_holoshift_prior import apply_transform, ca_coords_by_order, kabsch_fit

    labels = eval_labels_by_target(labels_csv)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for manifest_row in read_csv(manifest):
        target_id = manifest_row.get("target_id", "")
        if target_id not in labels or not manifest_row.get("apo_pdb"):
            continue
        try:
            apo_atoms = parse_atoms_for_chain(Path(manifest_row["apo_pdb"]), manifest_row.get("apo_chain") or None)
            _apo_keys, apo_ca = ca_coords_by_order(apo_atoms)
            has_holo = (
                manifest_row.get("has_cryptic_holo") == "1"
                and bool(manifest_row.get("holo_pdb"))
                and Path(manifest_row["holo_pdb"]).exists()
            )
            holo_ca = np.empty((0, 3), dtype=float)
            if has_holo:
                holo_atoms = parse_atoms_for_chain(Path(manifest_row["holo_pdb"]), manifest_row.get("holo_chain") or None)
                _holo_keys, holo_ca = ca_coords_by_order(holo_atoms)
            n_align = min(len(apo_ca), len(holo_ca)) if has_holo else len(apo_ca)
            if n_align == 0:
                raise ValueError("No CA atoms available for HoloShift scoring.")
            if has_holo:
                rot, trans = kabsch_fit(apo_ca[:n_align], holo_ca[:n_align])
            else:
                rot, trans = np.eye(3), np.zeros(3)
            apo_aligned = apply_transform(apo_ca[:n_align], rot, trans)
            holo_common = holo_ca[:n_align] if has_holo else np.full((n_align, 3), np.nan, dtype=float)

            for spec in HOLOSHIFT_SCORE_SPECS:
                if spec.get("requires_holo") and not has_holo:
                    continue
                pdb_path = holoshift_variant_path(prior_dir, target_id, spec["structure"])
                if not pdb_path.exists():
                    failures.append(
                        {
                            "target_id": target_id,
                            "structure": spec["structure"],
                            "stage": "load_variant",
                            "error": f"missing PDB: {pdb_path}",
                        }
                    )
                    continue
                variant_atoms = parse_atoms_for_chain(pdb_path, manifest_row.get("apo_chain") or None)
                _variant_keys, variant_ca = ca_coords_by_order(variant_atoms)
                n = min(n_align, len(variant_ca))
                if n == 0:
                    failures.append(
                        {
                            "target_id": target_id,
                            "structure": spec["structure"],
                            "stage": "load_variant",
                            "error": "No CA atoms in variant.",
                        }
                    )
                    continue
                variant_aligned = apply_transform(variant_ca[:n], rot, trans)
                apo_n = apo_aligned[:n]
                holo_n = holo_common[:n]
                pred_vec = variant_aligned - apo_n
                motion_norm = np.linalg.norm(pred_vec, axis=1)
                if has_holo:
                    true_vec = holo_n - apo_n
                    apo_to_holo = np.linalg.norm(true_vec, axis=1)
                    variant_to_holo = np.linalg.norm(variant_aligned - holo_n, axis=1)
                    delta_to_holo = apo_to_holo - variant_to_holo
                    true_norm = np.linalg.norm(true_vec, axis=1)
                else:
                    true_vec = np.full_like(pred_vec, np.nan)
                    apo_to_holo = np.full(n, np.nan, dtype=float)
                    variant_to_holo = np.full(n, np.nan, dtype=float)
                    delta_to_holo = np.full(n, np.nan, dtype=float)
                    true_norm = np.full(n, np.nan, dtype=float)
                projection = np.full(n, np.nan, dtype=float)
                cosine = np.full(n, np.nan, dtype=float)
                ok = np.logical_and(np.isfinite(true_norm), np.logical_and(true_norm > 1e-8, motion_norm > 1e-8))
                projection[ok] = np.sum(pred_vec[ok] * true_vec[ok], axis=1) / true_norm[ok]
                cosine[ok] = np.sum(pred_vec[ok] * true_vec[ok], axis=1) / (true_norm[ok] * motion_norm[ok])
                score_arrays = {
                    "motion_norm_ca_A": motion_norm,
                    "delta_distance_to_holo_A": delta_to_holo,
                    "projection_to_holo_A": projection,
                    "motion_cosine": cosine,
                }
                for label_row in labels[target_id]:
                    idx = label_row["residue_index"]
                    if idx >= n:
                        continue
                    rows.append(
                        {
                            "target_id": target_id,
                            "split": manifest_row.get("split", ""),
                            "structure": spec["structure"],
                            "method": spec["method"],
                            "score_kind": spec["score_kind"],
                            "residue_index": idx,
                            "label": label_row["label"],
                            "score": finite_or_blank(score_arrays[spec["score_column"]][idx]),
                            "motion_norm_ca_A": finite_or_blank(motion_norm[idx]),
                            "delta_distance_to_holo_A": finite_or_blank(delta_to_holo[idx]),
                            "projection_to_holo_A": finite_or_blank(projection[idx]),
                            "motion_cosine": finite_or_blank(cosine[idx]),
                            "apo_distance_to_holo_A": finite_or_blank(apo_to_holo[idx]),
                            "variant_distance_to_holo_A": finite_or_blank(variant_to_holo[idx]),
                            "note": spec["note"],
                        }
                    )
        except Exception as exc:
            failures.append({"target_id": target_id, "stage": "holoshift_residue_scores", "error": str(exc)})
    return rows, failures


def max_f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    denom = np.maximum(precision + recall, 1e-12)
    return float(np.max(2.0 * precision * recall / denom))


def method_curves(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "max_f1": max_f1_from_pr(precision, recall),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def baseline_metrics_rows(
    y_true: np.ndarray,
    curves: dict[str, dict[str, Any]],
    source: Path,
    sheet_name: str,
) -> list[dict[str, Any]]:
    positives = int(np.sum(y_true))
    rows: list[dict[str, Any]] = []
    for method, payload in curves.items():
        rows.append(
            {
                "method": method,
                "status": "available",
                "n": int(len(y_true)),
                "n_positive": positives,
                "n_negative": int(len(y_true) - positives),
                "positive_prevalence": positives / len(y_true),
                "roc_auc": payload["roc_auc"],
                "average_precision": payload["average_precision"],
                "max_f1": payload["max_f1"],
                "source_data_xlsx": str(source),
                "sheet_name": sheet_name,
                "score_source": "pocketminer_source_data",
                "score_kind": "published_residue_score",
                "note": "Published source-data scores are pooled and do not contain target/residue IDs.",
            }
        )
    return rows


def sesame_unavailable_row() -> dict[str, Any]:
    return {
        "method": "Sesame",
        "status": "unavailable",
        "n": "",
        "n_positive": "",
        "n_negative": "",
        "positive_prevalence": "",
        "roc_auc": "",
        "average_precision": "",
        "max_f1": "",
        "source_data_xlsx": "",
        "sheet_name": "",
        "score_source": "",
        "score_kind": "",
        "note": "No public implementation or downloadable inference artifacts were found; excluded from numeric curves.",
    }


def holoshift_metric_rows(curves: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in curves.items():
        n = int(payload["n"])
        positives = int(payload["n_positive"])
        rows.append(
            {
                "method": method,
                "status": "available",
                "n": n,
                "n_positive": positives,
                "n_negative": int(payload["n_negative"]),
                "positive_prevalence": positives / n,
                "roc_auc": payload["roc_auc"],
                "average_precision": payload["average_precision"],
                "max_f1": payload["max_f1"],
                "source_data_xlsx": "",
                "sheet_name": "",
                "score_source": payload.get("score_source", ""),
                "score_kind": payload.get("score_kind", ""),
                "note": payload.get("note", ""),
            }
        )
    return rows


def curve_rows(curves: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in curves.items():
        for idx, (x, y) in enumerate(zip(payload["fpr"], payload["tpr"], strict=False)):
            rows.append({"curve": "roc", "method": method, "point_index": idx, "x": float(x), "y": float(y)})
        for idx, (x, y) in enumerate(zip(payload["recall"], payload["precision"], strict=False)):
            rows.append({"curve": "pr", "method": method, "point_index": idx, "x": float(x), "y": float(y)})
    return rows


def curves_from_score_rows(score_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    curves: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        if row.get("score") == "":
            continue
        by_method.setdefault(str(row["method"]), []).append(row)
    for method, rows in by_method.items():
        y_true = np.asarray([int(row["label"]) for row in rows], dtype=int)
        scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
        if len(set(y_true.tolist())) < 2:
            failures.append({"method": method, "error": "Need both positive and negative labels."})
            continue
        payload = method_curves(y_true, scores)
        payload["n"] = int(len(y_true))
        payload["n_positive"] = int(y_true.sum())
        payload["n_negative"] = int(len(y_true) - y_true.sum())
        payload["score_source"] = "holoshift_aligned_residue_scores"
        payload["score_kind"] = rows[0].get("score_kind", "")
        payload["note"] = rows[0].get("note", "")
        curves[method] = payload
    return curves, failures


def availability_payload(source_data_xlsx: Path, sheet_name: str) -> dict[str, Any]:
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "baselines": {
            "PocketMiner": {
                "status": "available",
                "role": "published cryptic-pocket residue predictor baseline",
                "source_data_xlsx": str(source_data_xlsx),
                "sheet_name": sheet_name,
                "code_url": "https://github.com/Mickdub/gvp/tree/pocket_pred",
                "paper_url": "https://www.nature.com/articles/s41467-023-36699-3",
            },
            "CryptoSite": {
                "status": "available",
                "role": "published baseline included in PocketMiner source data",
                "source_data_xlsx": str(source_data_xlsx),
                "sheet_name": sheet_name,
            },
            "HoloShift": {
                "status": "available",
                "role": "aligned residue-level conformational-prior scores on local PocketMiner cryptic targets",
                "score_modes": [spec["method"] for spec in HOLOSHIFT_SCORE_SPECS],
            },
            "Sesame": {
                "status": "unavailable",
                "role": "apo-to-holo/cryptic-pocket opening baseline candidate",
                "paper_url": "https://arxiv.org/abs/2509.05302",
                "openreview_url": "https://openreview.net/forum?id=4kt87NSJrZ",
                "reason": (
                    "The arXiv/OpenReview records do not provide a repository or downloadable inference artifact, "
                    "and GitHub repository/code searches for the paper title and arXiv id returned no matches."
                ),
                "github_searches": [
                    "gh search repos 'Sesame protein pockets'",
                    "gh search code '\"Opening the door to protein pockets\"'",
                    "gh search code '\"2509.05302\" \"Sesame\"'",
                ],
            },
        },
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.linewidth": 1.5,
            "font.family": "sans-serif",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(width=1.3)


def style_for_method(method: str) -> dict[str, Any]:
    return METHOD_STYLES.get(method, {"color": "#34495e", "linewidth": 2.2, "linestyle": "-"})


def label_suffix(payload: dict[str, Any], metric_name: str) -> str:
    suffix = f"{payload[metric_name]:.2f}"
    if payload.get("n") and int(payload["n"]) != 1846:
        suffix += f", n={int(payload['n'])}"
    return suffix


def plot_roc(ax: plt.Axes, curves: dict[str, dict[str, Any]], title: str) -> None:
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, linewidth=1.8, label="Random")
    for method, payload in curves.items():
        cfg = style_for_method(method)
        ax.plot(
            payload["fpr"],
            payload["tpr"],
            color=cfg["color"],
            lw=cfg["linewidth"],
            linestyle=cfg["linestyle"],
            label=f"{method} (AUC={label_suffix(payload, 'roc_auc')})",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    style_axes(ax)


def plot_pr(ax: plt.Axes, curves: dict[str, dict[str, Any]], prevalence: Any, title: str) -> None:
    if isinstance(prevalence, list):
        for value, label, color in prevalence:
            ax.axhline(value, color=color, linestyle=":", alpha=0.5, linewidth=1.8, label=f"{label}={value:.2f}")
    else:
        ax.axhline(prevalence, color="k", linestyle=":", alpha=0.5, linewidth=1.8, label=f"Prevalence={prevalence:.2f}")
    for method, payload in curves.items():
        cfg = style_for_method(method)
        ax.plot(
            payload["recall"],
            payload["precision"],
            color=cfg["color"],
            lw=cfg["linewidth"],
            linestyle=cfg["linestyle"],
            label=f"{method} (AP={label_suffix(payload, 'average_precision')})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    style_axes(ax)


def save_figures(out_dir: Path, curves: dict[str, dict[str, Any]], prevalence: Any, title_prefix: str, basename: str) -> None:
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    plot_roc(ax1, curves, f"{title_prefix}: ROC")
    plot_pr(ax2, curves, prevalence, f"{title_prefix}: PR")
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_roc_pr.png")
    fig.savefig(out_dir / f"{basename}_roc_pr.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    plot_roc(ax, curves, f"{title_prefix}: ROC")
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_roc.png")
    fig.savefig(out_dir / f"{basename}_roc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    plot_pr(ax, curves, prevalence, f"{title_prefix}: PR")
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_pr.png")
    fig.savefig(out_dir / f"{basename}_pr.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y_true, scores_by_method = load_paper_scores(args.source_data_xlsx, args.sheet_name)
    if len(set(y_true.tolist())) < 2:
        raise ValueError("Need both positive and negative labels to plot ROC/PR curves.")

    baseline_curves = {method: method_curves(y_true, scores) for method, scores in scores_by_method.items()}
    for payload in baseline_curves.values():
        payload["n"] = int(len(y_true))
    prevalence = float(np.mean(y_true))
    baseline_metrics = baseline_metrics_rows(y_true, baseline_curves, args.source_data_xlsx, args.sheet_name)
    write_csv(args.out_dir / "pocketminer_baseline_metrics.csv", baseline_metrics + [sesame_unavailable_row()])
    write_csv(args.out_dir / "pocketminer_baseline_curve_points.csv", curve_rows(baseline_curves))
    (args.out_dir / "baseline_availability.json").write_text(
        json.dumps(availability_payload(args.source_data_xlsx, args.sheet_name), indent=2),
        encoding="utf-8",
    )
    save_figures(args.out_dir, baseline_curves, prevalence, args.title_prefix, "pocketminer_baseline")

    combined_curves = dict(baseline_curves)
    combined_metrics = baseline_metrics + [sesame_unavailable_row()]
    failures: list[dict[str, Any]] = []
    if args.include_holoshift and args.holoshift_prior_dir.exists():
        hs_rows, hs_failures = build_holoshift_residue_scores(
            manifest=args.manifest,
            labels_csv=args.labels_csv,
            prior_dir=args.holoshift_prior_dir,
        )
        write_csv(args.out_dir / "holoshift_aligned_residue_scores.csv", hs_rows)
        hs_curves, curve_failures = curves_from_score_rows(hs_rows)
        failures = hs_failures + curve_failures
        write_csv(args.out_dir / "holoshift_aligned_failures.csv", failures)
        combined_curves.update(hs_curves)
        combined_metrics = baseline_metrics + holoshift_metric_rows(hs_curves) + [sesame_unavailable_row()]
        write_csv(args.out_dir / "pocketminer_holoshift_aligned_metrics.csv", combined_metrics)
        write_csv(args.out_dir / "pocketminer_holoshift_aligned_curve_points.csv", curve_rows(combined_curves))
        hs_prevalences = [payload["n_positive"] / payload["n"] for payload in hs_curves.values()]
        hs_prevalence = float(np.mean(hs_prevalences)) if hs_prevalences else prevalence
        prevalence_lines = [
            (prevalence, "Source prev", "#7f8c8d"),
            (hs_prevalence, "HoloShift prev", "#8e44ad"),
        ]
        save_figures(
            args.out_dir,
            combined_curves,
            prevalence_lines,
            "PocketMiner Baselines + HoloShift Aligned Scores",
            "pocketminer_holoshift_aligned",
        )

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "baseline_metrics": baseline_metrics,
                "combined_metrics": combined_metrics,
                "n_holoshift_failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
