"""Plot residue-level ROC/PR curves for TBDT state-shift detection.

This script aligns score-only external baselines with HoloShift predictions by
turning the displacement target into a residue classification task:

positive residue := ||target_delta|| >= positive_threshold

Model predictions are scored by predicted displacement magnitude. External
baselines such as low AF2 pLDDT or surface exposure are scored directly from
the processed graph features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evopoint_da.pipeline.eval_tbdt_state import (
    _as_mapping,
    _extract_prediction,
    _extract_regions,
    _extract_target,
    _load_pt,
)


BASELINE_CHOICES = ("af2_low_plddt", "af2_surface_rsa", "af2_surface_sasa")
BASELINE_LABELS = {
    "af2_low_plddt": "AF2 low pLDDT",
    "af2_surface_rsa": "AF2 surface RSA",
    "af2_surface_sasa": "AF2 surface SASA",
}
PREDICTION_LABELS = {
    "holoshift_scaffold_blend": "HoloShift-TBDT scaffold blend",
    "nearest_template": "Nearest template transfer",
    "family_state_average": "Family/state average transfer",
}


@dataclass(frozen=True)
class MethodScores:
    name: str
    label: str
    scores: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate residue-level ROC/PR curves for TBDT state-shift detection. "
            "The common positive label is target displacement magnitude above a threshold."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Processed .pt samples or directories. Ignored when --sample-list is supplied.",
    )
    parser.add_argument(
        "--sample-list",
        default=None,
        help="Text file containing one processed .pt sample path per line.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help=(
            "Prediction directory to score by ||pred_delta||. May be passed multiple times. "
            "A directory is matched to samples by file stem."
        ),
    )
    parser.add_argument(
        "--external-baseline",
        action="append",
        choices=BASELINE_CHOICES,
        default=[],
        help="Score-only external baseline to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Region mask to evaluate. May be passed multiple times. Default: eval.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=1.0,
        help="Residues with target displacement magnitude >= threshold Angstrom are positives.",
    )
    parser.add_argument("--sasa-feature-index", type=int, default=129)
    parser.add_argument("--rsa-feature-index", type=int, default=130)
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/classification_curves")
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def _collect_samples(inputs: list[str], sample_list: str | None) -> list[Path]:
    files: list[Path] = []
    if sample_list:
        with open(sample_list, "r", encoding="utf-8") as handle:
            files.extend(Path(line.strip()) for line in handle if line.strip() and not line.startswith("#"))
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.pt")))
        elif path.is_file() and path.suffix == ".pt":
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path is not a .pt file or directory: {path}")
    deduped = sorted({path.resolve(): path for path in files}.values())
    if not deduped:
        raise FileNotFoundError("No sample .pt files found.")
    return deduped


def _parse_prediction_specs(specs: list[str]) -> list[tuple[str, str, Path]]:
    parsed: list[tuple[str, str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--prediction must use NAME=DIR format, got: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Prediction name is empty in: {spec}")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {path}")
        label = PREDICTION_LABELS.get(name, name.replace("_", " "))
        parsed.append((name, label, path))
    return parsed


def _as_1d_float(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().cpu().float().reshape(-1)


def _feature_column(sample: dict[str, Any], index: int, sample_path: Path) -> torch.Tensor:
    x = sample.get("x")
    if x is None:
        raise ValueError(f"{sample_path} has no x feature matrix.")
    x_tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if x_tensor.dim() != 2 or x_tensor.size(1) <= index:
        raise ValueError(f"{sample_path} x shape {tuple(x_tensor.shape)} does not contain feature index {index}.")
    return x_tensor.detach().cpu().float()[:, index].reshape(-1)


def _plddt_score(sample: dict[str, Any], sample_path: Path) -> torch.Tensor:
    value = sample.get("plddt")
    if value is None:
        metadata = sample.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("plddt")
    if value is None:
        raise ValueError(f"{sample_path} has no explicit plddt field for the AF2 low-pLDDT baseline.")
    else:
        plddt = _as_1d_float(value)
        if bool((plddt > 1.5).any()):
            plddt = plddt / 100.0
    return 1.0 - plddt.clamp(0.0, 1.0)


def _baseline_score(
    baseline_name: str,
    sample: dict[str, Any],
    sample_path: Path,
    args: argparse.Namespace,
) -> torch.Tensor:
    if baseline_name == "af2_low_plddt":
        return _plddt_score(sample, sample_path)
    if baseline_name == "af2_surface_rsa":
        return _feature_column(sample, int(args.rsa_feature_index), sample_path)
    if baseline_name == "af2_surface_sasa":
        return _feature_column(sample, int(args.sasa_feature_index), sample_path)
    raise ValueError(f"Unknown external baseline: {baseline_name}")


def _prediction_score(prediction_dir: Path, sample_path: Path, n: int) -> torch.Tensor:
    pred_path = prediction_dir / f"{sample_path.stem}.pt"
    if not pred_path.exists():
        matches = sorted(prediction_dir.rglob(f"{sample_path.stem}.pt"))
        if not matches:
            raise FileNotFoundError(f"No prediction found for {sample_path.stem} in {prediction_dir}")
        pred_path = matches[0]
    prediction = _extract_prediction(_load_pt(pred_path), sample_path.stem, n)
    return torch.linalg.vector_norm(prediction, dim=-1)


def _roc_curve(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = y_true.astype(bool)
    s = score.astype(float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return np.array([np.nan]), np.array([np.nan]), float("nan")

    distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[distinct].astype(float)
    fp = (1 + distinct - np.cumsum(y)[distinct]).astype(float)
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _pr_curve(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    y = y_true.astype(bool)
    s = score.astype(float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    positives = int(y.sum())
    if positives == 0:
        return np.array([np.nan]), np.array([np.nan]), float("nan"), float("nan")

    distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[distinct].astype(float)
    fp = (1 + distinct - np.cumsum(y)[distinct]).astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    recall_curve = np.r_[0.0, recall]
    precision_curve = np.r_[1.0, precision]
    auprc_trapz = float(np.trapezoid(precision_curve, recall_curve))
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    return recall_curve, precision_curve, auprc_trapz, ap


def _save_points_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_curves(
    out_path: Path,
    *,
    region: str,
    curve_type: str,
    curves: dict[str, dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    metric_key: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    metric_label = {"auroc": "AUROC", "average_precision": "AP"}.get(metric_key, metric_key.upper())
    for label, payload in curves.items():
        x = payload[x_key]
        y = payload[y_key]
        metric = payload.get(metric_key, float("nan"))
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        suffix = f" ({metric_label}={metric:.3f})" if math.isfinite(float(metric)) else ""
        ax.plot(x, y, linewidth=2.0, label=f"{label}{suffix}")
    if curve_type == "roc":
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="0.6", label="Random")
    ax.set_title(f"TBDT {region} residue state-shift {curve_type.upper()}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples = _collect_samples(args.inputs, args.sample_list)
    prediction_specs = _parse_prediction_specs(args.prediction)
    baselines = list(args.external_baseline) or ["af2_low_plddt", "af2_surface_rsa"]
    regions = list(args.region) or ["eval"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_region: dict[str, list[np.ndarray]] = {region: [] for region in regions}
    scores_by_region: dict[str, dict[str, MethodScores]] = {region: {} for region in regions}
    for region in regions:
        for name, label, _ in prediction_specs:
            scores_by_region[region][name] = MethodScores(name=name, label=label, scores=[])
        for baseline in baselines:
            scores_by_region[region][baseline] = MethodScores(
                name=baseline,
                label=BASELINE_LABELS[baseline],
                scores=[],
            )

    sample_rows: list[dict[str, Any]] = []
    for sample_path in samples:
        sample = _as_mapping(_load_pt(sample_path))
        target = _extract_target(sample, sample_path)
        n = int(target.size(0))
        target_mag = torch.linalg.vector_norm(target, dim=-1)
        region_masks = _extract_regions(sample, sample_path.stem, n, {}, include_all=True)

        method_scores: dict[str, torch.Tensor] = {}
        for name, _, pred_dir in prediction_specs:
            method_scores[name] = _prediction_score(pred_dir, sample_path, n)
        for baseline in baselines:
            method_scores[baseline] = _baseline_score(baseline, sample, sample_path, args)

        for region in regions:
            if region not in region_masks:
                continue
            mask = region_masks[region].bool()
            if not bool(mask.any()):
                continue
            labels = (target_mag[mask] >= float(args.positive_threshold)).cpu().numpy().astype(bool)
            labels_by_region[region].append(labels)
            sample_rows.append(
                {
                    "sample_id": sample_path.stem,
                    "region": region,
                    "n_residues": int(mask.sum().item()),
                    "n_positive": int(labels.sum()),
                    "positive_rate": float(labels.mean()) if labels.size else float("nan"),
                }
            )
            for method_name, score in method_scores.items():
                if score.numel() != n:
                    raise ValueError(
                        f"Score length mismatch for {method_name}/{sample_path.stem}: "
                        f"{score.numel()} vs sample length {n}"
                    )
                scores_by_region[region][method_name].scores.append(score[mask].cpu().numpy().astype(float))

    summary_rows: list[dict[str, Any]] = []
    points_rows: list[dict[str, Any]] = []
    output_files: dict[str, dict[str, str]] = {}

    for region in regions:
        y_true = np.concatenate(labels_by_region[region]) if labels_by_region[region] else np.array([], dtype=bool)
        if y_true.size == 0:
            raise ValueError(f"No residues collected for region '{region}'.")
        region_curves: dict[str, dict[str, Any]] = {}
        for method_name, method in scores_by_region[region].items():
            scores = np.concatenate(method.scores) if method.scores else np.array([], dtype=float)
            if scores.size != y_true.size:
                raise ValueError(
                    f"Collected score count mismatch for {method_name}/{region}: {scores.size} vs {y_true.size}"
                )
            fpr, tpr, auroc = _roc_curve(y_true, scores)
            recall, precision, auprc, ap = _pr_curve(y_true, scores)
            label = method.label
            region_curves[label] = {
                "method": method_name,
                "scores": scores,
                "fpr": fpr,
                "tpr": tpr,
                "recall": recall,
                "precision": precision,
                "auroc": auroc,
                "auprc": auprc,
                "average_precision": ap,
            }
            summary_rows.append(
                {
                    "region": region,
                    "method": method_name,
                    "label": label,
                    "n_residues": int(y_true.size),
                    "n_positive": int(y_true.sum()),
                    "n_negative": int(y_true.size - y_true.sum()),
                    "positive_threshold_angstrom": float(args.positive_threshold),
                    "positive_rate": float(y_true.mean()),
                    "auroc": auroc,
                    "auprc_trapezoid": auprc,
                    "average_precision": ap,
                    "score_mean": float(np.mean(scores)),
                    "score_median": float(np.median(scores)),
                }
            )
            for x_value, y_value in zip(fpr.tolist(), tpr.tolist(), strict=False):
                points_rows.append(
                    {
                        "region": region,
                        "method": method_name,
                        "curve": "roc",
                        "x": x_value,
                        "y": y_value,
                    }
                )
            for x_value, y_value in zip(recall.tolist(), precision.tolist(), strict=False):
                points_rows.append(
                    {
                        "region": region,
                        "method": method_name,
                        "curve": "pr",
                        "x": x_value,
                        "y": y_value,
                    }
                )

        roc_path = out_dir / f"roc_{region}.png"
        pr_path = out_dir / f"pr_{region}.png"
        _plot_curves(
            roc_path,
            region=region,
            curve_type="roc",
            curves=region_curves,
            x_key="fpr",
            y_key="tpr",
            x_label="False positive rate",
            y_label="True positive rate",
            metric_key="auroc",
            dpi=int(args.dpi),
        )
        _plot_curves(
            pr_path,
            region=region,
            curve_type="pr",
            curves=region_curves,
            x_key="recall",
            y_key="precision",
            x_label="Recall",
            y_label="Precision",
            metric_key="average_precision",
            dpi=int(args.dpi),
        )
        output_files[region] = {"roc": str(roc_path), "pr": str(pr_path)}

    summary_csv = out_dir / "classification_curve_summary.csv"
    points_csv = out_dir / "classification_curve_points.csv"
    sample_csv = out_dir / "classification_curve_samples.csv"
    _save_points_csv(
        summary_csv,
        rows=summary_rows,
        columns=[
            "region",
            "method",
            "label",
            "n_residues",
            "n_positive",
            "n_negative",
            "positive_threshold_angstrom",
            "positive_rate",
            "auroc",
            "auprc_trapezoid",
            "average_precision",
            "score_mean",
            "score_median",
        ],
    )
    _save_points_csv(points_csv, rows=points_rows, columns=["region", "method", "curve", "x", "y"])
    _save_points_csv(
        sample_csv,
        rows=sample_rows,
        columns=["sample_id", "region", "n_residues", "n_positive", "positive_rate"],
    )

    report = {
        "samples": [str(path) for path in samples],
        "n_samples": len(samples),
        "positive_definition": f"target displacement magnitude >= {float(args.positive_threshold):.3f} A",
        "regions": regions,
        "prediction_methods": [{"name": name, "path": str(path)} for name, _, path in prediction_specs],
        "external_baselines": baselines,
        "summary": summary_rows,
        "output_files": output_files,
        "summary_csv": str(summary_csv),
        "points_csv": str(points_csv),
        "sample_csv": str(sample_csv),
    }
    report_path = out_dir / "classification_curve_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    report = evaluate(parse_args())
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
