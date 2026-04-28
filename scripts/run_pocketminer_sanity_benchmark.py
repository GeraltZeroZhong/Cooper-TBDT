#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PocketMiner residue-level cryptic-pocket sanity metrics.")
    p.add_argument("--labels-csv", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_residue_labels.csv"))
    p.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional residue predictions with target_id,residue_index,score columns.",
    )
    p.add_argument(
        "--source-data-xlsx",
        type=Path,
        default=Path("outputs/pocketminer_holoshift/external/pocketminer_source_data.xlsx"),
        help="Nature source-data workbook. Used to reproduce the paper-level Figure 5 sanity metrics.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift/sanity"))
    p.add_argument("--top-k", default="5,10,20")
    return p.parse_args()


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
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


def parse_top_k(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def classification_metrics(y_true: list[int], scores: list[float]) -> dict[str, Any]:
    if len(set(y_true)) < 2:
        return {"n": len(y_true), "n_positive": sum(y_true), "n_negative": len(y_true) - sum(y_true)}
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "n": len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "max_f1": float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-12))),
    }


def topk_rows(rows: list[dict[str, Any]], top_k_values: list[int]) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[row["target_id"]].append(row)
    out: list[dict[str, Any]] = []
    for target_id, target_rows in sorted(by_target.items()):
        positives = sum(int(row["label"]) for row in target_rows)
        if positives == 0:
            continue
        ranked = sorted(target_rows, key=lambda row: float(row["score"]), reverse=True)
        for k in top_k_values:
            top = ranked[: min(k, len(ranked))]
            hits = sum(int(row["label"]) for row in top)
            out.append(
                {
                    "target_id": target_id,
                    "top_k": k,
                    "hits": hits,
                    "precision": hits / len(top) if top else "",
                    "recall": hits / positives if positives else "",
                    "n_positive": positives,
                    "n_labeled": len(ranked),
                }
            )
    return out


def evaluate_predictions(labels_csv: Path, predictions_csv: Path, top_k_values: list[int]) -> dict[str, Any]:
    labels = {
        (row["target_id"], int(row["residue_index"])): int(row["label"])
        for row in read_csv(labels_csv)
        if int(row.get("is_eval", "0")) == 1
    }
    scored_rows: list[dict[str, Any]] = []
    for row in read_csv(predictions_csv):
        key = (row["target_id"], int(row["residue_index"]))
        label = labels.get(key)
        if label is None:
            continue
        scored_rows.append(
            {
                "target_id": row["target_id"],
                "residue_index": int(row["residue_index"]),
                "label": int(label),
                "score": float(row["score"]),
            }
        )
    y_true = [int(row["label"]) for row in scored_rows]
    scores = [float(row["score"]) for row in scored_rows]
    return {
        "pooled": classification_metrics(y_true, scores),
        "residue_rows": scored_rows,
        "topk_rows": topk_rows(scored_rows, top_k_values),
    }


def evaluate_source_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "unavailable", "reason": f"missing source data workbook: {path}"}
    df = pd.read_excel(path, sheet_name="Figure 5c,d")
    pm_col = "PocketMiner Predcitions"
    cryptosite_col = "CryptoSite Predictions"
    true_col = "True Value"
    rows = df[[pm_col, cryptosite_col, true_col]].dropna()
    y_true = [int(x) for x in rows[true_col].tolist()]
    pm = [float(x) for x in rows[pm_col].tolist()]
    cryptosite = [float(x) for x in rows[cryptosite_col].tolist()]
    return {
        "status": "ok",
        "source": str(path),
        "pocketminer": classification_metrics(y_true, pm),
        "cryptosite": classification_metrics(y_true, cryptosite),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_top_k(args.top_k)
    source = evaluate_source_data(args.source_data_xlsx)
    (args.out_dir / "source_data_sanity.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    prediction_status: dict[str, Any]
    if args.predictions_csv is not None:
        prediction_eval = evaluate_predictions(args.labels_csv, args.predictions_csv, top_k_values)
        write_csv(args.out_dir / "pocketminer_prediction_residue_scores.csv", prediction_eval["residue_rows"])
        write_csv(args.out_dir / "pocketminer_prediction_topk_by_target.csv", prediction_eval["topk_rows"])
        prediction_status = {
            "status": "ok",
            "predictions_csv": str(args.predictions_csv),
            "pooled": prediction_eval["pooled"],
        }
    else:
        prediction_status = {
            "status": "unavailable",
            "reason": "No residue-level predictions CSV was provided. Run scripts/run_pocketminer_inference.py first.",
        }
    (args.out_dir / "prediction_sanity.json").write_text(json.dumps(prediction_status, indent=2), encoding="utf-8")
    summary = {"source_data_sanity": source, "prediction_sanity": prediction_status}
    (args.out_dir / "sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
