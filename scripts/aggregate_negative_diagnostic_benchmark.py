#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-target docking/readiness diagnostic benchmark outputs.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--docking-dir", type=Path, default=None)
    p.add_argument("--readiness-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/negative_diagnostic_benchmark/aggregate"))
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
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


def f(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def median_iqr(values: list[float]) -> tuple[float | str, float | str, float | str]:
    if not values:
        return "", "", ""
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def as_float_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = [f(row.get(key)) for row in rows]
    return [x for x in vals if x is not None]


def summarize_pose_rows(rows: list[dict[str, str]], rmsd_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("pose_valid") == "1":
            grouped[(row.get("target_id", ""), row.get("structure", ""))].append(row)

    target_rows: list[dict[str, Any]] = []
    for (target_id, structure), pose_rows in sorted(grouped.items()):
        ranked = sorted(pose_rows, key=lambda row: int(float(row.get("rank", 10**9) or 10**9)))
        rmsds = [f(row.get("rmsd")) for row in ranked]
        rmsds = [x for x in rmsds if x is not None]
        scores = [f(row.get("score")) for row in ranked]
        scores = [x for x in scores if x is not None]
        best_row = min(ranked, key=lambda row: f(row.get("rmsd")) if f(row.get("rmsd")) is not None else math.inf)
        first_hit = next(
            (int(row["rank"]) for row in ranked if (f(row.get("rmsd")) or math.inf) < rmsd_threshold),
            "",
        )
        row: dict[str, Any] = {
            "target_id": target_id,
            "structure": structure,
            "n_poses": len(ranked),
            "top1_rmsd": f(ranked[0].get("rmsd")) if ranked else "",
            "top1_score": f(ranked[0].get("score")) if ranked else "",
            "best_rmsd": f(best_row.get("rmsd")) if ranked else "",
            "best_rank": int(best_row["rank"]) if ranked else "",
            "first_hit_rank": first_hit,
            "score_rmsd_pearson": "",
            "score_pose_discordant": int(int(best_row["rank"]) != 1) if ranked else "",
        }
        if len(rmsds) >= 2 and len(scores) == len(rmsds):
            row["score_rmsd_pearson"] = float(np.corrcoef(np.asarray(scores), np.asarray(rmsds))[0, 1])
        for n in (1, 3, 5, 20):
            topn = [pose for pose in ranked if int(pose["rank"]) <= n]
            best_topn = min((f(pose.get("rmsd")) or math.inf for pose in topn), default=math.inf)
            row[f"best_rmsd_top{n}"] = "" if math.isinf(best_topn) else float(best_topn)
            row[f"top{n}_success"] = int(
                any((f(pose.get("rmsd")) or math.inf) < rmsd_threshold for pose in topn)
            )
        target_rows.append(row)

    by_arm: list[dict[str, Any]] = []
    structures = sorted({row["structure"] for row in target_rows})
    for structure in structures:
        sr = [row for row in target_rows if row["structure"] == structure]
        top1 = [f(row["top1_rmsd"]) for row in sr]
        top1 = [x for x in top1 if x is not None]
        best = [f(row["best_rmsd"]) for row in sr]
        best = [x for x in best if x is not None]
        med_top1, q1_top1, q3_top1 = median_iqr(top1)
        med_best, q1_best, q3_best = median_iqr(best)
        first_hits = as_float_list(sr, "first_hit_rank")
        pearsons = as_float_list(sr, "score_rmsd_pearson")
        by_arm.append(
            {
                "structure": structure,
                "n_targets": len(sr),
                "top1_success_rate": float(np.mean([int(row["top1_success"]) for row in sr])) if sr else "",
                "top3_success_rate": float(np.mean([int(row["top3_success"]) for row in sr])) if sr else "",
                "top5_success_rate": float(np.mean([int(row["top5_success"]) for row in sr])) if sr else "",
                "top20_success_rate": float(np.mean([int(row["top20_success"]) for row in sr])) if sr else "",
                "top1_rmsd_median": med_top1,
                "top1_rmsd_q1": q1_top1,
                "top1_rmsd_q3": q3_top1,
                "best_rmsd_median": med_best,
                "best_rmsd_q1": q1_best,
                "best_rmsd_q3": q3_best,
                "first_hit_available_n": sum(1 for row in sr if str(row.get("first_hit_rank", "")).strip()),
                "first_hit_rank_median": median_iqr(first_hits)[0],
                "first_hit_rank_q3": median_iqr(first_hits)[2],
                "score_rmsd_pearson_median": median_iqr(pearsons)[0],
                "score_pose_discordant_fraction": (
                    float(np.mean([int(row["score_pose_discordant"]) for row in sr])) if sr else ""
                ),
            }
        )
        for n in (1, 3, 5, 20):
            vals = as_float_list(sr, f"best_rmsd_top{n}")
            med, q1, q3 = median_iqr(vals)
            by_arm[-1][f"best_rmsd_top{n}_median"] = med
            by_arm[-1][f"best_rmsd_top{n}_q1"] = q1
            by_arm[-1][f"best_rmsd_top{n}_q3"] = q3
    return target_rows, by_arm


def augment_readiness_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(row) for row in rows]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in out:
        by_target[row.get("target_id", "")].append(row)

    rmsd_metrics = [
        "global_ca_rmsd_direct_vs_true",
        "global_ca_rmsd_aligned_vs_true",
        "pocket_ca_rmsd_direct_vs_true",
        "pocket_ca_rmsd_aligned_vs_true",
        "pocket_sidechain_heavy_rmsd_direct_vs_true",
        "pocket_sidechain_heavy_rmsd_aligned_vs_true",
    ]
    for target_rows in by_target.values():
        baseline = next((row for row in target_rows if row.get("structure") == "raw_af2"), None)
        if baseline is None:
            baseline = next((row for row in target_rows if row.get("kind", "").startswith("baseline")), None)
        if baseline is None:
            continue
        for row in target_rows:
            for metric in rmsd_metrics:
                base = f(baseline.get(metric))
                current = f(row.get(metric))
                key = f"delta_{metric}_vs_raw"
                row[key] = "" if base is None or current is None else base - current
    return out


def summarize_readiness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_structure: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_structure[row.get("structure", "")].append(row)
    metrics = [
        "pocket_ca_rmsd_direct_vs_true",
        "delta_pocket_ca_rmsd_direct_vs_true_vs_raw",
        "pocket_sidechain_heavy_rmsd_direct_vs_true",
        "delta_pocket_sidechain_heavy_rmsd_direct_vs_true_vs_raw",
        "pocket_chi1_mae_deg",
        "pocket_chi2_mae_deg",
        "pocket_chi1_changed_recovered_rate",
        "pocket_chi2_changed_recovered_rate",
        "pocket_chi1_changed_improved_rate",
        "pocket_chi2_changed_improved_rate",
        "pocket_grid_shape_jaccard_vs_true",
        "pocket_grid_false_open_fraction_vs_true",
        "pocket_grid_false_blocked_fraction_vs_true",
        "ligand_clash_pairs_lt_cutoff",
        "ligand_steric_overlap_score",
        "pocket_ca_delta_cosine_mean_vs_raw_to_true",
        "pocket_ca_delta_projection_mean_vs_raw_to_true",
    ]
    out: list[dict[str, Any]] = []
    for structure, sr in sorted(by_structure.items()):
        row: dict[str, Any] = {"structure": structure, "n_rows": len(sr)}
        for metric in metrics:
            vals = [f(item.get(metric)) for item in sr]
            vals = [x for x in vals if x is not None]
            med, q1, q3 = median_iqr(vals)
            row[f"{metric}_median"] = med
            row[f"{metric}_q1"] = q1
            row[f"{metric}_q3"] = q3
        out.append(row)
    return out


def enrichment_status(manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    has_decoys = any(
        str(row.get("decoy_sdf", "")).strip() or str(row.get("decoys_sdf", "")).strip() for row in manifest_rows
    )
    has_actives = any(
        str(row.get("active_sdf", "")).strip() or str(row.get("actives_sdf", "")).strip() for row in manifest_rows
    )
    if has_decoys and has_actives:
        return {
            "status": "ready_for_vs",
            "reason": "Manifest contains active and decoy SDF columns.",
            "required_outputs": ["ROC-AUC", "PR-AUC", "EF1", "EF5", "BEDROC"],
        }
    return {
        "status": "unavailable",
        "reason": "No active/decoy SDF panel is present in the manifest; do not make VS/enrichment claims.",
        "required_inputs": ["active_sdf or actives_sdf", "decoy_sdf or decoys_sdf"],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(args.manifest)
    pose_rows = read_csv(args.docking_dir / "poses_all.csv" if args.docking_dir else None)
    readiness_rows = read_csv(args.readiness_csv)

    target_pose_rows, pose_by_arm = summarize_pose_rows(pose_rows, args.rmsd_threshold)
    augmented_readiness_rows = augment_readiness_rows(readiness_rows)
    readiness_by_arm = summarize_readiness(augmented_readiness_rows)
    status = enrichment_status(manifest_rows)

    write_csv(args.out_dir / "target_pose_diagnostics.csv", target_pose_rows)
    write_csv(args.out_dir / "target_readiness_diagnostics.csv", augmented_readiness_rows)
    write_csv(args.out_dir / "table_docking_pose_power_by_arm.csv", pose_by_arm)
    write_csv(args.out_dir / "table_binding_readiness_by_arm.csv", readiness_by_arm)
    (args.out_dir / "vs_enrichment_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    report = {
        "n_manifest_targets": len(manifest_rows),
        "n_pose_target_arm_rows": len(target_pose_rows),
        "n_readiness_rows": len(augmented_readiness_rows),
        "vs_enrichment_status": status,
    }
    (args.out_dir / "aggregate_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
