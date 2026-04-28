#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a compact PocketMiner/HoloShift benchmark report.")
    p.add_argument("--benchmark-dir", type=Path, default=Path("outputs/pocketminer_holoshift"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    try:
        if value == "":
            return ""
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:.3f}"
    except (TypeError, ValueError):
        return str(value)


def table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows available._"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return out


def main() -> None:
    args = parse_args()
    bench = args.benchmark_dir
    out = args.out or bench / "pocketminer_holoshift_report.md"
    manifest_status = read_json(bench / "pocketminer_manifest_status.json")
    source_sanity = read_json(bench / "sanity" / "source_data_sanity.json")
    prediction_sanity = read_json(bench / "sanity" / "prediction_sanity.json")
    prior_status = read_json(bench / "holoshift_prior_full" / "cryptic_prior_status.json")
    prior_summary = read_csv(bench / "holoshift_prior_full" / "cryptic_prior_summary_by_structure.csv")
    failures = read_csv(bench / "holoshift_prior_full" / "cryptic_prior_failures.csv")

    selected = [
        row
        for row in prior_summary
        if row.get("structure")
        in {
            "apo",
            "holoshift_unrelaxed",
            "holoshift_scale_2p00",
            "nma_m1_pos_0p50A",
            "nma_m1_neg_0p50A",
            "random_0p50A",
        }
    ]
    columns = [
        "structure",
        "n_targets",
        "positive_ca_rmsd_vs_holo_median",
        "delta_positive_ca_rmsd_vs_apo_median",
        "motion_cosine_mean_median",
        "motion_projection_mean_median",
        "pocket_grid_shape_jaccard_vs_true_median",
        "pocket_grid_false_blocked_fraction_vs_true_median",
        "ligand_clash_pairs_lt_cutoff_median",
    ]
    pm = source_sanity.get("pocketminer", {})
    cs = source_sanity.get("cryptosite", {})
    lines = [
        "# PocketMiner Sanity + HoloShift Cryptic-Pocket Prior",
        "",
        "## Inputs",
        "",
        f"- PocketMiner label targets: {manifest_status.get('n_targets', '')}.",
        f"- Apo-holo-ligand cryptic targets with local structures: {manifest_status.get('n_cryptic_holo_targets', '')}.",
        f"- Labeled eval residues: {manifest_status.get('n_eval_residues', '')}; positives: {manifest_status.get('n_positive_residues', '')}; negatives: {manifest_status.get('n_negative_residues', '')}.",
        "",
        "## PocketMiner Sanity",
        "",
        f"- Source-data PocketMiner ROC-AUC: {fmt(pm.get('roc_auc', ''))}; PR-AUC: {fmt(pm.get('pr_auc', ''))}.",
        f"- Source-data CryptoSite ROC-AUC: {fmt(cs.get('roc_auc', ''))}; PR-AUC: {fmt(cs.get('pr_auc', ''))}.",
        f"- Local residue-level inference status: {prediction_sanity.get('status', 'missing')}.",
        "",
        "## HoloShift Cryptic-Pocket Prior",
        "",
        f"- Targets evaluated: {prior_status.get('n_targets_requested', '')}.",
        f"- Metric rows: {prior_status.get('n_metric_rows', '')}.",
        f"- Failures: {prior_status.get('n_failures', '')}.",
        f"- HoloShift rows present: {prior_status.get('has_holoshift_rows', '')}.",
        "",
        *table(selected, columns),
        "",
        "## Interpretation",
        "",
        "- HoloShift produces a small positive median shift toward holo on cryptic ligand-lining residues, but the effect size is modest.",
        "- The scale ensemble shows monotonic median geometry movement, which supports a conformational-prior interpretation rather than a docking-improvement claim.",
        "- NMA/random baselines remain essential: single-mode NMA can outperform or underperform depending on direction, so directionality and scale must be reported target-by-target.",
        "- This benchmark is out-of-distribution for HoloShift confidence features because PocketMiner apo structures are experimental PDBs, not AF2 models with pLDDT in B-factors.",
    ]
    if failures:
        lines += ["", "## Failures", "", *table(failures, list(failures[0].keys()))]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
