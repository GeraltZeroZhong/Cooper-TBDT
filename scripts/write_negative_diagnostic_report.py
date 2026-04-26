#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a compact negative/diagnostic benchmark completion report.")
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def md_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return ["_No rows available._"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            try:
                if value != "":
                    numeric = float(value)
                    value = str(int(numeric)) if numeric.is_integer() else f"{numeric:.3f}"
            except (TypeError, ValueError):
                pass
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def main() -> None:
    args = parse_args()
    bench = args.benchmark_dir
    out = args.out or bench / "benchmark_completion_report.md"

    manifest = read_csv(bench / "posebusters_multitarget_manifest.csv")
    redocking = read_json(bench / "redocking_gate_exh8" / "summary.json")
    consistency = read_json(bench / "redocking_gate_exh8" / "consistency_check.json")
    readiness_status = read_json(bench / "readiness_100" / "readiness_status.json")
    aggregate_status = read_json(bench / "aggregate" / "aggregate_status.json")
    docking_table = read_csv(bench / "aggregate" / "table_docking_pose_power_by_arm.csv")
    readiness_table = read_csv(bench / "aggregate" / "table_binding_readiness_by_arm.csv")
    vs_status = read_json(bench / "aggregate" / "vs_enrichment_status.json")

    split_counts: dict[str, int] = {}
    for row in manifest:
        split = row.get("split", "")
        split_counts[split] = split_counts.get(split, 0) + 1
    split_rows = [{"split": key, "n_targets": value} for key, value in sorted(split_counts.items())]

    sanity = redocking.get("redocking_sanity", {})
    by_structure = redocking.get("by_structure", {}).get("true_holo", {})
    topn = by_structure.get("topn_success", {})
    failures = redocking.get("failures", [])

    lines = [
        "# Negative/Diagnostic Docking Benchmark Completion Report",
        "",
        "## Scope",
        "",
        f"- Manifest targets: {len(manifest)} PoseBusters complexes.",
        "- Split: family-hash split from `protein_family` / UniProt mapping where available.",
        "- Local benchmark role: true-holo redocking gate plus predicted-structure readiness diagnostics.",
        "- HoloShift multi-target arm: unavailable in this local dataset because no complete HoloShift predictions were present under `outputs/holoshift_posebusters`.",
        "",
        "## Family-Level Split",
        "",
        *md_table(split_rows, ["split", "n_targets"]),
        "",
        "## True-Holo Redocking Gate",
        "",
        f"- Evaluated targets: {sanity.get('n_targets', '')}.",
        f"- Top-5 gate passed: {sanity.get('n_passed', '')}.",
        f"- Top-5 gate failed: {sanity.get('n_failed', '')}.",
        f"- Pipeline-level failures before pose statistics: {len(failures)}.",
        f"- Consistency check: {consistency.get('status', 'missing')}.",
        "",
        *md_table(
            [
                {
                    "top1": topn.get("top1_success_rate", ""),
                    "top3": topn.get("top3_success_rate", ""),
                    "top5": topn.get("top5_success_rate", ""),
                    "top20": topn.get("top20_success_rate", ""),
                }
            ],
            ["top1", "top3", "top5", "top20"],
        ),
        "",
        "## Docking Diagnostic Outputs",
        "",
        *md_table(
            docking_table,
            [
                "structure",
                "n_targets",
                "top1_success_rate",
                "top3_success_rate",
                "top5_success_rate",
                "top20_success_rate",
                "best_rmsd_median",
                "first_hit_rank_median",
                "score_pose_discordant_fraction",
            ],
        ),
        "",
        "## Pocket Readiness Outputs",
        "",
        f"- Readiness targets requested: {readiness_status.get('n_targets_requested', '')}.",
        f"- Readiness metric rows: {readiness_status.get('n_metric_rows', '')}.",
        f"- Readiness failures: {readiness_status.get('n_failures', '')}.",
        "",
        *md_table(
            readiness_table,
            [
                "structure",
                "n_rows",
                "pocket_ca_rmsd_direct_vs_true_median",
                "delta_pocket_ca_rmsd_direct_vs_true_vs_raw_median",
                "pocket_sidechain_heavy_rmsd_direct_vs_true_median",
                "pocket_grid_shape_jaccard_vs_true_median",
                "ligand_clash_pairs_lt_cutoff_median",
                "pocket_ca_delta_cosine_mean_vs_raw_to_true_median",
            ],
        ),
        "",
        "## VS/Enrichment",
        "",
        f"- Status: {vs_status.get('status', aggregate_status.get('vs_enrichment_status', {}).get('status', 'missing'))}.",
        f"- Reason: {vs_status.get('reason', aggregate_status.get('vs_enrichment_status', {}).get('reason', '')).rstrip('.')}.",
        "",
        "## Interpretation Guardrails",
        "",
        "- Do not claim virtual-screening enrichment without active/decoy panels.",
        "- Do not claim pose-power improvement outside targets that pass the true-holo redocking sanity gate.",
        "- Current local multi-target data supports a diagnostic benchmark scaffold and negative control analysis; it does not yet support a HoloShift-vs-baseline docking-improvement claim.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
