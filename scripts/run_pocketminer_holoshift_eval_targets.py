#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HoloShift unrelaxed predictions for all PocketMiner eval-label targets.")
    p.add_argument("--manifest", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_cryptic_manifest.csv"))
    p.add_argument("--labels-csv", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_residue_labels.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift/holoshift_prior_full"))
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("checkpoints/gvp_full_beta1_mse_selection/20260426-111016/best-selection-06-1.0231.ckpt"),
    )
    p.add_argument("--esm-weights", type=Path, default=Path("esmc_weights/esmc_600m_2024_12_v0.pth"))
    p.add_argument("--pca-path", type=Path, default=Path("data/pca_esmc_128.pkl"))
    p.add_argument("--allow-dssp-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, default=None)
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


def eval_target_ids(labels_csv: Path) -> set[str]:
    return {row["target_id"] for row in read_csv(labels_csv) if row.get("is_eval") == "1"}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = eval_target_ids(args.labels_csv)
    manifest_rows = [
        row
        for row in read_csv(args.manifest)
        if row.get("target_id") in eval_ids and row.get("apo_pdb")
    ]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in manifest_rows:
        target_id = row["target_id"]
        target_dir = args.out_dir / "targets" / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        out_pdb = target_dir / f"{target_id}_holoshift_unrelaxed.pdb"
        report = target_dir / f"{target_id}_holoshift_report.json"
        feature = target_dir / f"{target_id}_training_graph_features.pt"
        if args.reuse and out_pdb.exists() and report.exists():
            results.append({"target_id": target_id, "status": "reused", "output_pdb": str(out_pdb)})
            continue
        cmd = [
            str(args.python),
            "run_Predict.py",
            "--pdb_file",
            row["apo_pdb"],
            "--feature_out",
            str(feature),
            "--esm_weights",
            str(args.esm_weights),
            "--pca_path",
            str(args.pca_path),
            "--ckpt_path",
            str(args.ckpt_path),
            "--chain_id",
            row.get("apo_chain") or "A",
            "--output_pdb",
            str(out_pdb),
            "--report_json",
            str(report),
            "--no-run_relax",
        ]
        if args.allow_dssp_fallback:
            cmd.append("--allow_dssp_fallback")
        try:
            subprocess.run(cmd, check=True)
            results.append({"target_id": target_id, "status": "ran", "output_pdb": str(out_pdb)})
        except Exception as exc:
            failures.append({"target_id": target_id, "status": "failed", "error": str(exc)})

    write_csv(args.out_dir / "eval_target_holoshift_runs.csv", results)
    write_csv(args.out_dir / "eval_target_holoshift_failures.csv", failures)
    status = {
        "n_eval_targets_requested": len(manifest_rows),
        "n_reused": sum(1 for row in results if row["status"] == "reused"),
        "n_ran": sum(1 for row in results if row["status"] == "ran"),
        "n_failures": len(failures),
        "experimental_bfactor_warning": (
            "PocketMiner apo PDBs are experimental structures; HoloShift feature builder treats CA B-factors as pLDDT."
        ),
    }
    (args.out_dir / "eval_target_holoshift_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
