#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


FIVE_WAY_ARMS = [
    ("raw_af2", "Raw AF2", "poses_raw_af2.csv", "raw_af2"),
    ("af2_openmm_relax", "AF2 + OpenMM Relax", "poses_af2_openmm_relax.csv", "af2_openmm_heavy_relax"),
    ("holoshift_unrelaxed", "HoloShift unrelaxed", "poses_holoshift_unrelaxed.csv", "holoshift_shift_only"),
    ("holoshift_openmm_relax", "HoloShift + OpenMM Relax", "poses_holoshift.csv", "holoshift_heavy_relax"),
    ("true_holo", "True Holo PDB", "poses_true_holo.csv", "true_holo"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a publication-oriented diagnostic benchmark package from current HoloShift outputs."
    )
    p.add_argument(
        "--five-way-dir",
        type=Path,
        default=Path("outputs/docking_four_way/5S8I_2LY_publishable_single_dsspfix"),
    )
    p.add_argument(
        "--binding-readiness-dir",
        type=Path,
        default=Path("outputs/binding_readiness/5S8I_2LY_publishable_single_dsspfix"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/publication_diagnostic/5S8I_2LY_publishable_single_dsspfix"),
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


def as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def fmt(value: Any, digits: int = 3) -> str:
    val = as_float(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def summarize_pose_file(path: Path) -> dict[str, Any]:
    rows = [row for row in read_csv(path) if row.get("status") == "ok" and row.get("pose_valid") == "1"]
    rows.sort(key=lambda row: int(float(row.get("rank", 10**9) or 10**9)))
    if not rows:
        return {
            "n_poses": 0,
            "top1_rmsd": "",
            "top1_score": "",
            "best_rmsd": "",
            "best_rank": "",
            "top3_success": "",
            "top5_success": "",
            "top20_success": "",
        }
    best = min(rows, key=lambda row: as_float(row.get("rmsd")) if as_float(row.get("rmsd")) is not None else math.inf)
    return {
        "n_poses": len(rows),
        "top1_rmsd": as_float(rows[0].get("rmsd")),
        "top1_score": as_float(rows[0].get("score")),
        "best_rmsd": as_float(best.get("rmsd")),
        "best_rank": int(float(best.get("rank", 0) or 0)),
        "top3_success": int(any((as_float(row.get("rmsd")) or math.inf) < 2.0 for row in rows if int(row["rank"]) <= 3)),
        "top5_success": int(any((as_float(row.get("rmsd")) or math.inf) < 2.0 for row in rows if int(row["rank"]) <= 5)),
        "top20_success": int(any((as_float(row.get("rmsd")) or math.inf) < 2.0 for row in rows if int(row["rank"]) <= 20)),
    }


def build_pilot_metrics(five_way_dir: Path, binding_dir: Path) -> list[dict[str, Any]]:
    structure_rows = {row["structure"]: row for row in read_csv(binding_dir / "structure_readiness_metrics.csv")}
    rows: list[dict[str, Any]] = []
    raw_pose: dict[str, Any] | None = None
    raw_struct = structure_rows.get("raw_af2", {})
    for label, display, pose_rel, structure_label in FIVE_WAY_ARMS:
        pose = summarize_pose_file(five_way_dir / pose_rel)
        if label == "raw_af2":
            raw_pose = pose
        struct = structure_rows.get(structure_label, {})
        row = {
            "arm": label,
            "display_name": display,
            "n_poses": pose.get("n_poses", ""),
            "top1_rmsd": pose.get("top1_rmsd", ""),
            "top1_score": pose.get("top1_score", ""),
            "best_rmsd": pose.get("best_rmsd", ""),
            "best_rank": pose.get("best_rank", ""),
            "top3_success": pose.get("top3_success", ""),
            "top5_success": pose.get("top5_success", ""),
            "top20_success": pose.get("top20_success", ""),
            "delta_top1_rmsd_vs_raw": "",
            "delta_best_rmsd_vs_raw": "",
            "pocket_ca_rmsd_direct_vs_true": struct.get("pocket_ca_rmsd_direct_vs_true", ""),
            "delta_pocket_ca_rmsd_vs_raw": "",
            "pocket_sidechain_heavy_rmsd_direct_vs_true": struct.get(
                "pocket_sidechain_heavy_rmsd_direct_vs_true", ""
            ),
            "pocket_chi1_mae_deg": struct.get("pocket_chi1_mae_deg", ""),
            "ligand_clash_pairs_lt_2A": struct.get("ligand_clash_pairs_lt_cutoff", ""),
            "pocket_grid_shape_jaccard_vs_true": struct.get("pocket_grid_shape_jaccard_vs_true", ""),
            "pocket_ca_delta_cosine_vs_raw_to_true": struct.get("pocket_ca_delta_cosine_mean_vs_raw_to_true", ""),
            "pocket_ca_delta_projection_vs_raw_to_true": struct.get(
                "pocket_ca_delta_projection_mean_vs_raw_to_true", ""
            ),
        }
        if raw_pose is not None:
            for metric, out_key in [("top1_rmsd", "delta_top1_rmsd_vs_raw"), ("best_rmsd", "delta_best_rmsd_vs_raw")]:
                current = as_float(pose.get(metric))
                raw = as_float(raw_pose.get(metric))
                if current is not None and raw is not None:
                    row[out_key] = current - raw
        current_pocket = as_float(struct.get("pocket_ca_rmsd_direct_vs_true"))
        raw_pocket = as_float(raw_struct.get("pocket_ca_rmsd_direct_vs_true"))
        if current_pocket is not None and raw_pocket is not None:
            row["delta_pocket_ca_rmsd_vs_raw"] = current_pocket - raw_pocket
        rows.append(row)
    return rows


def experiment_matrix() -> list[dict[str, str]]:
    return [
        {
            "module": "A. Apo-to-holo conformational prior",
            "claim_tested": "HoloShift predicts holo-like pocket/backbone displacement, even if docking does not improve.",
            "required_data": ">=100 apo/holo/ligand complexes; experimental apo when possible plus AF2 apo; train/test split by protein family.",
            "arms": "apo/raw AF2; AF2+OpenMM; HoloShift unrelaxed; HoloShift+relax; scale ensemble; NMA/random baselines; true holo upper bound.",
            "metrics": "global/pocket CA RMSD; DeltaRMSD=RMSD_apo-RMSD_pred; percent improved; motion cosine/projection/norm ratio; bootstrap CI and paired Wilcoxon.",
            "publishable_threshold": "Median Delta pocket-CA RMSD > 0 with 95% CI excluding 0, or >=55-60% paired pocket improvement over apo/AF2.",
            "pilot_status": "Implemented for 5S8I_2LY in binding_readiness metrics.",
        },
        {
            "module": "B. Pocket readiness and failure diagnosis",
            "claim_tested": "Docking failure is explained by side-chain/pocket-readiness bottlenecks, not only backbone RMSD.",
            "required_data": "Same complexes, with crystal ligands and curated pocket residues.",
            "arms": "All A arms plus side-chain repack variants and failed-prep accounting.",
            "metrics": "side-chain heavy RMSD; chi1/chi2 MAE; changed-rotamer recovery; ligand clash/overlap; free-volume Jaccard; Meeko/Vina prep failures; OpenMM energy.",
            "publishable_threshold": "Show statistically significant association between readiness metrics and docking success/failure; identify failure modes by regime.",
            "pilot_status": "Implemented for 5S8I_2LY; HoloShift backbone shift is tiny and side-chain/readiness is not rescued.",
        },
        {
            "module": "C. Docking as diagnostic, not primary success claim",
            "claim_tested": "Generated conformations are not sufficient for one-shot pose prediction unless pocket and side chains are also fixed.",
            "required_data": "Only targets where true-holo redocking sanity gate passes; ideally >=50 retained complexes after gate.",
            "arms": "Five-way receptors plus side-chain repack/ensemble variants; true holo as upper bound.",
            "metrics": "Top-1/3/5/20 pose RMSD success; best-in-topN; first-hit rank; score-RMSD discordance; paired bootstrap/Wilcoxon.",
            "publishable_threshold": "Do not claim docking improvement unless Top-N success improves significantly over AF2/apo with true-holo gate passing.",
            "pilot_status": "Implemented; 5S8I has no non-holo <2A hit, so supports diagnostic/negative claim.",
        },
        {
            "module": "D. Cryptic/pocket-discovery downstream",
            "claim_tested": "A holo-like prior can make pocket-detection tools more holo-consistent even when docking fails.",
            "required_data": "Cryptic-pocket benchmark with ligand-defined positives, e.g. PocketMiner validation/test style labels.",
            "arms": "apo/AF2, HoloShift unrelaxed, HoloShift ensemble, holo.",
            "metrics": "PocketMiner/fpocket/PUResNet residue PR-AUC/ROC-AUC; ligand-contact residue F1; predicted-pocket volume overlap.",
            "publishable_threshold": "Improved PR-AUC/ROC-AUC or contact F1 over apo/AF2 with paired CIs; visualize representative successes/failures.",
            "pilot_status": "Not runnable from current single ligand without pocket-prediction panel; workflow specified.",
        },
        {
            "module": "E. Virtual screening / enrichment",
            "claim_tested": "Receptor conformations affect ranking of actives vs decoys; useful only if pose/VS signals survive.",
            "required_data": "DUD-E/LIT-PCBA/PDBBind-like actives+decoys mapped to each receptor, not just one ligand.",
            "arms": "Five-way receptors; optionally ensemble docking with non-oracle receptor selection.",
            "metrics": "ROC-AUC; PR-AUC; EF1/EF5; BEDROC; paired target-level deltas; active pose sanity when crystal pose exists.",
            "publishable_threshold": "Significant target-level enrichment improvement; otherwise report unavailable/negative.",
            "pilot_status": "Unavailable for 5S8I_2LY because only one ligand/no decoys.",
        },
    ]


def metric_dictionary() -> list[dict[str, str]]:
    return [
        {"metric": "Delta pocket CA RMSD", "definition": "pocket_CA_RMSD(raw/apo,true_holo) - pocket_CA_RMSD(pred,true_holo)", "interpretation": ">0 means conformational prior moved pocket closer to holo."},
        {"metric": "Motion cosine", "definition": "cosine(predicted_CA_delta, raw_to_true_CA_delta) over pocket residues", "interpretation": ">0 means directionally holo-like; near 0 means noise; <0 means harmful direction."},
        {"metric": "Projection ratio", "definition": "dot(pred_delta,true_delta)/||true_delta||^2", "interpretation": "0=no captured motion, 1=correct magnitude, >1=overshoot."},
        {"metric": "Pocket grid Jaccard", "definition": "Jaccard of free/blocked grid points in ligand shell vs true holo", "interpretation": "Measures ligand-accommodation shape similarity."},
        {"metric": "Changed-rotamer recovery", "definition": "fraction of apo-holo changed chi angles recovered within tolerance", "interpretation": "Tests whether side-chain bottleneck is fixed."},
        {"metric": "Top-N pose success", "definition": "any docked pose in top N with ligand heavy-atom RMSD <2A", "interpretation": "Primary docking diagnostic; only valid when true-holo redocking gate passes."},
        {"metric": "First-hit rank", "definition": "rank of first <2A pose", "interpretation": "Separates sampling success from scoring failure."},
        {"metric": "VS enrichment", "definition": "EF1/EF5/BEDROC/ROC-AUC on active-decoy panel", "interpretation": "Only claim screening value with multi-ligand panels."},
    ]


def dataset_manifest_template() -> list[dict[str, str]]:
    return [
        {
            "target_id": "example_target",
            "protein_family": "kinase_or_other_split_group",
            "split": "train_or_val_or_test",
            "source": "PDBBind/D3PM/Apo-Holo/PocketMiner/custom",
            "apo_pdb": "/abs/path/apo.pdb",
            "holo_pdb": "/abs/path/holo.pdb",
            "raw_af2_pdb": "/abs/path/af2.pdb",
            "ligand_sdf": "/abs/path/ligand.sdf",
            "reference_ligand_sdf": "/abs/path/crystal_ligand.sdf",
            "apo_chain": "A",
            "holo_chain": "A",
            "uniprot_id": "P00000",
            "pocket_residue_source": "ligand_6A_or_curated",
            "resolution_A": "1.8",
            "notes": "strictly no homolog leakage across splits",
        }
    ]


def write_report(path: Path, pilot_rows: list[dict[str, Any]]) -> None:
    raw = next(row for row in pilot_rows if row["arm"] == "raw_af2")
    hs_unrelaxed = next(row for row in pilot_rows if row["arm"] == "holoshift_unrelaxed")
    hs_relax = next(row for row in pilot_rows if row["arm"] == "holoshift_openmm_relax")
    true_holo = next(row for row in pilot_rows if row["arm"] == "true_holo")

    lines = [
        "# Publication Diagnostic Benchmark Package",
        "",
        "## Thesis",
        "",
        "Do not frame HoloShift as a one-shot docking improver yet. Frame it as a ligand-agnostic conformational prior and a diagnostic benchmark for why small receptor perturbations fail to translate into docking gains.",
        "",
        "The publishable unit is a paired benchmark that separates three questions: did the backbone move toward holo, did the pocket become ligand-ready, and did docking/scoring benefit after a true-holo sanity gate?",
        "",
        "## Sesame Paper Takeaways",
        "",
        "- Sesame evaluates apo-to-holo generation first with RMSD and an improvement statistic where positive values mean the generated structure is closer to holo than apo.",
        "- It adds pocket-specific and cryptic-pocket diagnostics, not just docking.",
        "- Its docking section reconstructs side chains under a matched protocol for apo, holo, and model structures before docking, which is directly relevant to our side-chain failure mode.",
        "- It explicitly names side chains as a limitation and future direction; our current result fits that narrative but should be made more rigorous and less anecdotal.",
        "",
        "## Current 5S8I Pilot",
        "",
        "| Arm | Top-1 RMSD | Best RMSD | Best rank | Top-20 | pocket CA RMSD | side-chain RMSD | shape Jaccard | clashes <2A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pilot_rows:
        lines.append(
            "| "
            f"{row['display_name']} | "
            f"{fmt(row['top1_rmsd'])} | "
            f"{fmt(row['best_rmsd'])} | "
            f"{row['best_rank']} | "
            f"{row['top20_success']} | "
            f"{fmt(row['pocket_ca_rmsd_direct_vs_true'])} | "
            f"{fmt(row['pocket_sidechain_heavy_rmsd_direct_vs_true'])} | "
            f"{fmt(row['pocket_grid_shape_jaccard_vs_true'])} | "
            f"{row['ligand_clash_pairs_lt_2A']} |"
        )

    lines += [
        "",
        "Pilot interpretation:",
        "",
        f"- HoloShift unrelaxed best RMSD is {fmt(hs_unrelaxed['best_rmsd'])} vs raw AF2 {fmt(raw['best_rmsd'])}, but Top-20 success remains {hs_unrelaxed['top20_success']}.",
        f"- HoloShift + OpenMM best RMSD is {fmt(hs_relax['best_rmsd'])}; relaxation does not rescue docking.",
        f"- True holo reaches best RMSD {fmt(true_holo['best_rmsd'])}, so the docking stack has a positive control but non-holo receptors do not cross the 2A threshold.",
        "",
        "## Publishable Experiment Workflow",
        "",
        "1. Curate a paired apo/holo/ligand manifest with family-level splits and ligand-defined pockets.",
        "2. Generate five-way receptors for every target: raw AF2 or apo, AF2/OpenMM, HoloShift unrelaxed, HoloShift/OpenMM, true holo.",
        "3. Add diagnostic baselines: NMA/random displacement, HoloShift scale ensemble, CA-restrained side-chain repack, and matched side-chain reconstruction for all arms.",
        "4. Compute conformational-prior metrics before docking: global/pocket CA RMSD, DeltaRMSD, motion cosine/projection, pocket-grid Jaccard, clash/volume, chi recovery, structure-quality failures.",
        "5. Run docking only after true-holo redocking passes the sanity gate; report failures and no-pose cases instead of silently dropping them.",
        "6. For screening claims, add active/decoy panels and compute EF1/EF5/BEDROC/ROC-AUC. With one ligand, mark VS unavailable.",
        "7. Stratify by motion regime, pocket openness, ligand buriedness, AF2 confidence/PAE, and side-chain-change burden.",
        "8. Use paired bootstrap CIs and Wilcoxon/sign tests at target level. Show per-target raincloud/scatter plots, not only means.",
        "",
        "## Decision Rules",
        "",
        "- Conformational-prior claim: allowed if paired pocket DeltaRMSD is positive with confidence intervals and motion direction/projection beats NMA/random controls.",
        "- Docking-improvement claim: allowed only if Top-N pose success improves significantly over raw AF2/apo on targets whose true-holo redocking passes.",
        "- Diagnostic/negative claim: allowed if backbone improvements fail to translate into docking and the failure is localized to side-chain/readiness/scoring bottlenecks.",
        "",
        "## Data Products To Release",
        "",
        "- `dataset_manifest_template.csv`: required schema for the multi-target benchmark.",
        "- `experiment_matrix.csv`: claim-to-metric matrix and publishable thresholds.",
        "- `metric_dictionary.csv`: metric definitions and interpretation.",
        "- `pilot_core_metrics.csv`: current 5S8I pilot values.",
        "- Per-target receptor PDBs, docking poses, failures, and command logs for reproducibility.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    five_way_dir = args.five_way_dir.resolve()
    binding_dir = args.binding_readiness_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows = build_pilot_metrics(five_way_dir, binding_dir)
    write_csv(out_dir / "pilot_core_metrics.csv", pilot_rows)
    write_csv(out_dir / "experiment_matrix.csv", experiment_matrix())
    write_csv(out_dir / "metric_dictionary.csv", metric_dictionary())
    write_csv(out_dir / "dataset_manifest_template.csv", dataset_manifest_template())
    write_report(out_dir / "publication_diagnostic_report.md", pilot_rows)
    print(f"Publication diagnostic report: {out_dir / 'publication_diagnostic_report.md'}")
    print(f"Pilot metrics: {out_dir / 'pilot_core_metrics.csv'}")


if __name__ == "__main__":
    main()
