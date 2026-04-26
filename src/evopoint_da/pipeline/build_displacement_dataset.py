"""Build paired AF2/Holo displacement targets.

Moved from scripts/ into src/ so it can be imported by other pipeline steps.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import torch

from evopoint_da.data.alignment import compute_displacement_target
from evopoint_da.data.paths import DEFAULT_PDB_UNIPROT_MAPPING
from evopoint_da.data.structure import StructureParser


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Δr targets from AF2/Holo structure pairs.")
    p.add_argument("--af2_dir", default="data/raw_af2")
    p.add_argument("--holo_dir", default="data/raw_pdb")
    p.add_argument("--out_dir", default="data/processed_pairs")
    p.add_argument("--mapping_file", default=DEFAULT_PDB_UNIPROT_MAPPING)
    p.add_argument("--report_path", default="artifacts/build_pairs_report.json")
    return p.parse_args()


def load_pdb_to_uniprot_mapping(mapping_file: str) -> dict[str, str]:
    with open(mapping_file, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    return {pdb_id.upper(): uniprot_id for pdb_id, uniprot_id in raw_mapping.items()}


def build_uniprot_to_af2_path(af2_dir: str) -> dict[str, str]:
    uniprot_to_path: dict[str, str] = {}
    for af2_path in glob.glob(os.path.join(af2_dir, "*.pdb")):
        stem = os.path.splitext(os.path.basename(af2_path))[0]
        uniprot_id = stem[3:] if stem.startswith("AF-") else stem
        uniprot_to_path[uniprot_id] = af2_path
    return uniprot_to_path


def compute_rmsd(delta_r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(delta_r), axis=1))))


def build_case_insensitive_file_index(directory: str, pattern: str = "*.pdb") -> dict[str, str]:
    index = {}
    for path in glob.glob(os.path.join(directory, pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        index[stem.lower()] = path
    return index


def main() -> None:
    args = args_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    parser = StructureParser()

    pdb_to_uniprot = load_pdb_to_uniprot_mapping(args.mapping_file)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_dir)
    af2_casefold = {k.lower(): v for k, v in uniprot_to_af2_path.items()}
    holo_casefold = build_case_insensitive_file_index(args.holo_dir)

    built = 0
    rmsd_values: list[float] = []
    residue_counts: list[int] = []
    skip_reasons: Counter[str] = Counter()
    for pdb_id, uniprot_id in sorted(pdb_to_uniprot.items()):
        af2 = uniprot_to_af2_path.get(uniprot_id) or af2_casefold.get(uniprot_id.lower())
        if not af2:
            skip_reasons["missing_af2"] += 1
            continue

        holo = os.path.join(args.holo_dir, f"{pdb_id}.pdb")
        if not os.path.exists(holo):
            holo = holo_casefold.get(pdb_id.lower())
        if not holo or not os.path.exists(holo):
            skip_reasons["missing_holo"] += 1
            continue

        af2_chains = parser.parse_ca_structure(af2)
        holo_chains = parser.parse_ca_structure(holo)
        if not af2_chains or not holo_chains:
            print(f"[debug] Skip {pdb_id}: parse failed or all chains < 15 AA")
            skip_reasons["parse_failed_or_short_chain"] += 1
            continue

        try:
            delta_r, ids, af2_aligned, af2_idx, _holo_idx, best_af2_chain_id = compute_displacement_target(af2_chains, holo_chains)
        except ValueError as e:
            print(f"[debug] Skip {pdb_id}: {e}")
            skip_reasons[f"align_error:{type(e).__name__}"] += 1
            continue

        selected_af2_data = af2_chains[best_af2_chain_id]
        rmsd = compute_rmsd(delta_r)
        rmsd_values.append(rmsd)
        residue_counts.append(len(ids))

        out = {
            "pair_id": pdb_id,
            "residue_ids": ids,
            "af2_pos": torch.tensor(af2_aligned),
            "holo_pos": torch.tensor(af2_aligned + delta_r),
            "y_delta": torch.tensor(delta_r),
            "plddt": torch.tensor(selected_af2_data["plddts"][af2_idx]).unsqueeze(1),
            "sequence": "".join(selected_af2_data["sequence"][i] for i in af2_idx.tolist()),
        }
        torch.save(out, os.path.join(args.out_dir, f"{pdb_id}.pt"))
        built += 1
        print(f"[debug] Built {pdb_id}: chain {best_af2_chain_id}, res={len(ids)}, rmsd={rmsd:.4f} Å")

    report = {
        "mapping_pairs_total": len(pdb_to_uniprot),
        "built_pairs": built,
        "skipped_pairs": int(len(pdb_to_uniprot) - built),
        "skip_reason_counts": dict(skip_reasons),
        "rmsd_stats": {
            "mean": float(np.mean(rmsd_values)) if rmsd_values else None,
            "median": float(np.median(rmsd_values)) if rmsd_values else None,
            "p90": float(np.percentile(rmsd_values, 90)) if rmsd_values else None,
        },
        "residue_count_stats": {
            "mean": float(np.mean(residue_counts)) if residue_counts else None,
            "median": float(np.median(residue_counts)) if residue_counts else None,
            "p90": float(np.percentile(residue_counts, 90)) if residue_counts else None,
        },
    }
    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Built {built} paired samples.")


if __name__ == "__main__":
    main()
