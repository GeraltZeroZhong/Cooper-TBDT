"""Build graph features (ESM+PCA+structure+edge features) from processed pairs.

Moved from scripts/ into src/ so the data pipeline code is colocated with project modules.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from evopoint_da.data.features import (
    ESMFeatureExtractor,
    PCAReducer,
    compute_structural_node_features,
)
from evopoint_da.data.graph import build_gvp_graph_features, build_knn_edges, parse_pae_matrix
from evopoint_da.data.paths import DEFAULT_PDB_UNIPROT_MAPPING

PLDDT_SCALE_MAX = 100.0
SASA_SCALE_MAX = 250.0


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add ESM/PCA + pLDDT + SASA + KNN/PAE edge features.")
    p.add_argument("--pair_dir", default="data/processed_pairs")
    p.add_argument("--output_dir", default="data/processed_graphs")
    p.add_argument("--esm_weights", required=True)
    p.add_argument("--pca_path", default="data/pca_esmc_128.pkl")
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--fit_pca", action="store_true")
    p.add_argument("--pae_dir", default="data/raw_af2")
    p.add_argument("--af2_structure_dir", default="data/raw_af2")
    p.add_argument("--mapping_file", default=DEFAULT_PDB_UNIPROT_MAPPING)
    p.add_argument("--contact_radius", type=float, default=10.0)
    p.add_argument("--surface_sasa_threshold", type=float, default=1.0)
    p.add_argument("--report_path", default="artifacts/build_features_report.json")
    return p.parse_args()


def load_pdb_to_uniprot_mapping(mapping_file: str) -> dict[str, str]:
    with open(mapping_file, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    return {pdb_id.upper(): uniprot_id for pdb_id, uniprot_id in raw_mapping.items()}


def build_uniprot_to_af2_path(af2_dir: str) -> dict[str, str]:
    uniprot_to_path = {}
    for af2_path in glob.glob(os.path.join(af2_dir, "*.pdb")):
        stem = os.path.splitext(os.path.basename(af2_path))[0]
        uniprot_id = stem[3:] if stem.startswith("AF-") else stem
        uniprot_to_path[uniprot_id] = af2_path
    return uniprot_to_path


def _resolve_pae_path(pae_dir: str, uniprot_id: str | None, pair_stem: str) -> str | None:
    """Prefer AF-<UNIPROT> naming (from get_af2.py), fallback to pair_id naming."""
    candidates: list[str] = []
    if uniprot_id:
        candidates.extend([
            os.path.join(pae_dir, f"AF-{uniprot_id}.npy"),
            os.path.join(pae_dir, f"AF-{uniprot_id}.json"),
            os.path.join(pae_dir, f"{uniprot_id}.npy"),
            os.path.join(pae_dir, f"{uniprot_id}.json"),
        ])

    candidates.extend([
        os.path.join(pae_dir, f"{pair_stem}.npy"),
        os.path.join(pae_dir, f"{pair_stem}.json"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pair_dir, "*.pt")))
    extractor = ESMFeatureExtractor(model_path=args.esm_weights)
    pca = PCAReducer(n_components=args.pca_dim)
    pdb_to_uniprot = load_pdb_to_uniprot_mapping(args.mapping_file)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_structure_dir)
    af2_casefold = {k.lower(): v for k, v in uniprot_to_af2_path.items()}

    if args.fit_pca:
        buf = []
        for f in tqdm(files, desc="Fitting PCA", unit="file"):
            d = torch.load(f, weights_only=True)
            buf.append(extractor.extract_residue_embeddings(d["sequence"]))
        pca.fit(buf)
        pca.save(args.pca_path)
    else:
        pca.load(args.pca_path)

    skip_reasons: Counter[str] = Counter()
    feature_dims: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    y_norm_means: list[float] = []
    plddt_means: list[float] = []
    processed_count = 0
    for f in tqdm(files, desc="Building graph features", unit="file"):
        d = torch.load(f, weights_only=True)
        stem = d["pair_id"]
        emb = extractor.extract_residue_embeddings(d["sequence"])
        x_esm = pca.transform(emb)

        uniprot_id = pdb_to_uniprot.get(stem.upper())
        af2_file = None
        if uniprot_id:
            af2_file = uniprot_to_af2_path.get(uniprot_id) or af2_casefold.get(uniprot_id.lower())

        if not af2_file:
            print(f"[debug] Skip {stem}: AF2 structure not found via mapping file")
            skip_reasons["missing_af2_structure"] += 1
            continue

        structural = compute_structural_node_features(
            af2_file,
            d["residue_ids"],
            neighbor_radius=args.contact_radius,
            surface_sasa_threshold=args.surface_sasa_threshold,
        )

        plddt_raw = d["plddt"].float()
        if plddt_raw.size(0) != x_esm.size(0):
            min_len = min(int(plddt_raw.size(0)), int(x_esm.size(0)))
            if min_len <= 0:
                skip_reasons["invalid_zero_length"] += 1
                continue
            plddt_raw = plddt_raw[:min_len]
            x_esm = x_esm[:min_len]
            structural = {k: v[:min_len] for k, v in structural.items()}
            d["af2_pos"] = d["af2_pos"][:min_len]
            d["y_delta"] = d["y_delta"][:min_len]
            d["residue_ids"] = d["residue_ids"][:min_len]
            skip_reasons["length_mismatch_truncated"] += 1

        sasa = (structural["sasa"].float() / SASA_SCALE_MAX).clamp(0.0, 1.0)
        rsa = structural["rsa"].float().clamp(0.0, 1.0)
        residue_depth = (structural["residue_depth"].float() / 20.0).clamp(0.0, 1.0)
        coordination_number = (structural["coordination_number"].float() / 32.0).clamp(0.0, 1.0)
        hse = (structural["hse"].float() / 32.0).clamp(0.0, 1.0)
        dihedral_sincos = structural["dihedral_sincos"].float()
        dssp_3state = structural["dssp_3state"].float()
        plddt = (plddt_raw / PLDDT_SCALE_MAX).clamp(0.0, 1.0)
        x = torch.cat(
            [
                x_esm,
                plddt,
                sasa,
                rsa,
                residue_depth,
                coordination_number,
                hse,
                dihedral_sincos,
                dssp_3state,
            ],
            dim=1,
        )

        pae_path = _resolve_pae_path(args.pae_dir, uniprot_id, stem)
        if pae_path is None:
            skip_reasons["missing_pae_fallback_zero"] += 1
        pae = parse_pae_matrix(pae_path, len(d["residue_ids"]))
        edge_index, edge_attr = build_knn_edges(d["af2_pos"].float(), k=args.k, pae=pae)
        node_v, edge_s, edge_v = build_gvp_graph_features(d["af2_pos"].float(), edge_index, edge_attr)

        out = {
            "pair_id": stem,
            "residue_ids": d["residue_ids"],
            "x": x,
            "node_v": node_v,
            "pos": d["af2_pos"].float(),
            "y_delta": d["y_delta"].float(),
            "plddt": plddt_raw,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_s": edge_s,
            "edge_v": edge_v,
        }
        torch.save(out, os.path.join(args.output_dir, f"{stem}.pt"))
        processed_count += 1
        feature_dims.append(int(x.size(1)))
        node_counts.append(int(x.size(0)))
        edge_counts.append(int(edge_index.size(1)))
        y_norm_means.append(float(torch.norm(d["y_delta"].float(), dim=-1).mean().item()))
        plddt_means.append(float(plddt.mean().item()))

    report = {
        "input_pairs": len(files),
        "processed_graphs": processed_count,
        "skipped_pairs": int(len(files) - processed_count),
        "skip_reason_counts": dict(skip_reasons),
        "feature_dim_unique": sorted(set(feature_dims)),
        "node_count_stats": {
            "mean": float(np.mean(node_counts)) if node_counts else None,
            "median": float(np.median(node_counts)) if node_counts else None,
            "p90": float(np.percentile(node_counts, 90)) if node_counts else None,
        },
        "edge_count_stats": {
            "mean": float(np.mean(edge_counts)) if edge_counts else None,
            "median": float(np.median(edge_counts)) if edge_counts else None,
            "p90": float(np.percentile(edge_counts, 90)) if edge_counts else None,
        },
        "mean_target_disp_stats": {
            "mean": float(np.mean(y_norm_means)) if y_norm_means else None,
            "median": float(np.median(y_norm_means)) if y_norm_means else None,
        },
        "plddt_stats_normalized": {
            "mean": float(np.mean(plddt_means)) if plddt_means else None,
            "median": float(np.median(plddt_means)) if plddt_means else None,
        },
    }
    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Processed {processed_count}/{len(files)} graph files into {args.output_dir}")


if __name__ == "__main__":
    main()
