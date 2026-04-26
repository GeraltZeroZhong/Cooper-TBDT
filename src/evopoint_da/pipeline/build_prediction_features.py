"""Build training-compatible graph features for inference-only prediction.

This path intentionally does not consume an AF2/holo pair. It builds the same
node and graph feature schema used during training from the exact PDB chain
that will be passed to ``run_Predict.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evopoint_da.data.features import ESMFeatureExtractor, PCAReducer, compute_structural_node_features
from evopoint_da.data.graph import build_gvp_graph_features, build_knn_edges, parse_pae_matrix_for_residue_ids
from evopoint_da.data.structure import StructureParser, select_chain

PLDDT_SCALE_MAX = 100.0
SASA_SCALE_MAX = 250.0
FEATURE_SCHEMA_VERSION = "holoshift_training_graph_v1"
MAX_ESMC_SEQUENCE_LEN = 1022


@dataclass(frozen=True)
class PredictionFeatureBuildConfig:
    pdb_file: Path
    output_pt: Path
    esm_weights: Path
    pca_path: Path
    pae_path: Path | None = None
    chain_id: str | None = None
    device: str | None = None
    pca_dim: int = 128
    k: int = 16
    contact_radius: float = 10.0
    surface_sasa_threshold: float = 1.0
    require_pae: bool = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_node_features(
    *,
    pdb_file: Path,
    residue_ids: list[str],
    sequence: str,
    plddt_raw: torch.Tensor,
    extractor: ESMFeatureExtractor,
    pca: PCAReducer,
    contact_radius: float,
    surface_sasa_threshold: float,
    require_dssp: bool,
) -> torch.Tensor:
    if len(sequence) > MAX_ESMC_SEQUENCE_LEN:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds the ESMC inference limit "
            f"({MAX_ESMC_SEQUENCE_LEN}); split or crop explicitly before building features."
        )

    emb = extractor.extract_residue_embeddings(sequence)
    x_esm = pca.transform(emb)
    if x_esm.size(0) != len(residue_ids):
        raise ValueError(
            f"ESM/PCA feature length ({x_esm.size(0)}) does not match residue_ids ({len(residue_ids)})."
        )

    structural = compute_structural_node_features(
        str(pdb_file),
        residue_ids,
        neighbor_radius=contact_radius,
        surface_sasa_threshold=surface_sasa_threshold,
        require_dssp=require_dssp,
    )
    plddt = (plddt_raw / PLDDT_SCALE_MAX).clamp(0.0, 1.0)
    sasa = (structural["sasa"].float() / SASA_SCALE_MAX).clamp(0.0, 1.0)
    rsa = structural["rsa"].float().clamp(0.0, 1.0)
    residue_depth = (structural["residue_depth"].float() / 20.0).clamp(0.0, 1.0)
    coordination_number = (structural["coordination_number"].float() / 32.0).clamp(0.0, 1.0)
    hse = (structural["hse"].float() / 32.0).clamp(0.0, 1.0)
    dihedral_sincos = structural["dihedral_sincos"].float()
    dssp_3state = structural["dssp_3state"].float()

    return torch.cat(
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


def build_prediction_feature_graph(cfg: PredictionFeatureBuildConfig) -> dict[str, Any]:
    parser = StructureParser()
    parsed_chains = parser.parse_ca_structure(str(cfg.pdb_file), strict=True)
    if not parsed_chains:
        raise ValueError(f"No usable protein chain found in {cfg.pdb_file}")
    selected_chain_id, parsed = select_chain(parsed_chains, cfg.chain_id)

    residue_ids = list(parsed["residue_ids"])
    sequence = str(parsed["sequence"])
    pos = torch.as_tensor(parsed["coords"], dtype=torch.float32)
    plddt_raw = torch.as_tensor(parsed["plddts"], dtype=torch.float32).reshape(-1, 1)

    extractor = ESMFeatureExtractor(str(cfg.esm_weights), device=cfg.device)
    pca = PCAReducer(n_components=cfg.pca_dim)
    pca.load(str(cfg.pca_path))
    x = _build_node_features(
        pdb_file=cfg.pdb_file,
        residue_ids=residue_ids,
        sequence=sequence,
        plddt_raw=plddt_raw,
        extractor=extractor,
        pca=pca,
        contact_radius=cfg.contact_radius,
        surface_sasa_threshold=cfg.surface_sasa_threshold,
        require_dssp=True,
    )

    if x.size(1) != 144:
        raise ValueError(f"Training feature schema expected 144 node channels, got {x.size(1)}.")

    pae_path = cfg.pae_path
    if cfg.require_pae and (pae_path is None or not pae_path.exists()):
        raise FileNotFoundError(f"PAE is required but was not found: {pae_path}")
    pae = parse_pae_matrix_for_residue_ids(str(pae_path) if pae_path else None, residue_ids, strict=cfg.require_pae)
    edge_index, edge_attr = build_knn_edges(pos, k=cfg.k, pae=pae)
    node_v, edge_s, edge_v = build_gvp_graph_features(pos, edge_index, edge_attr)

    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "task_definition": "ligand_agnostic_ca_displacement",
        "source_pdb": str(cfg.pdb_file),
        "source_pdb_sha256": _sha256_file(cfg.pdb_file),
        "selected_chain": selected_chain_id,
        "esm_weights": str(cfg.esm_weights),
        "esm_weights_sha256": _sha256_file(cfg.esm_weights),
        "pca_path": str(cfg.pca_path),
        "pca_sha256": _sha256_file(cfg.pca_path),
        "pae_path": str(pae_path) if pae_path else "",
        "pae_sha256": _sha256_file(pae_path) if pae_path and pae_path.exists() else "",
        "pae_required": bool(cfg.require_pae),
        "k": int(cfg.k),
        "contact_radius": float(cfg.contact_radius),
        "surface_sasa_threshold": float(cfg.surface_sasa_threshold),
        "dssp_required": True,
        "feature_dim": int(x.size(1)),
        "n_nodes": int(x.size(0)),
        "dssp_3state_order": ["helix", "strand", "coil"],
        "dssp_3state_counts": [float(v) for v in x[:, -3:].sum(dim=0).tolist()],
    }

    payload = {
        "x": x,
        "node_v": node_v,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_s": edge_s,
        "edge_v": edge_v,
        "plddt": plddt_raw,
        "residue_ids": residue_ids,
        "sequence": sequence,
        "metadata": metadata,
    }
    cfg.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cfg.output_pt)
    return {"output_pt": str(cfg.output_pt), **metadata}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build training-compatible HoloShift prediction features.")
    p.add_argument("--pdb-file", required=True, type=Path)
    p.add_argument("--output-pt", required=True, type=Path)
    p.add_argument("--esm-weights", required=True, type=Path)
    p.add_argument("--pca-path", required=True, type=Path)
    p.add_argument("--pae-path", type=Path, default=None)
    p.add_argument("--chain-id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--contact-radius", type=float, default=10.0)
    p.add_argument("--surface-sasa-threshold", type=float, default=1.0)
    p.add_argument("--require-pae", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = build_prediction_feature_graph(
        PredictionFeatureBuildConfig(
            pdb_file=args.pdb_file,
            output_pt=args.output_pt,
            esm_weights=args.esm_weights,
            pca_path=args.pca_path,
            pae_path=args.pae_path,
            chain_id=args.chain_id,
            device=args.device,
            pca_dim=args.pca_dim,
            k=args.k,
            contact_radius=args.contact_radius,
            surface_sasa_threshold=args.surface_sasa_threshold,
            require_pae=args.require_pae,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
