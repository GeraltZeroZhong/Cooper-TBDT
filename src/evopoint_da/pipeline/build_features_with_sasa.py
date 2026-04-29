"""Build strict TBDT GVP graph features from processed displacement pairs."""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from evopoint_da.data.features import (
    ESMFeatureExtractor,
    PCAReducer,
    compute_structural_node_features,
)
from evopoint_da.data.graph import (
    build_gvp_graph_features,
    build_knn_edges,
    parse_pae_matrix_for_indices,
    parse_pae_matrix_for_residue_ids,
)

PLDDT_SCALE_MAX = 100.0
SASA_SCALE_MAX = 250.0
AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
TBDT_OPTIONAL_KEYS = (
    "family_id",
    "state_id",
    "substrate_id",
    "region_id",
    "barrel_core_mask",
    "plug_mask",
    "extracellular_loop_mask",
    "tonb_box_mask",
    "substrate_contact_mask",
    "eval_mask",
    "loss_weight",
    "metadata",
)
TBDT_NODE_KEYS = {
    "region_id",
    "barrel_core_mask",
    "plug_mask",
    "extracellular_loop_mask",
    "tonb_box_mask",
    "substrate_contact_mask",
    "eval_mask",
    "loss_weight",
}
AFDB_MODEL_RE = re.compile(r"^AF-(?P<uniprot>.+?)-F\d+-model_v\d+$")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add ESM/PCA + pLDDT + SASA + KNN/PAE edge features.")
    p.add_argument("--pair_dir", default="data/processed_pairs")
    p.add_argument("--output_dir", default="data/processed_graphs")
    p.add_argument("--esm_weights")
    p.add_argument("--pca_path", default="data/pca_esmc_128.pkl")
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument(
        "--smoke-test-features",
        action="store_true",
        help="Use deterministic lightweight sequence features instead of ESM/PCA. For smoke tests only.",
    )
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--fit_pca", action="store_true")
    p.add_argument("--pae_dir", default="data/raw_af2")
    p.add_argument(
        "--allow-missing-pae",
        action="store_true",
        help=(
            "Allow missing PAE by filling the PAE edge channel with zeros. "
            "Use only when the missing-PAE count is reported as a limitation."
        ),
    )
    p.add_argument("--af2_structure_dir", default="data/raw_af2")
    p.add_argument("--contact_radius", type=float, default=10.0)
    p.add_argument("--surface_sasa_threshold", type=float, default=1.0)
    p.add_argument("--report_path", default="artifacts/build_features_report.json")
    return p.parse_args()


def build_uniprot_to_af2_path(af2_dir: str) -> dict[str, str]:
    uniprot_to_path = {}
    for af2_path in glob.glob(os.path.join(af2_dir, "*.pdb")):
        stem = os.path.splitext(os.path.basename(af2_path))[0]
        match = AFDB_MODEL_RE.match(stem)
        if match:
            uniprot_id = match.group("uniprot")
        else:
            uniprot_id = stem[3:] if stem.startswith("AF-") else stem
        uniprot_to_path[uniprot_id] = af2_path
    return uniprot_to_path


def _resolve_pae_path(pae_dir: str, uniprot_id: str | None, pair_stem: str) -> str | None:
    """Prefer AF-<UNIPROT> naming, then pair_id naming."""
    candidates: list[str] = []
    if uniprot_id:
        uniprot_variants = []
        for value in (uniprot_id, uniprot_id.upper()):
            if value and value not in uniprot_variants:
                uniprot_variants.append(value)
        for value in uniprot_variants:
            candidates.extend([
                os.path.join(pae_dir, f"AF-{value}.npy"),
                os.path.join(pae_dir, f"AF-{value}.json"),
                os.path.join(pae_dir, f"{value}.npy"),
                os.path.join(pae_dir, f"{value}.json"),
            ])

    candidates.extend([
        os.path.join(pae_dir, f"{pair_stem}.npy"),
        os.path.join(pae_dir, f"{pair_stem}.json"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _metadata_af2_path(sample: dict) -> str | None:
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    manifest_row = metadata.get("manifest_row", {})
    raw_paths = [metadata.get("af2_pdb")]
    if isinstance(manifest_row, dict):
        raw_paths.append(manifest_row.get("af2_pdb"))
    for raw_path in raw_paths:
        if not raw_path:
            continue
        path = str(raw_path)
        if os.path.isabs(path):
            candidates = [path]
        else:
            candidates = [
                os.path.abspath(path),
                os.path.abspath(os.path.join("data", path)),
            ]
        for resolved in candidates:
            if os.path.exists(resolved):
                return resolved
    return None


def _metadata_uniprot_id(sample: dict) -> str | None:
    value = sample.get("uniprot_id")
    if value:
        return str(value).strip()
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("uniprot_id")
        if value:
            return str(value).strip()
        manifest_row = metadata.get("manifest_row", {})
        if isinstance(manifest_row, dict):
            value = manifest_row.get("uniprot_id")
            if value:
                return str(value).strip()
    return None


def _invalidate_dataset_cache(output_dir: str) -> int:
    processed_dir = os.path.join(output_dir, "processed")
    removed = 0
    for cache_path in glob.glob(os.path.join(processed_dir, "graph_cache_*.pt")):
        os.remove(cache_path)
        removed += 1
    return removed


def _copy_optional_fields(sample: dict, n_nodes: int) -> dict:
    copied = {}
    for key in TBDT_OPTIONAL_KEYS:
        if key not in sample:
            continue
        value = sample[key]
        if key in TBDT_NODE_KEYS:
            value_t = torch.as_tensor(value)
            if value_t.dim() > 0 and value_t.size(0) != n_nodes:
                value_t = value_t[:n_nodes]
            copied[key] = value_t
        else:
            copied[key] = value
    return copied


def _smoke_sequence_features(sequence: str, dim: int) -> torch.Tensor:
    """Deterministic 128-dim stand-in for ESM/PCA, reserved for smoke tests."""
    n = len(sequence)
    x = torch.zeros((n, dim), dtype=torch.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    if n == 0 or dim == 0:
        return x

    pos_base = min(len(AA_ORDER), dim)
    for i, aa in enumerate(sequence):
        x[i, aa_to_idx.get(aa.upper(), aa_to_idx["X"]) % dim] = 1.0
        if dim > pos_base:
            x[i, pos_base + (i % (dim - pos_base))] = 1.0
        if dim >= 2:
            phase = float(i) / max(1.0, float(n - 1))
            x[i, -2] = np.sin(2.0 * np.pi * phase)
            x[i, -1] = np.cos(2.0 * np.pi * phase)
    return x


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pair_dir, "*.pt")))
    if args.smoke_test_features and args.fit_pca:
        raise ValueError("--smoke-test-features cannot be combined with --fit_pca.")
    if not args.smoke_test_features and not args.esm_weights:
        raise ValueError("--esm_weights is required unless --smoke-test-features is set.")

    extractor = None if args.smoke_test_features else ESMFeatureExtractor(model_path=args.esm_weights)
    pca = None if args.smoke_test_features else PCAReducer(n_components=args.pca_dim)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_structure_dir)
    af2_casefold = {k.lower(): v for k, v in uniprot_to_af2_path.items()}

    if args.fit_pca:
        buf = []
        for f in tqdm(files, desc="Fitting PCA", unit="file"):
            d = torch.load(f, weights_only=True)
            if extractor is None:
                raise RuntimeError("Internal error: smoke-test feature mode should not fit PCA.")
            buf.append(extractor.extract_residue_embeddings(d["sequence"]))
        assert pca is not None
        pca.fit(buf)
        pca.save(args.pca_path)
    elif pca is not None:
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
        if args.smoke_test_features:
            x_esm = _smoke_sequence_features(d["sequence"], args.pca_dim)
        else:
            assert extractor is not None and pca is not None
            emb = extractor.extract_residue_embeddings(d["sequence"])
            x_esm = pca.transform(emb)

        uniprot_id = _metadata_uniprot_id(d)
        af2_file = _metadata_af2_path(d)
        if uniprot_id:
            af2_file = af2_file or uniprot_to_af2_path.get(uniprot_id) or af2_casefold.get(uniprot_id.lower())

        if not af2_file or not os.path.exists(af2_file):
            print(f"[debug] Skip {stem}: AF2 structure not found from sample metadata or AFDB directory")
            skip_reasons["missing_af2_structure"] += 1
            continue

        structural = compute_structural_node_features(
            af2_file,
            d["residue_ids"],
            neighbor_radius=args.contact_radius,
            surface_sasa_threshold=args.surface_sasa_threshold,
            require_residue_match=True,
        )

        plddt_raw = d["plddt"].float()
        expected_n = int(x_esm.size(0))
        lengths = {
            "plddt": int(plddt_raw.size(0)),
            "af2_pos": int(d["af2_pos"].size(0)),
            "y_delta": int(d["y_delta"].size(0)),
            "residue_ids": len(d["residue_ids"]),
        }
        mismatched = {key: value for key, value in lengths.items() if value != expected_n}
        if mismatched:
            raise ValueError(f"{stem} has inconsistent node counts: sequence={expected_n}, mismatched={mismatched}")

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
            if not args.allow_missing_pae:
                raise FileNotFoundError(
                    f"Missing PAE for {stem}. Provide PAE files in {args.pae_dir} or pass "
                    "--allow-missing-pae to explicitly use zero PAE edge features."
                )
            skip_reasons["missing_pae_zero_edge_feature"] += 1
        strict_pae = not bool(args.allow_missing_pae)
        if "af2_indices" in d:
            pae = parse_pae_matrix_for_indices(
                pae_path,
                [int(i) for i in d["af2_indices"].tolist()],
                strict=strict_pae,
            )
        else:
            pae = parse_pae_matrix_for_residue_ids(
                pae_path,
                list(d["residue_ids"]),
                strict=strict_pae,
            )
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
        out.update(_copy_optional_fields(d, int(x.size(0))))
        torch.save(out, os.path.join(args.output_dir, f"{stem}.pt"))
        processed_count += 1
        feature_dims.append(int(x.size(1)))
        node_counts.append(int(x.size(0)))
        edge_counts.append(int(edge_index.size(1)))
        y_norm_means.append(float(torch.norm(d["y_delta"].float(), dim=-1).mean().item()))
        plddt_means.append(float(plddt.mean().item()))

    invalidated_dataset_caches = _invalidate_dataset_cache(args.output_dir) if processed_count else 0
    report = {
        "input_pairs": len(files),
        "processed_graphs": processed_count,
        "skipped_pairs": int(len(files) - processed_count),
        "skip_reason_counts": dict(skip_reasons),
        "invalidated_dataset_caches": invalidated_dataset_caches,
        "smoke_test_features": bool(args.smoke_test_features),
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
        "strict_residue_matching": True,
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
