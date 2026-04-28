#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from run_binding_readiness_benchmark import (
    AtomRecord,
    build_ligand_shell_grid,
    count_ligand_clashes,
    direct_rmsd,
    generate_nma_ensemble,
    jaccard,
    parse_pdb_atoms,
    pocket_free_mask,
    rewrite_pdb_with_residue_shifts,
    write_csv,
)
from run_multi_target_readiness import generate_scaled_holoshift


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HoloShift as a cryptic-pocket conformational prior.")
    p.add_argument("--manifest", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_cryptic_manifest.csv"))
    p.add_argument("--labels-csv", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_residue_labels.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift/holoshift_prior"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-holoshift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/gvp_full_beta1_mse_selection/20260426-111016/best-selection-06-1.0231.ckpt"))
    p.add_argument("--esm-weights", type=Path, default=Path("esmc_weights/esmc_600m_2024_12_v0.pth"))
    p.add_argument("--pca-path", type=Path, default=Path("data/pca_esmc_128.pkl"))
    p.add_argument("--include-nma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-holoshift-scales", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nma-modes", type=int, default=1)
    p.add_argument("--nma-amplitude", type=float, default=0.5)
    p.add_argument("--random-amplitude", type=float, default=0.5)
    p.add_argument("--holoshift-scales", default="0.5,1.0,1.5,2.0")
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--grid-padding", type=float, default=3.0)
    p.add_argument("--grid-shell-cutoff", type=float, default=4.0)
    p.add_argument("--clash-cutoff", type=float, default=2.0)
    p.add_argument("--soft-clash-cutoff", type=float, default=2.5)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def labels_by_target(labels_csv: Path) -> dict[str, np.ndarray]:
    by_target: dict[str, list[tuple[int, int]]] = {}
    for row in read_csv(labels_csv):
        by_target.setdefault(row["target_id"], []).append((int(row["residue_index"]), int(row["label"])))
    return {
        target_id: np.asarray([label for _idx, label in sorted(items)], dtype=int)
        for target_id, items in by_target.items()
    }


def ca_coords_by_order(atoms: list[AtomRecord]) -> tuple[list[tuple[str, int, str]], np.ndarray]:
    keys: list[tuple[str, int, str]] = []
    coords: list[np.ndarray] = []
    seen: set[tuple[str, int, str]] = set()
    for atom in atoms:
        if atom.name != "CA" or atom.residue_key in seen:
            continue
        seen.add(atom.residue_key)
        keys.append(atom.residue_key)
        coords.append(atom.coord)
    return keys, np.asarray(coords, dtype=float)


def ligand_pdb_heavy_coords(path: Path) -> np.ndarray:
    atoms = parse_pdb_atoms(path, atom_records_only=False)
    coords = [atom.coord for atom in atoms if atom.record == "HETATM" and atom.element != "H"]
    if not coords:
        raise ValueError(f"No HETATM heavy atoms found in {path}")
    return np.asarray(coords, dtype=float)


def kabsch_fit(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(mobile) == 0:
        return np.eye(3), np.zeros(3)
    mobile_mean = mobile.mean(axis=0)
    ref_mean = reference.mean(axis=0)
    cov = (mobile - mobile_mean).T @ (reference - ref_mean)
    v, _s, wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, d]) @ wt
    trans = ref_mean - mobile_mean @ rot
    return rot, trans


def apply_transform(coords: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return coords @ rot + trans


def transform_atoms(atoms: list[AtomRecord], rot: np.ndarray, trans: np.ndarray) -> list[AtomRecord]:
    return [replace(atom, coord=apply_transform(atom.coord[None, :], rot, trans)[0]) for atom in atoms]


def mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def selected_rmsd(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if len(a) == 0 or not mask.any():
        return float("nan")
    return direct_rmsd(a[mask], b[mask])


def motion_stats(apo: np.ndarray, variant: np.ndarray, holo: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    cosines: list[float] = []
    projections: list[float] = []
    norms: list[float] = []
    for target_delta, pred_delta in zip(holo[mask] - apo[mask], variant[mask] - apo[mask], strict=False):
        tn = float(np.linalg.norm(target_delta))
        pn = float(np.linalg.norm(pred_delta))
        norms.append(pn)
        if tn > 1e-8 and pn > 1e-8:
            cosines.append(float(np.dot(target_delta, pred_delta) / (tn * pn)))
            projections.append(float(np.dot(pred_delta, target_delta) / (tn * tn)))
    return {
        "motion_cosine_mean": mean_or_nan(cosines),
        "motion_projection_mean": mean_or_nan(projections),
        "motion_norm_mean": mean_or_nan(norms),
    }


def generate_random_perturbation(raw_pdb: Path, out_pdb: Path, *, amplitude: float, seed: int) -> Path:
    atoms = parse_pdb_atoms(raw_pdb, atom_records_only=True)
    residue_keys = sorted({atom.residue_key for atom in atoms if atom.name == "CA"})
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(len(residue_keys), 3))
    norms = np.linalg.norm(vec, axis=1)
    norms[norms == 0] = 1.0
    vec = vec / norms[:, None] * amplitude
    shifts = {key: vec[idx] for idx, key in enumerate(residue_keys)}
    rewrite_pdb_with_residue_shifts(raw_pdb, out_pdb, shifts)
    return out_pdb


def run_holoshift(row: dict[str, str], target_dir: Path, args: argparse.Namespace) -> Path:
    target_id = row["target_id"]
    out_pdb = target_dir / f"{target_id}_holoshift_unrelaxed.pdb"
    report = target_dir / f"{target_id}_holoshift_report.json"
    feature = target_dir / f"{target_id}_training_graph_features.pt"
    if args.reuse and out_pdb.exists() and report.exists():
        return out_pdb
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
    subprocess.run(cmd, check=True)
    return out_pdb


def build_variants(row: dict[str, str], target_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    apo = Path(row["apo_pdb"])
    variants: list[dict[str, Any]] = [{"structure": "apo", "kind": "baseline", "pdb": apo}]
    failures: list[dict[str, Any]] = []
    hs_pdb: Path | None = None
    if args.run_holoshift:
        try:
            hs_pdb = run_holoshift(row, target_dir, args)
            variants.append({"structure": "holoshift_unrelaxed", "kind": "holoshift", "pdb": hs_pdb})
        except Exception as exc:
            failures.append({"target_id": row["target_id"], "stage": "holoshift", "error": str(exc)})
    if args.include_nma:
        for label, path in generate_nma_ensemble(apo, target_dir / "nma", n_modes=args.nma_modes, amplitude=args.nma_amplitude):
            variants.append({"structure": label, "kind": "nma", "pdb": path})
    if args.include_random:
        label = f"random_{args.random_amplitude:.2f}A".replace(".", "p")
        path = generate_random_perturbation(
            apo,
            target_dir / f"{label}.pdb",
            amplitude=args.random_amplitude,
            seed=int.from_bytes(row["target_id"].encode("utf-8")[:4].ljust(4, b"0"), "little"),
        )
        variants.append({"structure": label, "kind": "random", "pdb": path})
    if args.include_holoshift_scales and hs_pdb is not None and hs_pdb.exists():
        for scale in parse_scales(args.holoshift_scales):
            label = f"holoshift_scale_{scale:.2f}".replace(".", "p")
            path = generate_scaled_holoshift(apo, hs_pdb, target_dir / f"{label}.pdb", scale)
            variants.append({"structure": label, "kind": "holoshift_scale", "pdb": path, "scale": scale})
    return variants, failures


def evaluate_target(
    row: dict[str, str],
    labels: np.ndarray,
    target_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    apo_atoms = parse_pdb_atoms(Path(row["apo_pdb"]), chain_id=row.get("apo_chain") or None, atom_records_only=True)
    holo_atoms = parse_pdb_atoms(Path(row["holo_pdb"]), chain_id=row.get("holo_chain") or None, atom_records_only=True)
    ligand_coords = ligand_pdb_heavy_coords(Path(row["holo_ligand_pdb"]))
    _apo_keys, apo_ca = ca_coords_by_order(apo_atoms)
    _holo_keys, holo_ca = ca_coords_by_order(holo_atoms)
    n_align = min(len(apo_ca), len(holo_ca))
    rot, trans = kabsch_fit(apo_ca[:n_align], holo_ca[:n_align])
    apo_aligned_ca = apply_transform(apo_ca[:n_align], rot, trans)
    holo_ca_common = holo_ca[:n_align]
    label_common = labels[:n_align]
    pos_mask = label_common == 1
    neg_mask = label_common == 0
    eval_mask = np.logical_or(pos_mask, neg_mask)

    target_grid = build_ligand_shell_grid(
        ligand_coords,
        spacing=args.grid_spacing,
        padding=args.grid_padding,
        shell_cutoff=args.grid_shell_cutoff,
    )
    true_free = pocket_free_mask(holo_atoms, target_grid, exclusion_cutoff=args.clash_cutoff)
    raw_pos_rmsd = selected_rmsd(apo_aligned_ca, holo_ca_common, pos_mask)
    raw_eval_rmsd = selected_rmsd(apo_aligned_ca, holo_ca_common, eval_mask)
    variants, failures = build_variants(row, target_dir, args)
    metric_rows: list[dict[str, Any]] = []
    for variant in variants:
        atoms = parse_pdb_atoms(Path(variant["pdb"]), chain_id=row.get("apo_chain") or None, atom_records_only=True)
        transformed_atoms = transform_atoms(atoms, rot, trans)
        _keys, variant_ca = ca_coords_by_order(atoms)
        n = min(n_align, len(variant_ca))
        variant_aligned_ca = apply_transform(variant_ca[:n], rot, trans)
        holo_n = holo_ca_common[:n]
        apo_n = apo_aligned_ca[:n]
        labels_n = label_common[:n]
        pos_n = labels_n == 1
        neg_n = labels_n == 0
        eval_n = np.logical_or(pos_n, neg_n)
        free = pocket_free_mask(transformed_atoms, target_grid, exclusion_cutoff=args.clash_cutoff)
        clashes = count_ligand_clashes(
            transformed_atoms,
            ligand_coords,
            cutoff=args.clash_cutoff,
            soft_cutoff=args.soft_clash_cutoff,
        )
        pos_rmsd = selected_rmsd(variant_aligned_ca, holo_n, pos_n)
        eval_rmsd = selected_rmsd(variant_aligned_ca, holo_n, eval_n)
        neg_rmsd = selected_rmsd(variant_aligned_ca, holo_n, neg_n)
        stats = motion_stats(apo_n, variant_aligned_ca, holo_n, pos_n)
        metric_rows.append(
            {
                "target_id": row["target_id"],
                "split": row.get("split", ""),
                "structure": variant["structure"],
                "kind": variant.get("kind", ""),
                "receptor_pdb": str(variant["pdb"]),
                "n_residue_labels": int(len(labels)),
                "n_compared_ca": int(n),
                "n_positive": int(pos_n.sum()),
                "n_negative": int(neg_n.sum()),
                "positive_ca_rmsd_vs_holo": pos_rmsd,
                "delta_positive_ca_rmsd_vs_apo": raw_pos_rmsd - pos_rmsd if not math.isnan(pos_rmsd) else float("nan"),
                "eval_ca_rmsd_vs_holo": eval_rmsd,
                "delta_eval_ca_rmsd_vs_apo": raw_eval_rmsd - eval_rmsd if not math.isnan(eval_rmsd) else float("nan"),
                "negative_ca_rmsd_vs_holo": neg_rmsd,
                "pocket_grid_shape_jaccard_vs_true": jaccard(free, true_free),
                "pocket_grid_false_blocked_fraction_vs_true": (
                    float(np.logical_and(~free, true_free).sum() / true_free.sum()) if true_free.sum() else float("nan")
                ),
                "pocket_grid_false_open_fraction_vs_true": (
                    float(np.logical_and(free, ~true_free).sum() / (~true_free).sum()) if (~true_free).sum() else float("nan")
                ),
                "pocket_grid_free_volume_A3": float(free.sum() * (args.grid_spacing**3)),
                **stats,
                **clashes,
            }
        )
    return metric_rows, failures


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_structure: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_structure.setdefault(row["structure"], []).append(row)
    metrics = [
        "positive_ca_rmsd_vs_holo",
        "delta_positive_ca_rmsd_vs_apo",
        "motion_cosine_mean",
        "motion_projection_mean",
        "pocket_grid_shape_jaccard_vs_true",
        "pocket_grid_false_blocked_fraction_vs_true",
        "ligand_clash_pairs_lt_cutoff",
    ]
    out: list[dict[str, Any]] = []
    for structure, sr in sorted(by_structure.items()):
        row: dict[str, Any] = {"structure": structure, "n_targets": len(sr)}
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in sr if str(item.get(metric, "")) not in {"", "nan"}], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_median"] = float(np.median(vals)) if len(vals) else ""
            row[f"{metric}_q1"] = float(np.quantile(vals, 0.25)) if len(vals) else ""
            row[f"{metric}_q3"] = float(np.quantile(vals, 0.75)) if len(vals) else ""
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        row
        for row in read_csv(args.manifest)
        if row.get("has_cryptic_holo") == "1" and row.get("apo_pdb") and row.get("holo_pdb") and row.get("holo_ligand_pdb")
    ]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    labels = labels_by_target(args.labels_csv)
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in manifest_rows:
        target_dir = args.out_dir / "targets" / row["target_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows, target_failures = evaluate_target(row, labels[row["target_id"]], target_dir, args)
            metric_rows.extend(rows)
            failures.extend(target_failures)
        except Exception as exc:
            failures.append({"target_id": row["target_id"], "stage": "metrics", "error": str(exc)})
    write_csv(args.out_dir / "cryptic_prior_metrics_all.csv", metric_rows)
    write_csv(args.out_dir / "cryptic_prior_summary_by_structure.csv", summarize(metric_rows))
    write_csv(args.out_dir / "cryptic_prior_failures.csv", failures)
    status = {
        "n_targets_requested": len(manifest_rows),
        "n_metric_rows": len(metric_rows),
        "n_failures": len(failures),
        "has_holoshift_rows": any(row.get("structure") == "holoshift_unrelaxed" for row in metric_rows),
        "experimental_bfactor_warning": (
            "PocketMiner apo PDBs are experimental structures; HoloShift feature builder treats CA B-factors as pLDDT, "
            "so HoloShift arms should be interpreted as an out-of-distribution diagnostic unless confidence features are replaced."
        ),
    }
    (args.out_dir / "cryptic_prior_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
