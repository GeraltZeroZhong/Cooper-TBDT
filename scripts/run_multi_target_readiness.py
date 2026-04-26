#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from run_binding_readiness_benchmark import (
    compute_structure_metrics,
    generate_nma_ensemble,
    parse_pdb_atoms,
    rewrite_pdb_with_residue_shifts,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute multi-target pocket/readiness metrics from a receptor manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/negative_diagnostic_benchmark/readiness"))
    p.add_argument("--chain-id", default="A")
    p.add_argument("--pocket-cutoff", type=float, default=6.0)
    p.add_argument("--clash-cutoff", type=float, default=2.0)
    p.add_argument("--soft-clash-cutoff", type=float, default=2.5)
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--grid-padding", type=float, default=3.0)
    p.add_argument("--grid-shell-cutoff", type=float, default=4.0)
    p.add_argument("--include-nma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-holoshift-scales", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nma-modes", type=int, default=1)
    p.add_argument("--nma-amplitude", type=float, default=0.5)
    p.add_argument("--random-amplitude", type=float, default=0.5)
    p.add_argument("--holoshift-scales", default="0.5,1.0,1.5,2.0")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(raw: str, manifest_dir: Path) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def generate_scaled_holoshift(raw_pdb: Path, holoshift_unrelaxed_pdb: Path, out_pdb: Path, scale: float) -> Path:
    raw_atoms = parse_pdb_atoms(raw_pdb, atom_records_only=True)
    holo_atoms = parse_pdb_atoms(holoshift_unrelaxed_pdb, atom_records_only=True)
    raw_ca = {atom.residue_key: atom.coord for atom in raw_atoms if atom.name == "CA"}
    holo_ca = {atom.residue_key: atom.coord for atom in holo_atoms if atom.name == "CA"}
    shifts = {key: scale * (holo_ca[key] - raw_ca[key]) for key in sorted(set(raw_ca) & set(holo_ca))}
    rewrite_pdb_with_residue_shifts(raw_pdb, out_pdb, shifts)
    return out_pdb


def args_for_metrics(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pocket_cutoff=args.pocket_cutoff,
        clash_cutoff=args.clash_cutoff,
        soft_clash_cutoff=args.soft_clash_cutoff,
        grid_spacing=args.grid_spacing,
        grid_padding=args.grid_padding,
        grid_shell_cutoff=args.grid_shell_cutoff,
    )


def build_variants(row: dict[str, str], manifest_dir: Path, out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    target_id = row["target_id"]
    target_variant_dir = out_dir / "generated_variants" / target_id
    target_variant_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    receptor_cols = [
        ("raw_af2", "receptor_raw_af2", "baseline"),
        ("predicted", "receptor_predicted", "baseline_predicted"),
        ("holo_aligned_predicted", "receptor_holo_aligned_predicted", "baseline_predicted"),
        ("esmfold", "receptor_esmfold", "baseline_esmfold"),
        ("holoshift_unrelaxed", "receptor_holoshift_unrelaxed", "holoshift_shift"),
        ("holoshift", "receptor_holoshift", "holoshift_relax"),
        ("true_holo", "receptor_true_holo", "reference"),
    ]
    seen_paths: set[str] = set()
    for label, col, kind in receptor_cols:
        path = resolve_path(row.get(col, ""), manifest_dir)
        if path is not None and path.exists() and str(path) not in seen_paths:
            variants.append({"label": label, "path": path, "kind": kind})
            seen_paths.add(str(path))

    raw = resolve_path(row.get("receptor_raw_af2") or row.get("receptor_predicted", ""), manifest_dir)
    hs_unrelaxed = resolve_path(row.get("receptor_holoshift_unrelaxed", ""), manifest_dir)
    if raw is not None and raw.exists():
        if args.include_nma:
            for label, path in generate_nma_ensemble(
                raw,
                target_variant_dir / "nma_raw",
                n_modes=args.nma_modes,
                amplitude=args.nma_amplitude,
            ):
                variants.append({"label": label, "path": path, "kind": "nma_raw"})
        if args.include_random:
            random_label = f"random_{args.random_amplitude:.2f}A".replace(".", "p")
            path = generate_random_perturbation(
                raw,
                target_variant_dir / f"{random_label}.pdb",
                amplitude=args.random_amplitude,
                seed=int.from_bytes(target_id.encode("utf-8")[:4].ljust(4, b"0"), "little"),
            )
            variants.append({"label": random_label, "path": path, "kind": "random_raw"})
        if args.include_holoshift_scales and hs_unrelaxed is not None and hs_unrelaxed.exists():
            for scale in parse_scales(args.holoshift_scales):
                label = f"holoshift_scale_{scale:.2f}".replace(".", "p")
                path = generate_scaled_holoshift(raw, hs_unrelaxed, target_variant_dir / f"{label}.pdb", scale)
                variants.append({"label": label, "path": path, "kind": "holoshift_scale", "scale": scale})
    return variants


def main() -> None:
    args = parse_args()
    rows = read_csv(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    manifest_dir = args.manifest.resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metric_args = args_for_metrics(args)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        target_id = row.get("target_id", "")
        try:
            raw = resolve_path(row.get("receptor_raw_af2") or row.get("receptor_predicted", ""), manifest_dir)
            true_holo = resolve_path(row.get("receptor_true_holo", ""), manifest_dir)
            ligand = resolve_path(row.get("reference_ligand_sdf") or row.get("ligand_sdf", ""), manifest_dir)
            if raw is None or true_holo is None or ligand is None:
                raise ValueError("missing raw/predicted, true_holo, or ligand path")
            variants = build_variants(row, manifest_dir, args.out_dir, args)
            target_rows = compute_structure_metrics(
                variants,
                raw_af2=raw,
                true_holo=true_holo,
                ligand_sdf=ligand,
                chain_id=row.get("holo_chain") or row.get("chain_id") or args.chain_id,
                args=metric_args,
            )
            for metric_row in target_rows:
                metric_row["target_id"] = target_id
                metric_row["protein_family"] = row.get("protein_family", "")
                metric_row["split"] = row.get("split", "")
                all_rows.append(metric_row)
        except Exception as exc:
            failures.append({"target_id": target_id, "error": str(exc)})

    write_csv(args.out_dir / "structure_readiness_metrics_all.csv", all_rows)
    write_csv(args.out_dir / "structure_readiness_failures.csv", failures)
    status = {
        "n_targets_requested": len(rows),
        "n_metric_rows": len(all_rows),
        "n_failures": len(failures),
        "has_holoshift_rows": any(row.get("structure", "").startswith("holoshift") for row in all_rows),
    }
    (args.out_dir / "readiness_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
