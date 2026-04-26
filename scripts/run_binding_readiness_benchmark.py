#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evopoint_da.docking_eval.pipeline import DockingPipelineConfig, StructureSpec, run_docking_pipeline
from scripts.run_four_way_docking import openmm_relax_protein


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
VDW_RADII = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}

CHI_DEFS = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}


@dataclass(frozen=True)
class AtomRecord:
    line: str
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    element: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    @property
    def atom_key(self) -> tuple[str, int, str, str]:
        return (self.chain, self.resseq, self.icode, self.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run binding-readiness supplementary experiments for a four-way docking run: "
            "HoloShift scale ensemble, CA-relaxed side-chain repack proxy, NMA ensemble, "
            "docking, and pocket-level structural metrics."
        )
    )
    p.add_argument(
        "--source-four-way-dir",
        type=Path,
        default=Path("outputs/docking_four_way/5S8I_2LY_publishable_single_dsspfix"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/binding_readiness/5S8I_2LY_publishable_single_dsspfix"),
    )
    p.add_argument("--target-id", default=None)
    p.add_argument("--chain-id", default="A")
    p.add_argument("--openmm-python", type=Path, default=Path("/home/zero/miniconda3/envs/mdw/bin/python3.11"))
    p.add_argument("--docking-bin-dir", type=Path, default=Path("/home/zero/miniconda3/envs/mdw/bin"))
    p.add_argument("--openmm-restraint-k", type=float, default=1000.0)
    p.add_argument("--openmm-max-iterations", type=int, default=200)
    p.add_argument("--pocket-cutoff", type=float, default=6.0)
    p.add_argument("--clash-cutoff", type=float, default=2.0)
    p.add_argument("--soft-clash-cutoff", type=float, default=2.5)
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--grid-padding", type=float, default=3.0)
    p.add_argument("--grid-shell-cutoff", type=float, default=4.0)
    p.add_argument("--nma-modes", type=int, default=2)
    p.add_argument("--nma-amplitude", type=float, default=0.5)
    p.add_argument("--holoshift-scales", default="0.5,1.0,1.5,2.0")
    p.add_argument("--run-docking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=20)
    p.add_argument("--energy-range", type=float, default=5.0)
    p.add_argument("--vina-seed", type=int, default=20260408)
    p.add_argument("--ligand-seed", type=int, default=42)
    p.add_argument("--topn-levels", default="1,3,5,10,20")
    return p.parse_args()


def read_csv_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row in {path}, found {len(rows)}.")
    return rows[0]


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
        for row in rows:
            writer.writerow(row)


def parse_topn(raw: str) -> list[int]:
    values = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one top-N level is required.")
    return values


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _element_from_line(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    stripped = atom_name.strip()
    if not stripped:
        return ""
    if len(stripped) >= 2 and stripped[:2].upper() in {"CL", "BR"}:
        return stripped[:2].upper()
    return stripped[0].upper()


def parse_pdb_atoms(path: Path, *, chain_id: str | None = None, atom_records_only: bool = True) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            record = line[:6].strip()
            if atom_records_only and record != "ATOM":
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            if chain_id is not None and chain != chain_id:
                continue
            try:
                resseq = int(line[22:26])
                coord = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                )
            except ValueError:
                continue
            name = line[12:16].strip()
            atoms.append(
                AtomRecord(
                    line=line.rstrip("\n"),
                    record=record,
                    serial=int(line[6:11]) if line[6:11].strip().isdigit() else len(atoms) + 1,
                    name=name,
                    resname=line[17:20].strip(),
                    chain=chain,
                    resseq=resseq,
                    icode=line[26].strip(),
                    coord=coord,
                    element=_element_from_line(line, name),
                )
            )
    return atoms


def read_ligand_heavy_coords(path: Path) -> np.ndarray:
    from rdkit import Chem

    mols = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if not mols:
        raise ValueError(f"No molecule read from {path}")
    mol = mols[0]
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    if not coords:
        raise ValueError(f"No heavy atoms in {path}")
    return np.asarray(coords, dtype=float)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    cov = a0.T @ b0
    v, _s, wt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, d]) @ wt
    moved = a0 @ rot
    return float(np.sqrt(np.mean(np.sum((moved - b0) ** 2, axis=1))))


def direct_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norm = np.linalg.norm(b1)
    if norm == 0:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def angular_diff(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def residue_atom_map(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], dict[str, AtomRecord]]:
    out: dict[tuple[str, int, str], dict[str, AtomRecord]] = defaultdict(dict)
    for atom in atoms:
        out[atom.residue_key][atom.name] = atom
    return out


def ca_map(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], np.ndarray]:
    return {atom.residue_key: atom.coord for atom in atoms if atom.name == "CA"}


def atom_coord_map(atoms: list[AtomRecord], *, sidechain_only: bool = False, heavy_only: bool = True) -> dict[tuple[str, int, str, str], np.ndarray]:
    out: dict[tuple[str, int, str, str], np.ndarray] = {}
    for atom in atoms:
        if heavy_only and atom.element == "H":
            continue
        if sidechain_only and atom.name in BACKBONE_ATOMS:
            continue
        out[atom.atom_key] = atom.coord
    return out


def infer_pocket_residues(
    true_atoms: list[AtomRecord],
    ligand_coords: np.ndarray,
    *,
    cutoff: float,
) -> set[tuple[str, int, str]]:
    pocket: set[tuple[str, int, str]] = set()
    tree = cKDTree(ligand_coords)
    for atom in true_atoms:
        if atom.element == "H":
            continue
        dist, _idx = tree.query(atom.coord, k=1)
        if float(dist) <= cutoff:
            pocket.add(atom.residue_key)
    return pocket


def compute_chi_values(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], dict[str, float]]:
    residues = residue_atom_map(atoms)
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for key, atom_by_name in residues.items():
        resname = next(iter(atom_by_name.values())).resname
        defs = CHI_DEFS.get(resname, [])
        values: dict[str, float] = {}
        for idx, names in enumerate(defs, start=1):
            if all(name in atom_by_name for name in names):
                coords = [atom_by_name[name].coord for name in names]
                values[f"chi{idx}"] = angle_between(coords[0], coords[1], coords[2], coords[3])
        if values:
            out[key] = values
    return out


def mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def count_ligand_clashes(
    receptor_atoms: list[AtomRecord],
    ligand_coords: np.ndarray,
    *,
    cutoff: float,
    soft_cutoff: float,
) -> dict[str, float | int]:
    protein_coords = np.asarray([a.coord for a in receptor_atoms if a.element != "H"], dtype=float)
    if len(protein_coords) == 0:
        return {
            "ligand_min_receptor_distance": float("nan"),
            "ligand_mean_min_receptor_distance": float("nan"),
            "ligand_clash_pairs_lt_cutoff": 0,
            "ligand_soft_clash_pairs_lt_cutoff": 0,
            "ligand_atoms_with_clash": 0,
            "ligand_steric_overlap_score": 0.0,
        }
    tree = cKDTree(protein_coords)
    pair_counts = tree.query_ball_point(ligand_coords, r=soft_cutoff)
    min_dists, _idx = tree.query(ligand_coords, k=1)
    hard_pairs = 0
    soft_pairs = 0
    for lig_coord, candidates in zip(ligand_coords, pair_counts, strict=True):
        for idx in candidates:
            dist = float(np.linalg.norm(lig_coord - protein_coords[idx]))
            if dist < soft_cutoff:
                soft_pairs += 1
            if dist < cutoff:
                hard_pairs += 1
    overlap = float(np.maximum(0.0, cutoff - min_dists).sum())
    return {
        "ligand_min_receptor_distance": float(np.min(min_dists)),
        "ligand_mean_min_receptor_distance": float(np.mean(min_dists)),
        "ligand_clash_pairs_lt_cutoff": int(hard_pairs),
        "ligand_soft_clash_pairs_lt_cutoff": int(soft_pairs),
        "ligand_atoms_with_clash": int(np.sum(min_dists < cutoff)),
        "ligand_steric_overlap_score": overlap,
    }


def build_ligand_shell_grid(
    ligand_coords: np.ndarray,
    *,
    spacing: float,
    padding: float,
    shell_cutoff: float,
) -> np.ndarray:
    lo = ligand_coords.min(axis=0) - padding
    hi = ligand_coords.max(axis=0) + padding
    axes = [np.arange(lo[i], hi[i] + spacing * 0.5, spacing) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([m.ravel() for m in mesh], axis=1)
    ligand_tree = cKDTree(ligand_coords)
    dist, _ = ligand_tree.query(grid, k=1)
    return grid[dist <= shell_cutoff]


def pocket_free_mask(
    receptor_atoms: list[AtomRecord],
    grid: np.ndarray,
    *,
    exclusion_cutoff: float = 2.0,
) -> np.ndarray:
    coords = np.asarray([a.coord for a in receptor_atoms if a.element != "H"], dtype=float)
    if len(coords) == 0:
        return np.ones(len(grid), dtype=bool)
    tree = cKDTree(coords)
    dist, _ = tree.query(grid, k=1)
    return dist >= exclusion_cutoff


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def format_label_value(value: float) -> str:
    text = f"{value:.2f}".replace("-", "neg").replace(".", "p")
    return text


def rewrite_pdb_with_residue_shifts(
    input_pdb: Path,
    output_pdb: Path,
    shifts: dict[tuple[str, int, str], np.ndarray],
) -> None:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with input_pdb.open(encoding="utf-8", errors="replace") as src, output_pdb.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21].strip() or "_"
                icode = line[26].strip()
                try:
                    resseq = int(line[22:26])
                    coord = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    )
                except ValueError:
                    dst.write(line)
                    continue
                shift = shifts.get((chain, resseq, icode))
                if shift is not None:
                    new_coord = coord + shift
                    line = f"{line[:30]}{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}{line[54:]}"
            dst.write(line)


def generate_scaled_holoshift(raw_pdb: Path, holoshift_unrelaxed_pdb: Path, output_pdb: Path, scale: float) -> None:
    raw_atoms = parse_pdb_atoms(raw_pdb, atom_records_only=True)
    holo_atoms = parse_pdb_atoms(holoshift_unrelaxed_pdb, atom_records_only=True)
    raw_ca = ca_map(raw_atoms)
    holo_ca = ca_map(holo_atoms)
    shifts = {
        key: scale * (holo_ca[key] - raw_ca[key])
        for key in sorted(set(raw_ca) & set(holo_ca))
    }
    rewrite_pdb_with_residue_shifts(raw_pdb, output_pdb, shifts)


def generate_nma_ensemble(raw_pdb: Path, out_dir: Path, *, n_modes: int, amplitude: float) -> list[tuple[str, Path]]:
    atoms = parse_pdb_atoms(raw_pdb, atom_records_only=True)
    raw_ca = ca_map(atoms)
    residue_keys = sorted(raw_ca)
    coords = np.asarray([raw_ca[key] for key in residue_keys], dtype=float)
    n = len(coords)
    hessian = np.zeros((3 * n, 3 * n), dtype=float)
    cutoff = 10.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-6 or dist > cutoff:
                continue
            unit = diff / dist
            block = np.outer(unit, unit)
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)
            hessian[si, si] += block
            hessian[sj, sj] += block
            hessian[si, sj] -= block
            hessian[sj, si] -= block
    eigenvalues, eigenvectors = eigh(hessian)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    outputs: list[tuple[str, Path]] = []
    mode_start = 6
    for local_idx in range(n_modes):
        vec = eigenvectors[:, mode_start + local_idx].reshape(n, 3)
        rms = float(np.sqrt(np.mean(np.sum(vec**2, axis=1))))
        if rms <= 0:
            continue
        vec = vec / rms * amplitude
        for sign, sign_label in [(-1.0, "neg"), (1.0, "pos")]:
            shifts = {key: sign * vec[idx] for idx, key in enumerate(residue_keys)}
            label = f"nma_m{local_idx + 1}_{sign_label}_{format_label_value(amplitude)}A"
            path = out_dir / f"{label}.pdb"
            rewrite_pdb_with_residue_shifts(raw_pdb, path, shifts)
            outputs.append((label, path))
    return outputs


def maybe_ca_relax(
    input_pdb: Path,
    output_pdb: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report_path = output_pdb.with_suffix(".json")
    if args.reuse and output_pdb.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    report = openmm_relax_protein(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        restraint_k=args.openmm_restraint_k,
        max_iterations=args.openmm_max_iterations,
        restrain_selection="ca",
        openmm_python=args.openmm_python if args.openmm_python and args.openmm_python.exists() else None,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_variants(args: argparse.Namespace, manifest_row: dict[str, str]) -> tuple[list[dict[str, Any]], list[StructureSpec]]:
    prep = args.out_dir / "prepared_variants"
    prep.mkdir(parents=True, exist_ok=True)

    raw_af2 = Path(manifest_row["receptor_raw_af2"])
    af2_heavy_relax = Path(manifest_row["receptor_af2_openmm_relax"])
    holoshift_heavy_relax = Path(manifest_row["receptor_holoshift"])
    true_holo = Path(manifest_row["receptor_true_holo"])
    source_prep = args.source_four_way_dir / "prepared_inputs"
    holoshift_unrelaxed = source_prep / f"{manifest_row['target_id']}_holoshift_predict_openmm_relax_predicted_unrelaxed.pdb"
    if not holoshift_unrelaxed.exists():
        raise FileNotFoundError(f"Missing HoloShift unrelaxed PDB: {holoshift_unrelaxed}")

    variants: list[dict[str, Any]] = [
        {"label": "raw_af2", "path": raw_af2, "kind": "baseline"},
        {"label": "af2_openmm_heavy_relax", "path": af2_heavy_relax, "kind": "baseline"},
        {"label": "holoshift_shift_only", "path": holoshift_unrelaxed, "kind": "holoshift_shift"},
        {"label": "holoshift_heavy_relax", "path": holoshift_heavy_relax, "kind": "holoshift_shift_heavy_relax"},
        {"label": "true_holo", "path": true_holo, "kind": "reference"},
    ]

    raw_ca_relax = prep / "raw_af2_ca_repack.pdb"
    variants.append(
        {
            "label": "raw_af2_ca_repack",
            "path": raw_ca_relax,
            "kind": "openmm_ca_repack",
            "source": raw_af2,
            "relax": maybe_ca_relax(raw_af2, raw_ca_relax, args=args),
        }
    )

    hs_ca_repack = prep / "holoshift_shift_ca_repack.pdb"
    variants.append(
        {
            "label": "holoshift_shift_ca_repack",
            "path": hs_ca_repack,
            "kind": "holoshift_shift_ca_repack",
            "source": holoshift_unrelaxed,
            "relax": maybe_ca_relax(holoshift_unrelaxed, hs_ca_repack, args=args),
        }
    )

    for scale in parse_scales(args.holoshift_scales):
        label_base = f"holoshift_scale_{format_label_value(scale)}"
        scaled = prep / f"{label_base}.pdb"
        if not args.reuse or not scaled.exists():
            generate_scaled_holoshift(raw_af2, holoshift_unrelaxed, scaled, scale)
        ca_path = prep / f"{label_base}_ca_repack.pdb"
        variants.append(
            {
                "label": f"{label_base}_ca_repack",
                "path": ca_path,
                "kind": "holoshift_scale_ca_repack",
                "scale": scale,
                "source": scaled,
                "relax": maybe_ca_relax(scaled, ca_path, args=args),
            }
        )

    nma_raw = generate_nma_ensemble(raw_af2, prep / "nma_raw", n_modes=args.nma_modes, amplitude=args.nma_amplitude)
    for label, nma_path in nma_raw:
        variants.append({"label": label, "path": nma_path, "kind": "nma_raw"})
        ca_path = prep / f"{label}_ca_repack.pdb"
        variants.append(
            {
                "label": f"{label}_ca_repack",
                "path": ca_path,
                "kind": "nma_ca_repack",
                "source": nma_path,
                "relax": maybe_ca_relax(nma_path, ca_path, args=args),
            }
        )

    specs = [StructureSpec(label=str(v["label"]), receptor_col=f"receptor_{v['label']}") for v in variants]
    return variants, specs


def compute_structure_metrics(
    variants: list[dict[str, Any]],
    *,
    raw_af2: Path,
    true_holo: Path,
    ligand_sdf: Path,
    chain_id: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ligand_coords = read_ligand_heavy_coords(ligand_sdf)
    true_atoms = parse_pdb_atoms(true_holo, chain_id=chain_id, atom_records_only=True)
    raw_atoms = parse_pdb_atoms(raw_af2, chain_id=chain_id, atom_records_only=True)
    pocket = infer_pocket_residues(true_atoms, ligand_coords, cutoff=args.pocket_cutoff)
    grid = build_ligand_shell_grid(
        ligand_coords,
        spacing=args.grid_spacing,
        padding=args.grid_padding,
        shell_cutoff=args.grid_shell_cutoff,
    )
    true_free = pocket_free_mask(true_atoms, grid, exclusion_cutoff=args.clash_cutoff)
    true_ca = ca_map(true_atoms)
    raw_ca = ca_map(raw_atoms)
    true_side = atom_coord_map(true_atoms, sidechain_only=True, heavy_only=True)
    true_chi = compute_chi_values(true_atoms)
    raw_chi = compute_chi_values(raw_atoms)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        label = str(variant["label"])
        atoms = parse_pdb_atoms(Path(variant["path"]), chain_id=chain_id, atom_records_only=True)
        variant_ca = ca_map(atoms)
        common_ca = sorted(set(variant_ca) & set(true_ca))
        pocket_ca = [key for key in common_ca if key in pocket]
        all_a = np.asarray([variant_ca[key] for key in common_ca], dtype=float)
        all_b = np.asarray([true_ca[key] for key in common_ca], dtype=float)
        pocket_a = np.asarray([variant_ca[key] for key in pocket_ca], dtype=float)
        pocket_b = np.asarray([true_ca[key] for key in pocket_ca], dtype=float)

        side = atom_coord_map(atoms, sidechain_only=True, heavy_only=True)
        pocket_side_keys = sorted(
            key
            for key in set(side) & set(true_side)
            if (key[0], key[1], key[2]) in pocket
        )
        side_a = np.asarray([side[key] for key in pocket_side_keys], dtype=float)
        side_b = np.asarray([true_side[key] for key in pocket_side_keys], dtype=float)

        raw_delta = []
        pred_delta = []
        target_delta_error = []
        for key in pocket_ca:
            if key in raw_ca:
                t = true_ca[key] - raw_ca[key]
                p = variant_ca[key] - raw_ca[key]
                raw_delta.append(t)
                pred_delta.append(p)
                target_delta_error.append(p - t)
        cosines = []
        projections = []
        for t, p in zip(raw_delta, pred_delta, strict=False):
            tn = float(np.linalg.norm(t))
            pn = float(np.linalg.norm(p))
            if tn > 1e-8 and pn > 1e-8:
                cosines.append(float(np.dot(t, p) / (tn * pn)))
                projections.append(float(np.dot(p, t) / (tn * tn)))

        chi = compute_chi_values(atoms)
        chi_metrics: dict[str, Any] = {}
        for chi_name in ("chi1", "chi2"):
            diffs = []
            changed = 0
            recovered = 0
            improved = 0
            for key in sorted(pocket):
                if chi_name not in chi.get(key, {}) or chi_name not in true_chi.get(key, {}):
                    continue
                err = angular_diff(chi[key][chi_name], true_chi[key][chi_name])
                diffs.append(err)
                if chi_name in raw_chi.get(key, {}):
                    raw_err = angular_diff(raw_chi[key][chi_name], true_chi[key][chi_name])
                    if raw_err > 60.0:
                        changed += 1
                        if err <= 40.0:
                            recovered += 1
                        if err + 1e-6 < raw_err:
                            improved += 1
            chi_metrics[f"pocket_{chi_name}_mae_deg"] = mean_or_nan(diffs)
            chi_metrics[f"pocket_{chi_name}_n"] = len([v for v in diffs if not math.isnan(v)])
            chi_metrics[f"pocket_{chi_name}_apo_holo_changed_n"] = changed
            chi_metrics[f"pocket_{chi_name}_changed_recovered_rate"] = recovered / changed if changed else float("nan")
            chi_metrics[f"pocket_{chi_name}_changed_improved_rate"] = improved / changed if changed else float("nan")

        free = pocket_free_mask(atoms, grid, exclusion_cutoff=args.clash_cutoff)
        clashes = count_ligand_clashes(
            atoms,
            ligand_coords,
            cutoff=args.clash_cutoff,
            soft_cutoff=args.soft_clash_cutoff,
        )
        row: dict[str, Any] = {
            "structure": label,
            "kind": variant.get("kind", ""),
            "receptor_pdb": str(variant["path"]),
            "pocket_residue_count": len(pocket),
            "global_ca_rmsd_direct_vs_true": direct_rmsd(all_a, all_b),
            "global_ca_rmsd_aligned_vs_true": kabsch_rmsd(all_a, all_b),
            "pocket_ca_rmsd_direct_vs_true": direct_rmsd(pocket_a, pocket_b),
            "pocket_ca_rmsd_aligned_vs_true": kabsch_rmsd(pocket_a, pocket_b),
            "pocket_sidechain_heavy_common_atoms": len(pocket_side_keys),
            "pocket_sidechain_heavy_rmsd_direct_vs_true": direct_rmsd(side_a, side_b),
            "pocket_sidechain_heavy_rmsd_aligned_vs_true": kabsch_rmsd(side_a, side_b),
            "pocket_ca_delta_rmse_vs_raw_to_true": (
                float(np.sqrt(np.mean(np.sum(np.asarray(target_delta_error) ** 2, axis=1))))
                if target_delta_error
                else float("nan")
            ),
            "pocket_ca_delta_cosine_mean_vs_raw_to_true": mean_or_nan(cosines),
            "pocket_ca_delta_projection_mean_vs_raw_to_true": mean_or_nan(projections),
            "pocket_grid_points": len(grid),
            "pocket_grid_free_volume_A3": float(free.sum() * (args.grid_spacing**3)),
            "pocket_grid_shape_jaccard_vs_true": jaccard(free, true_free),
            "pocket_grid_false_blocked_fraction_vs_true": (
                float(np.logical_and(~free, true_free).sum() / true_free.sum()) if true_free.sum() else float("nan")
            ),
            "pocket_grid_false_open_fraction_vs_true": (
                float(np.logical_and(free, ~true_free).sum() / (~true_free).sum()) if (~true_free).sum() else float("nan")
            ),
            **clashes,
            **chi_metrics,
        }
        rows.append(row)
    return rows


def write_manifest(
    path: Path,
    target_id: str,
    variants: list[dict[str, Any]],
    ligand_sdf: Path,
    reference_ligand_sdf: Path,
) -> None:
    row: dict[str, Any] = {
        "target_id": target_id,
        "ligand_sdf": str(ligand_sdf),
        "reference_ligand_sdf": str(reference_ligand_sdf),
    }
    for variant in variants:
        row[f"receptor_{variant['label']}"] = str(variant["path"])
    write_csv(path, [row])


def summarize_pose_metrics(poses_csv: Path, topn_levels: list[int]) -> list[dict[str, Any]]:
    with poses_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_structure: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("pose_valid") == "1":
            by_structure[row["structure"]].append(row)
    out: list[dict[str, Any]] = []
    for structure, structure_rows in sorted(by_structure.items()):
        ranked = sorted(structure_rows, key=lambda row: int(row["rank"]))
        rmsds = [float(row["rmsd"]) for row in ranked if row.get("rmsd", "").strip()]
        scores = [float(row["score"]) for row in ranked if row.get("score", "").strip()]
        first_hit = next((int(row["rank"]) for row in ranked if row.get("success_at_threshold") == "1"), None)
        row: dict[str, Any] = {
            "structure": structure,
            "n_poses": len(ranked),
            "top1_rmsd": float(ranked[0]["rmsd"]) if ranked and ranked[0].get("rmsd", "").strip() else float("nan"),
            "top1_score": float(ranked[0]["score"]) if ranked and ranked[0].get("score", "").strip() else float("nan"),
            "best_rmsd": min(rmsds) if rmsds else float("nan"),
            "best_rmsd_rank": (
                int(min(ranked, key=lambda r: float(r["rmsd"]) if r.get("rmsd", "").strip() else float("inf"))["rank"])
                if rmsds
                else ""
            ),
            "best_score": min(scores) if scores else float("nan"),
            "first_hit_rank": first_hit if first_hit is not None else "",
        }
        for topn in topn_levels:
            subset = [r for r in ranked if int(r["rank"]) <= topn]
            row[f"top{topn}_success"] = int(any(r.get("success_at_threshold") == "1" for r in subset))
        out.append(row)
    return out


def merge_metrics(structure_rows: list[dict[str, Any]], pose_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pose_by_structure = {row["structure"]: row for row in pose_rows}
    out = []
    for row in structure_rows:
        merged = dict(row)
        pose = pose_by_structure.get(str(row["structure"]), {})
        for key, value in pose.items():
            if key != "structure":
                merged[key] = value
        out.append(merged)
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    return f"{f:.{digits}f}"


def write_report(
    path: Path,
    *,
    structure_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    screening_status: dict[str, Any],
    docking_failures: list[dict[str, Any]],
) -> None:
    pose_by_structure = {row["structure"]: row for row in pose_rows}
    failure_by_structure = {str(row.get("structure", "")): row for row in docking_failures if row.get("structure")}
    lines = [
        "# Binding Readiness Supplementary Benchmark",
        "",
        "## Experiment Arms",
        "",
        "| Structure | Kind | Receptor |",
        "|---|---|---|",
    ]
    for variant in variants:
        lines.append(f"| {variant['label']} | {variant.get('kind', '')} | {variant['path']} |")

    lines.extend(
        [
            "",
            "## Structure Readiness Metrics",
            "",
            "| Structure | pocket CA RMSD | pocket side-chain RMSD | chi1 MAE | clashes <2A | pocket shape Jaccard |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in structure_rows:
        lines.append(
            "| "
            f"{row['structure']} | "
            f"{_fmt(row['pocket_ca_rmsd_direct_vs_true'])} | "
            f"{_fmt(row['pocket_sidechain_heavy_rmsd_direct_vs_true'])} | "
            f"{_fmt(row.get('pocket_chi1_mae_deg'), 1)} | "
            f"{row['ligand_clash_pairs_lt_cutoff']} | "
            f"{_fmt(row['pocket_grid_shape_jaccard_vs_true'])} |"
        )

    lines.extend(
        [
            "",
            "## Docking Metrics",
            "",
            "| Structure | Status | Top-1 RMSD | Best RMSD | Best rank | Top-5 | Top-20 | Top-1 score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in structure_rows:
        pose = pose_by_structure.get(str(row["structure"]), {})
        failure = failure_by_structure.get(str(row["structure"]))
        status = "ok" if pose else ("failed" if failure else "")
        lines.append(
            "| "
            f"{row['structure']} | "
            f"{status} | "
            f"{_fmt(pose.get('top1_rmsd'))} | "
            f"{_fmt(pose.get('best_rmsd'))} | "
            f"{pose.get('best_rmsd_rank', '')} | "
            f"{pose.get('top5_success', '')} | "
            f"{pose.get('top20_success', '')} | "
            f"{_fmt(pose.get('top1_score'))} |"
        )

    valid_pose_rows = [row for row in combined_rows if _fmt(row.get("best_rmsd"))]
    if combined_rows:
        best_shape = max(
            combined_rows,
            key=lambda r: float(r.get("pocket_grid_shape_jaccard_vs_true", float("nan")))
            if not math.isnan(float(r.get("pocket_grid_shape_jaccard_vs_true", float("nan"))))
            else -1.0,
        )
        lines.extend(
            [
                "",
                "## Quick Read",
                "",
                f"- Best local pocket-shape arm: {best_shape['structure']} (grid Jaccard={_fmt(best_shape.get('pocket_grid_shape_jaccard_vs_true'))}).",
                f"- VS/cross-docking enrichment: {screening_status['status']} ({screening_status['reason']}).",
            ]
        )
        if valid_pose_rows:
            best_pose = min(valid_pose_rows, key=lambda r: float(r.get("best_rmsd", "inf") or "inf"))
            lines.insert(
                -2,
                f"- Best docking RMSD arm: {best_pose['structure']} "
                f"(best RMSD={_fmt(best_pose.get('best_rmsd'))}, rank={best_pose.get('best_rmsd_rank', '')}).",
            )
        if docking_failures:
            lines.append(f"- Docking failures: {len(docking_failures)} arm(s); see `docking_failures.csv`.")

    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `structure_readiness_metrics.csv`: pocket geometry, chi angles, ligand clash, and grid shape metrics.",
            "- `pose_metrics.csv`: Top-N pose RMSD and Vina score metrics.",
            "- `docking_failures.csv`: receptor-preparation or docking failures, if any.",
            "- `combined_readiness_metrics.csv`: joined structure+docking table.",
            "- `docking/poses_all.csv`: raw pose-level docking table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.source_four_way_dir = args.source_four_way_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.docking_bin_dir and args.docking_bin_dir.exists():
        os.environ["PATH"] = os.pathsep.join([str(args.docking_bin_dir.resolve()), os.environ.get("PATH", "")])

    source_manifest = args.source_four_way_dir / "four_way_manifest.csv"
    manifest_row = read_csv_one(source_manifest)
    target_id = args.target_id or manifest_row["target_id"]
    ligand_sdf = Path(manifest_row["ligand_sdf"])
    reference_ligand_sdf = Path(manifest_row["reference_ligand_sdf"])
    raw_af2 = Path(manifest_row["receptor_raw_af2"])
    true_holo = Path(manifest_row["receptor_true_holo"])

    variants, specs = build_variants(args, manifest_row)
    artifacts = {
        "source_four_way_dir": str(args.source_four_way_dir),
        "target_id": target_id,
        "variants": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in variant.items()
            }
            for variant in variants
        ],
    }
    (args.out_dir / "variant_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    supplemental_manifest = args.out_dir / "binding_readiness_manifest.csv"
    write_manifest(supplemental_manifest, target_id, variants, ligand_sdf, reference_ligand_sdf)

    structure_rows = compute_structure_metrics(
        variants,
        raw_af2=raw_af2,
        true_holo=true_holo,
        ligand_sdf=reference_ligand_sdf,
        chain_id=args.chain_id,
        args=args,
    )
    write_csv(args.out_dir / "structure_readiness_metrics.csv", structure_rows)

    pose_rows: list[dict[str, Any]] = []
    docking_failures: list[dict[str, Any]] = []
    if args.run_docking:
        docking_dir = args.out_dir / "docking"
        cfg = DockingPipelineConfig(
            manifest=supplemental_manifest,
            output_dir=docking_dir,
            structures=specs,
            rmsd_threshold=2.0,
            topn_levels=parse_topn(args.topn_levels),
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            energy_range=args.energy_range,
            vina_seed=args.vina_seed,
            ligand_seed=args.ligand_seed,
            bootstrap_iter=200,
            bootstrap_seed=42,
            reuse=args.reuse,
            skip_failed=True,
        )
        docking_summary = run_docking_pipeline(cfg)
        docking_failures = list(docking_summary.get("failures", []))
        pose_rows = summarize_pose_metrics(docking_dir / "poses_all.csv", parse_topn(args.topn_levels))
        write_csv(args.out_dir / "pose_metrics.csv", pose_rows)
        write_csv(args.out_dir / "docking_failures.csv", docking_failures)
    else:
        existing_poses = args.out_dir / "docking" / "poses_all.csv"
        if existing_poses.exists():
            pose_rows = summarize_pose_metrics(existing_poses, parse_topn(args.topn_levels))
            write_csv(args.out_dir / "pose_metrics.csv", pose_rows)
        existing_summary = args.out_dir / "docking" / "summary.json"
        if existing_summary.exists():
            docking_failures = list(json.loads(existing_summary.read_text(encoding="utf-8")).get("failures", []))
            write_csv(args.out_dir / "docking_failures.csv", docking_failures)

    combined_rows = merge_metrics(structure_rows, pose_rows)
    write_csv(args.out_dir / "combined_readiness_metrics.csv", combined_rows)

    screening_status = {
        "status": "unavailable",
        "reason": "Only one reference ligand is present in the current 5S8I_2LY smoke/publishable target; no active/decoy panel or cross-target ligand set was provided.",
        "required_inputs": ["multiple known actives/decoys per target", "or multiple ligands across homologous receptor structures"],
    }
    (args.out_dir / "screening_enrichment_status.json").write_text(
        json.dumps(screening_status, indent=2),
        encoding="utf-8",
    )
    write_report(
        args.out_dir / "binding_readiness_report.md",
        structure_rows=structure_rows,
        pose_rows=pose_rows,
        combined_rows=combined_rows,
        variants=variants,
        screening_status=screening_status,
        docking_failures=docking_failures,
    )

    print(f"Binding-readiness report: {args.out_dir / 'binding_readiness_report.md'}")
    print(f"Combined metrics: {args.out_dir / 'combined_readiness_metrics.csv'}")


if __name__ == "__main__":
    main()
