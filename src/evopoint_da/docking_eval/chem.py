from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.center_x, self.center_y, self.center_z)

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.size_x, self.size_y, self.size_z)

    def as_dict(self) -> dict[str, float]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
        }


def _require_rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolAlign
    except ImportError as exc:
        raise RuntimeError(
            "The docking pipeline needs RDKit for ligand preparation and pose RMSD. "
            "Install the docking extra/environment dependencies: rdkit, meeko, and vina."
        ) from exc
    return Chem, AllChem, rdMolAlign


def _read_sdf_molecules(sdf_path: str | Path, *, remove_hs: bool = False, strict: bool = True) -> list[Any]:
    Chem, _, _ = _require_rdkit()
    path = Path(sdf_path)
    if not path.exists():
        raise FileNotFoundError(f"SDF file not found: {path}")
    molecules = [m for m in Chem.SDMolSupplier(str(path), removeHs=remove_hs) if m is not None]
    if strict and not molecules:
        raise ValueError(f"No valid molecules were read from SDF file: {path}")
    return molecules


def prepare_ligand_sdf(
    input_sdf: str | Path,
    output_sdf: str | Path,
    *,
    add_hydrogens: bool = True,
    embed_missing_conformer: bool = True,
    random_seed: int = 42,
) -> Path:
    Chem, AllChem, _ = _require_rdkit()
    molecules = _read_sdf_molecules(input_sdf, remove_hs=False)
    if len(molecules) != 1:
        raise ValueError(
            f"Expected exactly one ligand molecule in {input_sdf}, but found {len(molecules)}. "
            "Split multi-molecule SDF inputs before docking."
        )

    mol = Chem.Mol(molecules[0])
    if add_hydrogens and not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)

    if mol.GetNumConformers() == 0:
        if not embed_missing_conformer:
            raise ValueError(f"Ligand SDF has no conformer: {input_sdf}")
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise RuntimeError(f"3D conformer generation failed for ligand: {input_sdf}")
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol)

    out = Path(output_sdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out))
    writer.write(mol)
    writer.close()
    return out


def split_sdf_poses(input_sdf: str | Path, output_dir: str | Path, *, prefix: str = "pose") -> list[Path]:
    Chem, _, _ = _require_rdkit()
    molecules = _read_sdf_molecules(input_sdf, remove_hs=False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    width = max(3, len(str(len(molecules))))
    for idx, mol in enumerate(molecules, start=1):
        path = out_dir / f"{prefix}_{idx:0{width}d}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()
        paths.append(path)
    return paths


def infer_box_from_ligand_sdf(
    reference_ligand_sdf: str | Path,
    *,
    padding_angstrom: float = 8.0,
    min_size_angstrom: float = 16.0,
) -> DockingBox:
    molecules = _read_sdf_molecules(reference_ligand_sdf, remove_hs=False)
    mol = molecules[0]
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Reference ligand has no coordinates: {reference_ligand_sdf}")
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    if not coords:
        raise ValueError(f"Reference ligand has no heavy atoms: {reference_ligand_sdf}")
    arr = np.asarray(coords, dtype=float)
    center = arr.mean(axis=0)
    span = arr.max(axis=0) - arr.min(axis=0)
    size = np.maximum(span + 2.0 * float(padding_angstrom), float(min_size_angstrom))
    return DockingBox(
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
    )


def _remove_hydrogens(mol: Any) -> Any:
    Chem, _, _ = _require_rdkit()
    try:
        return Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    except TypeError:
        return Chem.RemoveHs(Chem.Mol(mol))


def _ordered_heavy_atom_rmsd(pose_mol: Any, ref_mol: Any) -> float:
    pose_conf = pose_mol.GetConformer()
    ref_conf = ref_mol.GetConformer()
    pose_indices = [atom.GetIdx() for atom in pose_mol.GetAtoms() if atom.GetAtomicNum() > 1]
    ref_indices = [atom.GetIdx() for atom in ref_mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(pose_indices) != len(ref_indices):
        raise ValueError(
            f"Heavy atom counts differ: pose={len(pose_indices)}, reference={len(ref_indices)}"
        )
    pose_atomic = [pose_mol.GetAtomWithIdx(i).GetAtomicNum() for i in pose_indices]
    ref_atomic = [ref_mol.GetAtomWithIdx(i).GetAtomicNum() for i in ref_indices]
    if pose_atomic != ref_atomic:
        raise ValueError("Heavy atom order differs and RDKit symmetry mapping failed.")

    sq = 0.0
    for pose_idx, ref_idx in zip(pose_indices, ref_indices):
        p = pose_conf.GetAtomPosition(pose_idx)
        r = ref_conf.GetAtomPosition(ref_idx)
        sq += (p.x - r.x) ** 2 + (p.y - r.y) ** 2 + (p.z - r.z) ** 2
    return math.sqrt(sq / len(pose_indices)) if pose_indices else float("nan")


def heavy_atom_rmsd_no_align(pose_mol: Any, ref_mol: Any) -> float:
    """Compute ligand heavy-atom RMSD in receptor coordinates.

    This intentionally does not superpose the ligand. Docking pose prediction
    success asks whether Vina placed the ligand in the crystal pocket frame.
    RDKit is used first for symmetry-aware atom mapping; ordered heavy atoms are
    used as a fallback for exported poses that preserve input atom order.
    """
    _, _, rdMolAlign = _require_rdkit()
    pose_no_h = _remove_hydrogens(pose_mol)
    ref_no_h = _remove_hydrogens(ref_mol)
    if pose_no_h.GetNumConformers() == 0 or ref_no_h.GetNumConformers() == 0:
        raise ValueError("Pose and reference molecules must both have 3D conformers.")
    try:
        return float(rdMolAlign.CalcRMS(pose_no_h, ref_no_h))
    except Exception:
        return _ordered_heavy_atom_rmsd(pose_mol, ref_mol)


def compute_pose_rmsds(pose_sdf: str | Path, reference_ligand_sdf: str | Path) -> list[float]:
    poses = _read_sdf_molecules(pose_sdf, remove_hs=False)
    refs = _read_sdf_molecules(reference_ligand_sdf, remove_hs=False)
    ref = refs[0]
    return [heavy_atom_rmsd_no_align(pose, ref) for pose in poses]

