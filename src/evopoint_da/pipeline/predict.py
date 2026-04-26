"""Reusable HoloShift prediction helpers.

This module backs ``run_Predict.py`` and can be imported by downstream
workflows such as docking evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser
from torch_geometric.data import Data

from evopoint_da.data.graph import build_knn_edges
from evopoint_da.data.structure import StructureParser, select_chain
from evopoint_da.models.module import EvoPointLitModule
from evopoint_da.utils.binning import build_bin_ranges as _build_bin_ranges

AA_ORDER = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}


def _summarize_prediction_bins(pred_norm: np.ndarray, edges: list[float]) -> dict[str, dict]:
    bin_stats: dict[str, dict] = {}
    total = int(len(pred_norm))
    for low, high, suffix in _build_bin_ranges(edges):
        if high is None:
            mask = pred_norm >= low
            label = f">={low:g}"
        else:
            mask = (pred_norm >= low) & (pred_norm < high)
            label = f"[{low:g}, {high:g})"
        values = pred_norm[mask]
        count = int(values.shape[0])
        bin_stats[suffix] = {
            "label": label,
            "count": count,
            "ratio": (count / total) if total > 0 else 0.0,
            "mean_abs_dr": float(values.mean()) if count > 0 else None,
            "median_abs_dr": float(np.median(values)) if count > 0 else None,
            "p90_abs_dr": float(np.percentile(values, 90)) if count > 0 else None,
            "max_abs_dr": float(values.max()) if count > 0 else None,
        }
    return bin_stats


def _build_auto_features(parsed: dict, expected_in: int) -> torch.Tensor:
    residue_names = parsed.get("residue_names", [])
    plddts = np.asarray(parsed.get("plddts", []), dtype=np.float32)
    coords = np.asarray(parsed.get("coords", []), dtype=np.float32)

    n = int(coords.shape[0])
    aa_onehot = np.zeros((n, len(AA_ORDER)), dtype=np.float32)
    for i, resname in enumerate(residue_names[:n]):
        idx = AA_TO_INDEX.get(str(resname).upper())
        if idx is not None:
            aa_onehot[i, idx] = 1.0

    plddt_col = plddts[:n].reshape(n, 1) if plddts.size else np.zeros((n, 1), dtype=np.float32)
    if plddt_col.max(initial=0.0) > 1.5:
        plddt_col = plddt_col / 100.0

    if n > 0:
        coord_mean = coords.mean(axis=0, keepdims=True)
        coord_std = coords.std(axis=0, keepdims=True) + 1e-6
        coord_norm = (coords - coord_mean) / coord_std
    else:
        coord_norm = np.zeros((0, 3), dtype=np.float32)

    base = np.concatenate([aa_onehot, plddt_col, coord_norm], axis=1)
    if base.shape[1] < expected_in:
        pad = np.zeros((n, expected_in - base.shape[1]), dtype=np.float32)
        base = np.concatenate([base, pad], axis=1)
    elif base.shape[1] > expected_in:
        base = base[:, :expected_in]
    return torch.from_numpy(base).float()


def _load_prediction_features(
    args: Any,
    parsed: dict,
    pos: torch.Tensor,
    expected_in: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if getattr(args, "feature_pt", None):
        feat = torch.load(args.feature_pt, weights_only=True)
        x = feat["x"].float()
        edge_index = feat.get("edge_index", None)
        edge_attr = feat.get("edge_attr", None)
        if edge_index is not None:
            edge_index = edge_index.long()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        return x, edge_index, edge_attr

    if not getattr(args, "allow_fallback_features", False):
        raise ValueError(
            "--feature_pt is required for normal inference. The automatic fallback "
            "features are AA one-hot + pLDDT + normalized coordinates and do not "
            "match the training feature pipeline. Pass a graph feature .pt built "
            "with the same ESM/PCA + structural-feature pipeline used for training, "
            "or add --allow_fallback_features for debugging only."
        )

    x = _build_auto_features(parsed, expected_in)
    print(
        "[warning] --allow_fallback_features enabled; using debug-only fallback "
        f"features with dim={x.size(1)}. These do not match training features.",
        file=sys.stderr,
    )
    save_auto_feature_pt = getattr(args, "save_auto_feature_pt", None)
    if save_auto_feature_pt:
        payload = {
            "x": x,
            "pos": pos,
            "plddt": torch.as_tensor(parsed.get("plddts", []), dtype=torch.float32).unsqueeze(1),
            "residue_ids": parsed.get("residue_ids", []),
            "sequence": parsed.get("sequence", ""),
        }
        torch.save(payload, save_auto_feature_pt)
        print(f"[info] Auto-built fallback features saved to: {save_auto_feature_pt}")
    return x, None, None


def _feature_edges_for_node_count(
    edge_index: torch.Tensor | None,
    edge_attr: torch.Tensor | None,
    n: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if edge_index is None or edge_attr is None:
        return None, None
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if edge_attr.dim() != 2 or edge_attr.size(0) != edge_index.size(1):
        raise ValueError(
            "edge_attr must have shape (E, edge_dim) matching edge_index; "
            f"got edge_attr={tuple(edge_attr.shape)}, edge_index={tuple(edge_index.shape)}"
        )
    if edge_index.numel() == 0:
        return edge_index.contiguous(), edge_attr.contiguous()
    valid = (edge_index[0] >= 0) & (edge_index[1] >= 0) & (edge_index[0] < n) & (edge_index[1] < n)
    if not bool(valid.all()):
        edge_index = edge_index[:, valid]
        edge_attr = edge_attr[valid]
    return edge_index.contiguous(), edge_attr.contiguous()


def _load_qhat(conformal_stats: str | None) -> float | None:
    if not conformal_stats:
        return None
    with open(conformal_stats, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats["qhat"])


def _build_prediction_report(
    args: Any,
    selected_chain_id: str,
    qhat: float | None,
    reject: bool,
    expected_in: int,
    observed_in: int,
    pred_norm: np.ndarray,
    safe_center: np.ndarray,
    disp_bin_edges: list[float],
) -> dict:
    pred_stats = {
        "count": int(len(pred_norm)),
        "mean_abs_dr": float(pred_norm.mean()) if len(pred_norm) else 0.0,
        "median_abs_dr": float(np.median(pred_norm)) if len(pred_norm) else 0.0,
        "p90_abs_dr": float(np.percentile(pred_norm, 90)) if len(pred_norm) else 0.0,
        "max_abs_dr": float(pred_norm.max()) if len(pred_norm) else 0.0,
    }
    return {
        "input": {
            "pdb_file": args.pdb_file,
            "feature_pt": getattr(args, "feature_pt", None),
            "allow_fallback_features": bool(getattr(args, "allow_fallback_features", False)),
            "conformal_stats": getattr(args, "conformal_stats", None),
            "selected_chain": selected_chain_id,
            "k": int(args.k),
            "device": args.device,
        },
        "model": {
            "ckpt_path": args.ckpt_path,
            "feature_dim_observed": int(observed_in),
            "feature_dim_expected": int(expected_in),
        },
        "conformal": {
            "qhat": qhat,
            "reject_threshold": float(args.reject_threshold) if qhat is not None else None,
            "decision": "REJECT" if reject else "ACCEPT",
            "safety_radius": qhat,
        },
        "prediction_summary": pred_stats,
        "prediction_bins": _summarize_prediction_bins(pred_norm, disp_bin_edges),
        "prediction_bin_edges": [float(v) for v in disp_bin_edges],
        "preview": {"safe_center_first5": safe_center[:5].tolist()},
        "artifacts": {
            "output_pdb": getattr(args, "output_pdb", None),
            "save_auto_feature_pt": getattr(args, "save_auto_feature_pt", None),
        },
    }


def _print_prediction_report(report: dict) -> None:
    print("\n=== Prediction Report ===")
    print("[Input]")
    print(f"  pdb_file: {report['input']['pdb_file']}")
    print(f"  selected_chain: {report['input']['selected_chain']}")
    print(f"  k: {report['input']['k']}, device: {report['input']['device']}")
    print("[Model]")
    print(f"  ckpt_path: {report['model']['ckpt_path']}")
    print(
        "  feature_dim: "
        f"observed={report['model']['feature_dim_observed']}, expected={report['model']['feature_dim_expected']}"
    )
    print("[Conformal]")
    qhat = report["conformal"]["qhat"]
    if qhat is None:
        print("  qhat: not provided; conformal reject disabled")
    else:
        print(f"  decision: {report['conformal']['decision']}")
        print(
            f"  qhat={qhat:.4f}, "
            f"threshold={report['conformal']['reject_threshold']:.4f}, "
            f"safety_radius={qhat:.4f}"
        )
    ps = report["prediction_summary"]
    print("[Prediction Summary]")
    print(f"  nodes: {ps['count']}")
    print(f"  mean|Δr|: {ps['mean_abs_dr']:.4f}")
    print(f"  median|Δr|: {ps['median_abs_dr']:.4f}")
    print(f"  p90|Δr|: {ps['p90_abs_dr']:.4f}")
    print(f"  max|Δr|: {ps['max_abs_dr']:.4f}")


def predict_displacement(args: Any) -> tuple[dict, str, np.ndarray, np.ndarray, dict]:
    parser = StructureParser()
    parsed_chains = parser.parse_ca_structure(args.pdb_file)
    if not parsed_chains:
        raise ValueError("Failed to parse input structure")
    selected_chain_id, parsed = select_chain(parsed_chains, getattr(args, "chain_id", None))
    pos = torch.tensor(parsed["coords"], dtype=torch.float32)

    model = EvoPointLitModule.load_from_checkpoint(args.ckpt_path, map_location=args.device, weights_only=False)
    model.eval().to(args.device)
    expected_in = int(model.hparams.in_channels)

    x, feature_edge_index, feature_edge_attr = _load_prediction_features(args, parsed, pos, expected_in)
    if x.size(0) != len(pos):
        n = min(x.size(0), len(pos))
        print(
            f"[warning] Feature length ({x.size(0)}) != selected chain length ({len(pos)}); truncating both to {n}.",
            file=sys.stderr,
        )
        x = x[:n]
        pos = pos[:n]

    edge_index, edge_attr = _feature_edges_for_node_count(feature_edge_index, feature_edge_attr, len(pos))
    if edge_index is None or edge_attr is None:
        edge_index, edge_attr = build_knn_edges(pos, k=args.k)
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr).to(args.device)

    if x.size(1) != expected_in:
        raise ValueError(f"Feature dim drift: input feature_dim={x.size(1)} but checkpoint expects in_channels={expected_in}")

    qhat = _load_qhat(getattr(args, "conformal_stats", None))
    with torch.no_grad():
        delta = model.predict_displacement(
            data,
            apply_inference_multiplier=bool(getattr(args, "apply_inference_multiplier", False)),
        )
    pred_norm = torch.norm(delta, dim=-1).detach().cpu().numpy()
    model_bin_edges = getattr(model.hparams, "test_disp_bin_edges", [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    disp_bin_edges = [float(v) for v in model_bin_edges]

    reject = qhat is not None and qhat > args.reject_threshold
    source_center = pos.detach().cpu().numpy()
    if reject:
        safe_center = source_center
        print(f"REJECT: qhat={qhat:.4f} > threshold={args.reject_threshold:.4f}; keeping original structure")
    else:
        safe_center = (pos.to(delta.device) + delta).detach().cpu().numpy()
        if qhat is None:
            print("ACCEPT: conformal qhat not provided; returning predicted structure")
        else:
            print(f"ACCEPT: qhat={qhat:.4f}; returning predicted structure with conformal safety sphere")

    report = _build_prediction_report(
        args=args,
        selected_chain_id=selected_chain_id,
        qhat=qhat,
        reject=reject,
        expected_in=expected_in,
        observed_in=x.size(1),
        pred_norm=pred_norm,
        safe_center=safe_center,
        disp_bin_edges=disp_bin_edges,
    )
    return report, selected_chain_id, source_center, safe_center, parsed


def _iter_chain_residues(structure, chain_id: str) -> Iterable:
    for model in structure:
        if chain_id in model:
            for residue in model[chain_id]:
                if residue.id[0] == " ":
                    yield residue
            return
    raise ValueError(f"Chain {chain_id} not found in {structure.id}")


def _write_guardrailed_pdb(input_pdb: str, chain_id: str, source_ca: np.ndarray, target_ca: np.ndarray, out_pdb: str) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb)

    residues = list(_iter_chain_residues(structure, chain_id))
    if len(residues) < len(source_ca):
        raise ValueError(
            f"Selected chain has fewer residues in full-atom PDB ({len(residues)}) than CA trace ({len(source_ca)})"
        )

    for residue, src_ca, dst_ca in zip(residues, source_ca, target_ca, strict=False):
        shift = dst_ca - src_ca
        for atom in residue.get_atoms():
            atom.coord = atom.coord + shift

    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


def _run_openmm_restrained_minimization(
    input_pdb: str,
    output_pdb: str,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
) -> dict[str, float | int | str]:
    try:
        import openmm
        from openmm import app, unit
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError(
            "OpenMM and PDBFixer are required for minimization in the current Python. "
            "Install them or pass --openmm_python pointing to an environment that has both."
        ) from exc

    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")
    fixer = PDBFixer(filename=input_pdb)
    fixer.removeHeterogens(False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoule_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    restrained_particles = 0
    for atom_idx, atom in enumerate(fixer.topology.atoms()):
        is_h = atom.element is not None and atom.element.symbol == "H"
        is_ca = atom.name == "CA"
        if restrain_selection == "heavy" and is_h:
            continue
        if restrain_selection == "ca" and not is_ca:
            continue
        pos_nm = fixer.positions[atom_idx].value_in_unit(unit.nanometer)
        restraint.addParticle(atom_idx, [pos_nm.x, pos_nm.y, pos_nm.z])
        restrained_particles += 1
    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.004 * unit.picoseconds,
    )
    simulation = app.Simulation(fixer.topology, system, integrator)
    simulation.context.setPositions(fixer.positions)
    before = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    after = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    os.makedirs(os.path.dirname(output_pdb) or ".", exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(fixer.topology, state.getPositions(), f, keepIds=True)

    return {
        "minimized_pdb": output_pdb,
        "restrained_particles": restrained_particles,
        "energy_before_kj_per_mol": before,
        "energy_after_kj_per_mol": after,
    }


def _run_external_openmm_minimize(
    *,
    python_bin: str,
    input_pdb: str,
    output_pdb: str,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
) -> dict[str, str]:
    code = r'''
import json
import sys
from openmm import app, unit
import openmm
from pdbfixer import PDBFixer

input_pdb, output_pdb = sys.argv[1], sys.argv[2]
restraint_k = float(sys.argv[3])
max_iterations = int(sys.argv[4])
restrain_selection = sys.argv[5]

forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")
fixer = PDBFixer(filename=input_pdb)
fixer.removeHeterogens(False)
fixer.findMissingResidues()
fixer.missingResidues = {}
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
system = forcefield.createSystem(
    fixer.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
)

restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", restraint_k * unit.kilojoule_per_mole / unit.nanometer**2)
restraint.addPerParticleParameter("x0")
restraint.addPerParticleParameter("y0")
restraint.addPerParticleParameter("z0")
restrained_particles = 0
for atom_idx, atom in enumerate(fixer.topology.atoms()):
    is_h = atom.element is not None and atom.element.symbol == "H"
    is_ca = atom.name == "CA"
    if restrain_selection == "heavy" and is_h:
        continue
    if restrain_selection == "ca" and not is_ca:
        continue
    pos_nm = fixer.positions[atom_idx].value_in_unit(unit.nanometer)
    restraint.addParticle(atom_idx, [pos_nm.x, pos_nm.y, pos_nm.z])
    restrained_particles += 1
system.addForce(restraint)

integrator = openmm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1 / unit.picosecond,
    0.004 * unit.picoseconds,
)
simulation = app.Simulation(fixer.topology, system, integrator)
simulation.context.setPositions(fixer.positions)
before = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
simulation.minimizeEnergy(maxIterations=max_iterations)
state = simulation.context.getState(getPositions=True, getEnergy=True)
after = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
with open(output_pdb, "w", encoding="utf-8") as handle:
    app.PDBFile.writeFile(fixer.topology, state.getPositions(), handle, keepIds=True)

print(json.dumps({
    "minimized_pdb": output_pdb,
    "openmm_python": sys.executable,
    "restrained_particles": restrained_particles,
    "energy_before_kj_per_mol": before,
    "energy_after_kj_per_mol": after,
}))
'''
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [python_bin, "-c", code, input_pdb, output_pdb, str(restraint_k), str(max_iterations), restrain_selection],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"External OpenMM minimization failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_relax_after_prediction(
    args: Any,
    selected_chain_id: str,
    source_center: np.ndarray,
    safe_center: np.ndarray,
    full_atom_pdb: str,
) -> dict[str, str]:
    output_dir = Path(args.relax_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "run_relax", True):
        return {}

    guardrailed_pdb = output_dir / "01_guardrailed_backbone.pdb"
    _write_guardrailed_pdb(args.pdb_file, selected_chain_id, source_center, safe_center, str(guardrailed_pdb))
    minimized_pdb = output_dir / "02_openmm_minimized.pdb"
    openmm_input = str(guardrailed_pdb)

    openmm_python = getattr(args, "openmm_python", None)
    if openmm_python:
        external = _run_external_openmm_minimize(
            python_bin=openmm_python,
            input_pdb=openmm_input,
            output_pdb=str(minimized_pdb),
            restraint_k=args.restraint_k,
            max_iterations=args.max_iterations,
            restrain_selection=args.restrain_selection,
        )
    else:
        external = _run_openmm_restrained_minimization(
            openmm_input,
            str(minimized_pdb),
            restraint_k=args.restraint_k,
            max_iterations=args.max_iterations,
            restrain_selection=args.restrain_selection,
        )

    artifacts = {
        "guardrailed_pdb": str(guardrailed_pdb),
        "minimized_pdb": str(minimized_pdb),
        "backend": "openmm",
        **external,
    }
    if full_atom_pdb:
        artifacts["output_pdb"] = full_atom_pdb
    return artifacts


def predict_and_relax(args: Any) -> dict:
    report, selected_chain_id, source_center, safe_center, parsed = predict_displacement(args)
    _print_prediction_report(report)

    if getattr(args, "report_json", None):
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report JSON written to: {args.report_json}")

    if getattr(args, "output_pdb", None):
        _write_guardrailed_pdb(args.pdb_file, selected_chain_id, source_center, safe_center, args.output_pdb)
        report["artifacts"]["output_pdb"] = args.output_pdb
        print(f"Predicted full-atom PDB written to: {args.output_pdb}")

    if getattr(args, "run_relax", True):
        full_atom_pdb = getattr(args, "output_pdb", None) or ""
        relax_artifacts = _run_relax_after_prediction(
            args=args,
            selected_chain_id=selected_chain_id,
            source_center=source_center,
            safe_center=safe_center,
            full_atom_pdb=full_atom_pdb,
        )
        if relax_artifacts:
            report["artifacts"]["relax"] = relax_artifacts
            print(f"[relax] Finished {relax_artifacts['backend']} relaxation.")
            print(f"[relax] OpenMM minimized: {relax_artifacts['minimized_pdb']}")

    if getattr(args, "report_json", None):
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report JSON updated: {args.report_json}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict HoloShift displacement and optionally run OpenMM relax.")
    p.add_argument("--pdb_file", required=True)
    p.add_argument(
        "--feature_pt",
        default=None,
        help=(
            "Prepared graph feature file with training-compatible x tensor. "
            "Required unless --allow_fallback_features is set for debugging."
        ),
    )
    p.add_argument(
        "--allow_fallback_features",
        action="store_true",
        help="Allow debug-only AA/pLDDT/coordinate fallback features when --feature_pt is omitted.",
    )
    p.add_argument(
        "--save_auto_feature_pt",
        default=None,
        help="Optional path to save debug-only auto-built fallback features (.pt).",
    )
    p.add_argument("--ckpt_path", required=True)
    p.add_argument(
        "--conformal_stats",
        default=None,
        help="Optional JSON generated by `python -m evopoint_da.pipeline.eval_run`; if omitted, reject is disabled.",
    )
    p.add_argument("--reject_threshold", type=float, default=5.0, help="Reject if qhat exceeds this threshold")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--chain_id", default=None, help="Optional chain ID to predict. Default: longest chain.")
    p.add_argument("--output_pdb", default=None, help="Optional output path for predicted full-atom PDB.")
    p.add_argument("--report_json", default=None, help="Optional path to save a detailed prediction report JSON.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--apply_inference_multiplier", action="store_true")
    p.add_argument(
        "--run_relax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OpenMM relax after prediction (default: enabled, disable via --no-run_relax).",
    )
    p.add_argument("--relax_output_dir", default="artifacts/relaxation", help="Output dir for relax artifacts.")
    p.add_argument(
        "--openmm_python",
        default=None,
        help="Optional Python executable from an environment with OpenMM and PDBFixer installed.",
    )
    p.add_argument("--restraint_k", type=float, default=1000.0, help="kJ/(mol*nm^2) for OpenMM restraints.")
    p.add_argument("--max_iterations", type=int, default=500, help="OpenMM minimization max iterations.")
    p.add_argument(
        "--restrain_selection",
        choices=["heavy", "ca"],
        default="heavy",
        help="Apply restraints to all heavy atoms or C-alpha atoms only.",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    predict_and_relax(parse_args(argv))


if __name__ == "__main__":
    main()
