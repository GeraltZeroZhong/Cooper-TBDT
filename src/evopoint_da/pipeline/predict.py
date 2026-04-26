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

from evopoint_da.data.structure import StructureParser, format_residue_id, select_chain
from evopoint_da.models.module import EvoPointLitModule
from evopoint_da.pipeline.build_prediction_features import PredictionFeatureBuildConfig, build_prediction_feature_graph
from evopoint_da.utils.binning import build_bin_ranges as _build_bin_ranges

DEFAULT_ESM_WEIGHTS = "esmc_weights/esmc_600m_2024_12_v0.pth"
DEFAULT_PCA_PATH = "data/pca_esmc_128.pkl"


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


def _load_prediction_features(
    args: Any,
    parsed: dict,
    pos: torch.Tensor,
    expected_in: int,
) -> dict[str, torch.Tensor | dict | None]:
    feature_pt = getattr(args, "feature_pt", None)
    if not feature_pt:
        raise ValueError(
            "--feature_pt is required. Build it with the training-compatible "
            "ESM/PCA + structural graph feature pipeline before prediction."
        )

    feature_path = Path(feature_pt)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature .pt not found: {feature_path}")

    feat = torch.load(feature_path, weights_only=True)
    required = ("x", "edge_index", "edge_attr", "residue_ids")
    missing = [key for key in required if key not in feat]
    if missing:
        raise ValueError(f"Feature .pt is missing required training graph fields: {missing}")

    x = feat["x"].float()
    edge_index = feat["edge_index"].long()
    edge_attr = feat["edge_attr"].float()

    parsed_residue_ids = list(parsed.get("residue_ids", []))
    feature_residue_ids = [str(value) for value in feat.get("residue_ids", [])]
    if feature_residue_ids != parsed_residue_ids:
        raise ValueError(
            "Feature residue_ids do not match the selected input chain. "
            f"feature_n={len(feature_residue_ids)}, pdb_n={len(parsed_residue_ids)}. "
            "Build features from the exact PDB/chain used for prediction."
        )

    feature_sequence = feat.get("sequence", None)
    parsed_sequence = str(parsed.get("sequence", ""))
    if feature_sequence is not None and str(feature_sequence) != parsed_sequence:
        raise ValueError("Feature sequence does not match the selected input chain sequence.")

    if x.size(0) != len(pos):
        raise ValueError(f"Feature length ({x.size(0)}) != selected chain length ({len(pos)}).")
    if x.size(1) != expected_in:
        raise ValueError(f"Feature dim drift: input feature_dim={x.size(1)} but checkpoint expects in_channels={expected_in}")

    edge_index, edge_attr = _feature_edges_for_node_count(edge_index, edge_attr, len(pos))
    if edge_index is None or edge_attr is None:
        raise ValueError("Training graph feature file must include non-null edge_index and edge_attr.")

    payload: dict[str, torch.Tensor | dict | None] = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_v": None,
        "edge_s": None,
        "edge_v": None,
        "plddt": feat.get("plddt", None),
        "metadata": feat.get("metadata", {}),
    }

    feature_pos = feat.get("pos", None)
    pos_matches = False
    if feature_pos is not None:
        feature_pos = feature_pos.float()
        if feature_pos.shape == pos.shape:
            max_delta = torch.linalg.vector_norm(feature_pos - pos, dim=-1).max().item() if pos.numel() else 0.0
            pos_matches = max_delta <= float(getattr(args, "feature_pos_tolerance", 1e-3))

    if pos_matches:
        for key in ("node_v", "edge_s", "edge_v"):
            value = feat.get(key, None)
            if value is not None:
                payload[key] = value.float()
    elif any(feat.get(key, None) is not None for key in ("node_v", "edge_s", "edge_v")):
        print(
            "[info] Feature pos differs from input PDB coordinates; rebuilding GVP vector features "
            "from the current PDB frame at model forward time.",
            file=sys.stderr,
        )
    return payload


def _path_or_none(value: str | os.PathLike | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def _resolve_feature_pt_path(args: Any) -> Path:
    feature_pt = _path_or_none(getattr(args, "feature_pt", None))
    if feature_pt is not None:
        return feature_pt

    feature_out = _path_or_none(getattr(args, "feature_out", None))
    if feature_out is not None:
        return feature_out

    pdb_stem = Path(args.pdb_file).stem
    report_json = _path_or_none(getattr(args, "report_json", None))
    output_pdb = _path_or_none(getattr(args, "output_pdb", None))
    if report_json is not None:
        out_dir = report_json.parent
    elif output_pdb is not None:
        out_dir = output_pdb.parent
    else:
        out_dir = Path("artifacts/prediction_features")
    return out_dir / f"{pdb_stem}_training_graph_features.pt"


def _ensure_prediction_feature_pt(args: Any, selected_chain_id: str) -> dict[str, Any] | None:
    """Resolve or build the strict training-compatible feature_pt for prediction."""
    feature_pt = _resolve_feature_pt_path(args)
    rebuild = bool(getattr(args, "rebuild_feature_pt", False))
    if feature_pt.exists() and not rebuild:
        args.feature_pt = str(feature_pt)
        return None

    esm_weights = _path_or_none(getattr(args, "esm_weights", DEFAULT_ESM_WEIGHTS))
    pca_path = _path_or_none(getattr(args, "pca_path", DEFAULT_PCA_PATH))
    missing_args = []
    if esm_weights is None:
        missing_args.append("--esm_weights")
    if pca_path is None:
        missing_args.append("--pca_path")
    if missing_args:
        raise ValueError(
            f"{feature_pt} does not exist and automatic feature generation is missing "
            f"{', '.join(missing_args)}. Provide --feature_pt or the training ESM/PCA inputs."
        )

    pae_path = _path_or_none(getattr(args, "pae_path", None))
    feature_device = getattr(args, "feature_device", "auto")
    report = build_prediction_feature_graph(
        PredictionFeatureBuildConfig(
            pdb_file=Path(args.pdb_file),
            output_pt=feature_pt,
            esm_weights=esm_weights,
            pca_path=pca_path,
            pae_path=pae_path,
            chain_id=selected_chain_id,
            device=None if feature_device in (None, "", "auto") else str(feature_device),
            pca_dim=int(getattr(args, "feature_pca_dim", 128)),
            k=int(getattr(args, "k", 16)),
            contact_radius=float(getattr(args, "feature_contact_radius", 10.0)),
            surface_sasa_threshold=float(getattr(args, "feature_surface_sasa_threshold", 1.0)),
            require_pae=bool(getattr(args, "require_pae", False)),
        )
    )
    args.feature_pt = str(feature_pt)
    print(f"[features] Built training-compatible feature_pt: {feature_pt}")
    return report


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
        bad = int((~valid).sum().item())
        raise ValueError(
            f"Feature edge_index contains {bad} edges outside the selected node range 0..{n - 1}. "
            "Build features from the exact PDB/chain used for prediction."
        )
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
            "conformal_stats": getattr(args, "conformal_stats", None),
            "selected_chain": selected_chain_id,
            "k": int(args.k),
            "device": args.device,
        },
        "task_definition": {
            "name": "ligand_agnostic_ca_displacement",
            "description": (
                "Predicts C-alpha displacement for a selected protein chain from AF2-derived, "
                "training-compatible ESM/PCA + structural graph features. It is not a ligand-aware "
                "side-chain repacking or holo pocket reconstruction model."
            ),
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

    feature_build_report = _ensure_prediction_feature_pt(args, selected_chain_id)
    feature_payload = _load_prediction_features(args, parsed, pos, expected_in)
    data_kwargs = {
        "x": feature_payload["x"],
        "pos": pos,
        "edge_index": feature_payload["edge_index"],
        "edge_attr": feature_payload["edge_attr"],
    }
    for optional_key in ("node_v", "edge_s", "edge_v", "plddt"):
        value = feature_payload.get(optional_key)
        if isinstance(value, torch.Tensor):
            data_kwargs[optional_key] = value
    data = Data(**data_kwargs).to(args.device)

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
        observed_in=feature_payload["x"].size(1),
        pred_norm=pred_norm,
        safe_center=safe_center,
        disp_bin_edges=disp_bin_edges,
    )
    report["feature_metadata"] = feature_payload.get("metadata") or {}
    if feature_build_report is not None:
        report["feature_build_report"] = feature_build_report
    return report, selected_chain_id, source_center, safe_center, parsed


def _iter_chain_residues(structure, chain_id: str) -> Iterable:
    for model in structure:
        if chain_id in model:
            for residue in model[chain_id]:
                if residue.id[0] == " ":
                    yield residue
            return
    raise ValueError(f"Chain {chain_id} not found in {structure.id}")


def _write_guardrailed_pdb(
    input_pdb: str,
    chain_id: str,
    residue_ids: list[str],
    source_ca: np.ndarray,
    target_ca: np.ndarray,
    out_pdb: str,
    *,
    ca_tolerance: float = 1e-3,
) -> dict[str, Any]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb)

    residues_by_id = {}
    for residue in _iter_chain_residues(structure, chain_id):
        rid = format_residue_id(chain_id, int(residue.id[1]), str(residue.id[2]).strip())
        residues_by_id[rid] = residue

    if not (len(residue_ids) == len(source_ca) == len(target_ca)):
        raise ValueError(
            "residue_ids, source_ca, and target_ca must have identical lengths; "
            f"got {len(residue_ids)}, {len(source_ca)}, {len(target_ca)}."
        )

    max_ca_delta = 0.0
    shifted_residues = 0
    shifted_atoms = 0
    for rid, src_ca, dst_ca in zip(residue_ids, source_ca, target_ca, strict=True):
        residue = residues_by_id.get(rid)
        if residue is None:
            raise ValueError(f"Residue {rid!r} from the prediction trace is missing in full-atom chain {chain_id}.")
        if "CA" not in residue:
            raise ValueError(f"Residue {rid!r} has no CA atom in full-atom chain {chain_id}.")
        ca_delta = float(np.linalg.norm(np.asarray(residue["CA"].coord, dtype=np.float32) - src_ca))
        max_ca_delta = max(max_ca_delta, ca_delta)
        if ca_delta > ca_tolerance:
            raise ValueError(
                f"Residue {rid!r} CA coordinate differs from parsed source trace by {ca_delta:.6f} A; "
                "refusing to apply residue shifts with ambiguous alignment."
            )
        shift = dst_ca - src_ca
        for atom in residue.get_atoms():
            atom.coord = atom.coord + shift
            shifted_atoms += 1
        shifted_residues += 1

    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)
    return {
        "method": "ca_guided_rigid_residue_translation",
        "warning": (
            "All atoms in each residue are translated by the predicted CA displacement. "
            "Side-chain rotamers are not repacked; use restrained minimization and docking redocking gates "
            "before interpreting this as a receptor model."
        ),
        "chain_id": chain_id,
        "shifted_residues": shifted_residues,
        "shifted_atoms": shifted_atoms,
        "max_source_ca_alignment_error": max_ca_delta,
        "ca_tolerance": ca_tolerance,
    }


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
    residue_ids: list[str],
    source_center: np.ndarray,
    safe_center: np.ndarray,
    full_atom_pdb: str,
) -> dict[str, Any]:
    output_dir = Path(args.relax_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "run_relax", True):
        return {}

    guardrailed_pdb = output_dir / "01_guardrailed_backbone.pdb"
    displacement_write = _write_guardrailed_pdb(
        args.pdb_file,
        selected_chain_id,
        residue_ids,
        source_center,
        safe_center,
        str(guardrailed_pdb),
    )
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
        "displacement_write": displacement_write,
        **external,
    }
    if full_atom_pdb:
        artifacts["output_pdb"] = full_atom_pdb
    return artifacts


def predict_and_relax(args: Any) -> dict:
    report, selected_chain_id, source_center, safe_center, parsed = predict_displacement(args)
    residue_ids = list(parsed.get("residue_ids", []))
    _print_prediction_report(report)

    if getattr(args, "report_json", None):
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report JSON written to: {args.report_json}")

    if getattr(args, "output_pdb", None):
        displacement_write = _write_guardrailed_pdb(
            args.pdb_file,
            selected_chain_id,
            residue_ids,
            source_center,
            safe_center,
            args.output_pdb,
        )
        report["artifacts"]["output_pdb"] = args.output_pdb
        report["artifacts"]["displacement_write"] = displacement_write
        print(f"Predicted full-atom PDB written to: {args.output_pdb}")

    if getattr(args, "run_relax", True):
        full_atom_pdb = getattr(args, "output_pdb", None) or ""
        relax_artifacts = _run_relax_after_prediction(
            args=args,
            selected_chain_id=selected_chain_id,
            residue_ids=residue_ids,
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
            "Prepared graph feature file. If omitted or missing, predict.py builds it with "
            "the training-compatible ESM/PCA + structural graph feature pipeline."
        ),
    )
    p.add_argument(
        "--feature_out",
        default=None,
        help=(
            "Output path for auto-generated feature_pt when --feature_pt is omitted. "
            "Defaults next to --report_json/--output_pdb, or artifacts/prediction_features/."
        ),
    )
    p.add_argument("--rebuild_feature_pt", action="store_true", help="Regenerate feature_pt even if it already exists.")
    p.add_argument("--esm_weights", default=DEFAULT_ESM_WEIGHTS, help="ESMC weights for automatic feature generation.")
    p.add_argument("--pca_path", default=DEFAULT_PCA_PATH, help="Training PCA model for automatic feature generation.")
    p.add_argument("--pae_path", default=None, help="Optional AF2 PAE JSON/NPY for automatic graph edge features.")
    p.add_argument("--require_pae", action="store_true", help="Fail automatic feature generation if PAE is missing.")
    p.add_argument("--feature_device", default="auto", help="Device for ESM feature construction; use auto/cpu/cuda.")
    p.add_argument("--feature_pca_dim", type=int, default=128)
    p.add_argument("--feature_contact_radius", type=float, default=10.0)
    p.add_argument("--feature_surface_sasa_threshold", type=float, default=1.0)
    p.add_argument(
        "--feature_pos_tolerance",
        type=float,
        default=1e-3,
        help="Direct-coordinate tolerance for reusing precomputed GVP vector features from feature_pt.",
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
