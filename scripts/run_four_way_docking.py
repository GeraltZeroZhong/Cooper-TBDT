#!/usr/bin/env python
"""Run four-way docking comparison for Raw AF2, AF2+OpenMM, HoloShift, and true holo.

The script wraps evopoint_da.docking_eval.pipeline so the same docking box,
ligand preparation, Vina settings, RMSD threshold, and summaries are used for
all four receptor inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evopoint_da.docking_eval.io_utils import read_table, write_csv
from evopoint_da.docking_eval.pipeline import DockingPipelineConfig, StructureSpec, run_docking_pipeline
from evopoint_da.pipeline.predict import predict_and_relax
from get_af2 import download_af2_model

STRUCTURES = [
    ("raw_af2", "receptor_raw_af2", "Raw AF2"),
    ("af2_openmm_relax", "receptor_af2_openmm_relax", "AF2 + OpenMM Relax"),
    ("holoshift", "receptor_holoshift", "HoloShift (Ours)"),
    ("true_holo", "receptor_true_holo", "True Holo PDB"),
]

STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
}


@dataclass(frozen=True)
class PreparedInputs:
    manifest: Path
    raw_af2: Path
    af2_openmm_relax: Path
    holoshift: Path
    true_holo: Path
    notes: list[str]


class _ResidueSubsetSelect(Select):
    def __init__(self, chain_id: str, residue_keys: set[tuple[int, str]]) -> None:
        self.chain_id = chain_id
        self.residue_keys = residue_keys

    def accept_chain(self, chain) -> bool:  # noqa: ANN001
        return chain.id == self.chain_id

    def accept_residue(self, residue) -> bool:  # noqa: ANN001
        if residue.id[0] != " ":
            return False
        key = (int(residue.id[1]), str(residue.id[2]).strip())
        return key in self.residue_keys


def _path(raw: str | Path | None) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    return Path(raw).expanduser().resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_af2(uniprot_id: str, version: int, output_dir: Path, *, reuse: bool = True) -> Path:
    uniprot_id = uniprot_id.strip()
    if not uniprot_id:
        raise ValueError("--raw-af2-uniprot cannot be empty.")
    output_path = output_dir / f"AF-{uniprot_id}-F1-model_v{version}.pdb"
    if reuse and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    downloaded = download_af2_model(
        uniprot_id,
        str(output_dir),
        version=version,
        filename=output_path.name,
    )
    if downloaded is None:
        raise FileNotFoundError(f"AlphaFold model was not found for UniProt {uniprot_id} v{version}.")
    return Path(downloaded)


def _get_chain(structure, chain_id: str):  # noqa: ANN001
    model = next(structure.get_models())
    if chain_id in model:
        return model[chain_id]
    available = ", ".join(chain.id for chain in model)
    raise ValueError(f"Chain {chain_id!r} not found in {structure.id}; available chains: {available}")


def _ca_residue_map(pdb_path: Path, chain_id: str) -> dict[tuple[int, str], tuple[Any, str]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chain = _get_chain(structure, chain_id)
    mapping: dict[tuple[int, str], tuple[Any, str]] = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if residue.get_resname().strip().upper() not in STANDARD_AA:
            continue
        if "CA" not in residue:
            continue
        key = (int(residue.id[1]), str(residue.id[2]).strip())
        mapping[key] = (residue["CA"], residue.get_resname().strip().upper())
    return mapping


def align_crop_to_reference(
    *,
    moving_pdb: Path,
    reference_pdb: Path,
    output_pdb: Path,
    moving_chain: str,
    reference_chain: str,
    min_ca_pairs: int = 10,
    match_residue_names: bool = True,
) -> dict[str, Any]:
    parser = PDBParser(QUIET=True)
    moving_structure = parser.get_structure(moving_pdb.stem, str(moving_pdb))
    reference_structure = parser.get_structure(reference_pdb.stem, str(reference_pdb))

    moving_map = _ca_residue_map(moving_pdb, moving_chain)
    reference_map = _ca_residue_map(reference_pdb, reference_chain)
    common_keys = sorted(set(moving_map) & set(reference_map))
    if not common_keys:
        raise ValueError(f"No common CA residue ids between {moving_pdb} and {reference_pdb}.")

    align_keys = common_keys
    if match_residue_names:
        name_matched = [key for key in common_keys if moving_map[key][1] == reference_map[key][1]]
        if len(name_matched) >= min_ca_pairs:
            align_keys = name_matched

    if len(align_keys) < min_ca_pairs:
        raise ValueError(
            f"Only {len(align_keys)} common CA atoms available for alignment; "
            f"need at least {min_ca_pairs}."
        )

    fixed_atoms = [reference_map[key][0] for key in align_keys]
    moving_atoms = [moving_map[key][0] for key in align_keys]
    superimposer = Superimposer()
    superimposer.set_atoms(fixed_atoms, moving_atoms)
    superimposer.apply(list(moving_structure.get_atoms()))

    _ensure_parent(output_pdb)
    io = PDBIO()
    io.set_structure(moving_structure)
    io.save(str(output_pdb), _ResidueSubsetSelect(moving_chain, set(common_keys)))

    return {
        "input": str(moving_pdb),
        "reference": str(reference_pdb),
        "output": str(output_pdb),
        "moving_chain": moving_chain,
        "reference_chain": reference_chain,
        "common_ca_residues": len(common_keys),
        "alignment_ca_residues": len(align_keys),
        "alignment_rmsd": float(superimposer.rms),
    }


def openmm_relax_protein(
    *,
    input_pdb: Path,
    output_pdb: Path,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
    openmm_python: Path | None = None,
    ph: float = 7.0,
) -> dict[str, Any]:
    try:
        import openmm
        from openmm import app, unit
        from pdbfixer import PDBFixer
    except ImportError as exc:
        if openmm_python is not None:
            return _openmm_relax_protein_external(
                input_pdb=input_pdb,
                output_pdb=output_pdb,
                restraint_k=restraint_k,
                max_iterations=max_iterations,
                restrain_selection=restrain_selection,
                openmm_python=openmm_python,
                ph=ph,
            )
        raise ImportError(
            "OpenMM and pdbfixer are required for relaxation in the current Python. "
            "Alternatively pass --openmm-python /path/to/python from an env that has them."
        ) from exc

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.removeHeterogens(False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    forcefield = app.ForceField("amber14/protein.ff14SB.xml")
    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", float(restraint_k) * unit.kilojoule_per_mole / unit.nanometer**2)
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
    simulation.minimizeEnergy(maxIterations=int(max_iterations))
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    after = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    _ensure_parent(output_pdb)
    with output_pdb.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(fixer.topology, state.getPositions(), handle, keepIds=True)

    return {
        "input": str(input_pdb),
        "output": str(output_pdb),
        "restraint_k": float(restraint_k),
        "restrain_selection": restrain_selection,
        "max_iterations": int(max_iterations),
        "restrained_particles": int(restrained_particles),
        "energy_before_kj_per_mol": float(before),
        "energy_after_kj_per_mol": float(after),
    }


def _openmm_relax_protein_external(
    *,
    input_pdb: Path,
    output_pdb: Path,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
    openmm_python: Path,
    ph: float,
) -> dict[str, Any]:
    code = r'''
import json
import sys
from pathlib import Path

import openmm
from openmm import app, unit
from pdbfixer import PDBFixer

input_pdb = Path(sys.argv[1])
output_pdb = Path(sys.argv[2])
restraint_k = float(sys.argv[3])
max_iterations = int(sys.argv[4])
restrain_selection = sys.argv[5]
ph = float(sys.argv[6])

fixer = PDBFixer(filename=str(input_pdb))
fixer.removeHeterogens(False)
fixer.findMissingResidues()
fixer.missingResidues = {}
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(ph)

forcefield = app.ForceField("amber14/protein.ff14SB.xml")
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

output_pdb.parent.mkdir(parents=True, exist_ok=True)
with output_pdb.open("w", encoding="utf-8") as handle:
    app.PDBFile.writeFile(fixer.topology, state.getPositions(), handle, keepIds=True)

print(json.dumps({
    "input": str(input_pdb),
    "output": str(output_pdb),
    "restraint_k": restraint_k,
    "restrain_selection": restrain_selection,
    "max_iterations": max_iterations,
    "restrained_particles": restrained_particles,
    "energy_before_kj_per_mol": before,
    "energy_after_kj_per_mol": after,
    "openmm_python": sys.executable,
}))
'''
    cmd = [
        str(openmm_python),
        "-c",
        code,
        str(input_pdb),
        str(output_pdb),
        str(restraint_k),
        str(max_iterations),
        restrain_selection,
        str(ph),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "External OpenMM relaxation failed.\n"
            f"COMMAND: {' '.join(cmd[:3])} ...\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"External OpenMM relaxation did not return JSON.\nSTDOUT:\n{proc.stdout}") from exc


def generate_holoshift_receptor(
    *,
    input_pdb: Path,
    output_pdb: Path,
    report_path: Path,
    checkpoint: Path,
    feature_pt: Path | None,
    chain_id: str,
    k: int,
    device: str,
    allow_fallback_features: bool,
    conformal_stats: Path | None,
    reject_threshold: float,
    apply_inference_multiplier: bool,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
    openmm_python: Path | None,
    reuse: bool,
) -> dict[str, Any]:
    if reuse and output_pdb.exists() and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    predicted_full_atom = output_pdb.with_name(output_pdb.stem + "_predicted_unrelaxed.pdb")
    relax_dir = output_pdb.with_name(output_pdb.stem + "_relax")
    resolved_device = device
    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    predict_args = argparse.Namespace(
        pdb_file=str(input_pdb),
        feature_pt=str(feature_pt) if feature_pt else None,
        allow_fallback_features=allow_fallback_features,
        save_auto_feature_pt=None,
        ckpt_path=str(checkpoint),
        conformal_stats=str(conformal_stats) if conformal_stats else None,
        reject_threshold=reject_threshold,
        k=k,
        chain_id=chain_id,
        output_pdb=str(predicted_full_atom),
        report_json=str(report_path),
        device=resolved_device,
        apply_inference_multiplier=apply_inference_multiplier,
        run_relax=True,
        relax_output_dir=str(relax_dir),
        openmm_python=str(openmm_python) if openmm_python else None,
        restraint_k=restraint_k,
        max_iterations=max_iterations,
        restrain_selection=restrain_selection,
    )
    report = predict_and_relax(predict_args)
    minimized = report.get("artifacts", {}).get("relax", {}).get("minimized_pdb")
    if not minimized:
        raise RuntimeError("HoloShift prediction did not produce an OpenMM-minimized receptor.")
    shutil.copyfile(minimized, output_pdb)
    pred_summary = report.get("prediction_summary", {})
    report.update(
        {
            "input_pdb": str(input_pdb),
            "predicted_unrelaxed_pdb": str(predicted_full_atom),
            "output_pdb": str(output_pdb),
            "checkpoint": str(checkpoint),
            "feature_pt": str(feature_pt) if feature_pt else "",
            "feature_source": "feature_pt" if feature_pt else "fallback_aa_plddt_coord",
            "selected_chain": chain_id,
            "device": resolved_device,
            "n_nodes": pred_summary.get("count", 0),
            "conformal_qhat": report.get("conformal", {}).get("qhat"),
            "conformal_rejected": report.get("conformal", {}).get("decision") == "REJECT",
            "apply_inference_multiplier": bool(apply_inference_multiplier),
            "prediction_delta_norm": {
                "mean": pred_summary.get("mean_abs_dr", 0.0),
                "median": pred_summary.get("median_abs_dr", 0.0),
                "p90": pred_summary.get("p90_abs_dr", 0.0),
                "max": pred_summary.get("max_abs_dr", 0.0),
            },
            "openmm_relax": report.get("artifacts", {}).get("relax", {}),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_manifest(path: Path, row: dict[str, str]) -> Path:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def prepare_single_target_inputs(args: argparse.Namespace) -> PreparedInputs:
    out_dir = _path(args.out_dir) or REPO_ROOT / "outputs/docking_four_way"
    work_dir = _path(args.work_dir) or out_dir / "prepared_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    true_holo = _path(args.true_holo_pdb)
    ligand_sdf = _path(args.ligand_sdf)
    reference_ligand_sdf = _path(args.reference_ligand_sdf)
    if true_holo is None or ligand_sdf is None or reference_ligand_sdf is None:
        raise ValueError("--true-holo-pdb, --ligand-sdf, and --reference-ligand-sdf are required without --manifest.")

    raw_af2_source = _path(args.raw_af2_pdb)
    if raw_af2_source is None:
        if not args.raw_af2_uniprot:
            raise ValueError("Provide --raw-af2-pdb or --raw-af2-uniprot without --manifest.")
        raw_af2_source = _download_af2(
            args.raw_af2_uniprot,
            int(args.af2_version),
            work_dir / "downloads",
            reuse=args.reuse_prepared,
        )
        notes.append(f"Downloaded Raw AF2 from AlphaFold DB: {raw_af2_source}")

    raw_af2 = raw_af2_source
    alignment_reports: list[dict[str, Any]] = []
    if args.align_crop:
        raw_af2 = work_dir / f"{args.target_id}_raw_af2_aligned_cropped.pdb"
        report = align_crop_to_reference(
            moving_pdb=raw_af2_source,
            reference_pdb=true_holo,
            output_pdb=raw_af2,
            moving_chain=args.moving_chain,
            reference_chain=args.reference_chain,
            min_ca_pairs=args.min_ca_pairs,
            match_residue_names=not args.no_match_residue_names,
        )
        alignment_reports.append({"label": "raw_af2", **report})

    af2_relax = _path(args.af2_openmm_relax_pdb)
    if af2_relax is None:
        af2_relax = work_dir / f"{args.target_id}_af2_openmm_relax.pdb"
        if not args.reuse_prepared or not af2_relax.exists():
            relax_report = openmm_relax_protein(
                input_pdb=raw_af2,
                output_pdb=af2_relax,
                restraint_k=args.openmm_restraint_k,
                max_iterations=args.openmm_max_iterations,
                restrain_selection=args.openmm_restrain_selection,
                openmm_python=_path(args.openmm_python),
            )
            (work_dir / f"{args.target_id}_openmm_relax_report.json").write_text(
                json.dumps(relax_report, indent=2),
                encoding="utf-8",
            )
        notes.append(f"Generated AF2 + OpenMM Relax receptor: {af2_relax}")

    holoshift_source = _path(args.holoshift_pdb)
    if holoshift_source is None:
        holoshift_ckpt = _path(args.holoshift_ckpt)
        if holoshift_ckpt is not None:
            holoshift = work_dir / f"{args.target_id}_holoshift_predict_openmm_relax.pdb"
            feature_pt = _path(args.holoshift_feature_pt)
            conformal_stats = _path(args.holoshift_conformal_stats)
            report = generate_holoshift_receptor(
                input_pdb=raw_af2,
                output_pdb=holoshift,
                report_path=work_dir / f"{args.target_id}_holoshift_predict_report.json",
                checkpoint=holoshift_ckpt,
                feature_pt=feature_pt,
                chain_id=args.moving_chain,
                k=args.holoshift_k,
                device=args.holoshift_device,
                allow_fallback_features=args.holoshift_allow_fallback_features,
                conformal_stats=conformal_stats,
                reject_threshold=args.holoshift_reject_threshold,
                apply_inference_multiplier=args.holoshift_apply_inference_multiplier,
                restraint_k=args.holoshift_openmm_restraint_k,
                max_iterations=args.holoshift_openmm_max_iterations,
                restrain_selection=args.holoshift_openmm_restrain_selection,
                openmm_python=_path(args.openmm_python),
                reuse=args.reuse_prepared,
            )
            notes.append(
                "Generated HoloShift receptor via checkpoint "
                f"{holoshift_ckpt}; feature_source={report.get('feature_source')}, "
                f"mean|delta|={report.get('prediction_delta_norm', {}).get('mean', 0.0):.3f} A."
            )
            if report.get("feature_source") == "fallback_aa_plddt_coord":
                notes.append(
                    "HoloShift Predict used debug fallback features because no --holoshift-feature-pt was supplied; "
                    "use training-compatible graph features for a real benchmark."
                )
        else:
            raise ValueError("Provide --holoshift-pdb or --holoshift-ckpt to build the HoloShift receptor.")
    elif args.align_crop_holoshift:
        holoshift = work_dir / f"{args.target_id}_holoshift_aligned_cropped.pdb"
        report = align_crop_to_reference(
            moving_pdb=holoshift_source,
            reference_pdb=true_holo,
            output_pdb=holoshift,
            moving_chain=args.moving_chain,
            reference_chain=args.reference_chain,
            min_ca_pairs=args.min_ca_pairs,
            match_residue_names=not args.no_match_residue_names,
        )
        alignment_reports.append({"label": "holoshift", **report})
    else:
        holoshift = holoshift_source

    if alignment_reports:
        (work_dir / f"{args.target_id}_alignment_report.json").write_text(
            json.dumps(alignment_reports, indent=2),
            encoding="utf-8",
        )

    manifest = _path(args.manifest_out) or out_dir / "four_way_manifest.csv"
    row = {
        "target_id": args.target_id,
        "receptor_raw_af2": str(raw_af2),
        "receptor_af2_openmm_relax": str(af2_relax),
        "receptor_holoshift": str(holoshift),
        "receptor_true_holo": str(true_holo),
        "ligand_sdf": str(ligand_sdf),
        "reference_ligand_sdf": str(reference_ligand_sdf),
    }
    _write_manifest(manifest, row)
    return PreparedInputs(
        manifest=manifest,
        raw_af2=raw_af2,
        af2_openmm_relax=af2_relax,
        holoshift=holoshift,
        true_holo=true_holo,
        notes=notes,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def _target_score_rows_with_deltas(score_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in score_rows:
        out: dict[str, Any] = dict(row)
        raw_score = _float_or_none(row.get("score_raw_af2"))
        raw_rmsd = _float_or_none(row.get("top1_rmsd_raw_af2"))
        for label, _col, _display in STRUCTURES:
            if label == "raw_af2":
                continue
            score = _float_or_none(row.get(f"score_{label}"))
            rmsd = _float_or_none(row.get(f"top1_rmsd_{label}"))
            out[f"delta_score_{label}_vs_raw_af2"] = (
                score - raw_score if score is not None and raw_score is not None else ""
            )
            out[f"delta_top1_rmsd_{label}_vs_raw_af2"] = (
                rmsd - raw_rmsd if rmsd is not None and raw_rmsd is not None else ""
            )
        out_rows.append(out)
    return out_rows


def _four_way_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_structure = summary.get("by_structure", {})
    for label, _col, display in STRUCTURES:
        payload = by_structure.get(label, {})
        top1 = payload.get("top1_success", {}) if isinstance(payload, dict) else {}
        topn = payload.get("topn_success", {}) if isinstance(payload, dict) else {}
        first_hit = payload.get("first_hit_rank", {}) if isinstance(payload, dict) else {}
        row: dict[str, Any] = {
            "structure": label,
            "display_name": display,
            "n_targets": top1.get("n_targets", ""),
            "top1_success_rate": top1.get("success_rate", ""),
            "top1_success_percent": top1.get("success_rate_percent", ""),
            "top1_success_n": top1.get("n_success", ""),
            "top1_rmsd_mean": top1.get("rmsd_mean", ""),
            "top1_rmsd_median": top1.get("rmsd_median", ""),
            "first_hit_rank_mean": first_hit.get("first_hit_rank_mean", ""),
            "first_hit_miss_rate": first_hit.get("miss_rate", ""),
        }
        row.update(topn)
        rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    num = _float_or_none(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def write_four_way_reports(out_dir: Path, notes: list[str]) -> None:
    summary_path = out_dir / "summary.json"
    scores_path = out_dir / "scores.csv"
    if not summary_path.exists() or not scores_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    score_rows = read_table(scores_path)
    summary_rows = _four_way_summary_rows(summary)
    score_delta_rows = _target_score_rows_with_deltas(score_rows)
    write_csv(out_dir / "four_way_summary.csv", summary_rows)
    write_csv(out_dir / "four_way_scores.csv", score_delta_rows)

    lines = ["# Four-Way Docking Comparison", ""]
    if notes:
        lines += ["## Notes"]
        lines += [f"- {note}" for note in notes]
        lines.append("")

    lines += ["## Aggregate", "", "| Structure | Top-1 success | Mean Top-1 RMSD | Top-3 | Top-5 | First hit rank |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        success = _fmt(row.get("top1_success_percent"), 2)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["display_name"]),
                    f"{success}%" if success else "",
                    _fmt(row.get("top1_rmsd_mean"), 3),
                    _fmt(row.get("top3_success_rate"), 3),
                    _fmt(row.get("top5_success_rate"), 3),
                    _fmt(row.get("first_hit_rank_mean"), 2),
                ]
            )
            + " |"
        )

    if score_delta_rows:
        lines += ["", "## Per-Target Top-1"]
        header = [
            "target_id",
            "score_raw_af2",
            "score_af2_openmm_relax",
            "score_holoshift",
            "score_true_holo",
            "top1_rmsd_raw_af2",
            "top1_rmsd_af2_openmm_relax",
            "top1_rmsd_holoshift",
            "top1_rmsd_true_holo",
        ]
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in score_delta_rows:
            values = [str(row.get("target_id", ""))]
            for key in header[1:]:
                values.append(_fmt(row.get(key), 3))
            lines.append("| " + " | ".join(values) + " |")

    (out_dir / "four_way_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_topn_levels(raw: str) -> list[int]:
    levels = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("--topn-levels must contain positive integers.")
    return levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Raw AF2 / AF2+OpenMM / HoloShift / True Holo docking comparison.")
    p.add_argument("--manifest", type=Path, default=None, help="Existing four-way manifest. Skips receptor prep.")
    p.add_argument("--target-id", default="target")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/docking_four_way"))
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--manifest-out", type=Path, default=None)

    p.add_argument("--raw-af2-pdb", default=None)
    p.add_argument("--raw-af2-uniprot", default=None)
    p.add_argument("--af2-version", type=int, default=6)
    p.add_argument("--af2-openmm-relax-pdb", default=None)
    p.add_argument("--holoshift-pdb", default=None)
    p.add_argument("--holoshift-ckpt", default=None, help="Checkpoint used to generate HoloShift receptor if --holoshift-pdb is absent.")
    p.add_argument(
        "--holoshift-feature-pt",
        default=None,
        help="Training-compatible graph feature .pt for HoloShift inference.",
    )
    p.add_argument(
        "--holoshift-allow-fallback-features",
        action="store_true",
        help="Use debug-only AA/pLDDT/coordinate features when --holoshift-feature-pt is unavailable.",
    )
    p.add_argument("--holoshift-conformal-stats", default=None)
    p.add_argument("--holoshift-reject-threshold", type=float, default=5.0)
    p.add_argument("--holoshift-device", default="auto")
    p.add_argument("--holoshift-k", type=int, default=16)
    p.add_argument("--holoshift-apply-inference-multiplier", action="store_true")
    p.add_argument("--holoshift-openmm-restraint-k", type=float, default=1000.0)
    p.add_argument("--holoshift-openmm-max-iterations", type=int, default=200)
    p.add_argument("--holoshift-openmm-restrain-selection", choices=["heavy", "ca"], default="heavy")
    p.add_argument("--true-holo-pdb", default=None)
    p.add_argument("--ligand-sdf", default=None)
    p.add_argument("--reference-ligand-sdf", default=None)
    p.add_argument("--align-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--align-crop-holoshift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reference-chain", default="A")
    p.add_argument("--moving-chain", default="A")
    p.add_argument("--min-ca-pairs", type=int, default=10)
    p.add_argument("--no-match-residue-names", action="store_true")
    p.add_argument("--reuse-prepared", action="store_true")

    p.add_argument(
        "--openmm-python",
        default=None,
        help="Optional Python executable from an environment with openmm+pdbfixer; used when current Python lacks them.",
    )
    p.add_argument("--openmm-restraint-k", type=float, default=1000.0)
    p.add_argument("--openmm-max-iterations", type=int, default=200)
    p.add_argument("--openmm-restrain-selection", choices=["heavy", "ca"], default="heavy")

    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--topn-levels", default="1,2,3,5,10")
    p.add_argument("--box-padding-angstrom", type=float, default=8.0)
    p.add_argument("--box-min-size-angstrom", type=float, default=16.0)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=9)
    p.add_argument("--energy-range", type=float, default=3.0)
    p.add_argument("--vina-seed", type=int, default=20260408)
    p.add_argument("--ligand-seed", type=int, default=42)
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--reuse", action="store_true", help="Reuse docking intermediates.")
    p.add_argument("--skip-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    notes: list[str] = []
    if args.manifest:
        manifest = _path(args.manifest)
        if manifest is None:
            raise ValueError("--manifest cannot be empty.")
    else:
        prepared = prepare_single_target_inputs(args)
        manifest = prepared.manifest
        notes.extend(prepared.notes)

    out_dir = _path(args.out_dir)
    if out_dir is None:
        raise ValueError("--out-dir cannot be empty.")

    cfg = DockingPipelineConfig(
        manifest=manifest,
        output_dir=out_dir,
        structures=[StructureSpec(label=label, receptor_col=col) for label, col, _display in STRUCTURES],
        rmsd_threshold=args.rmsd_threshold,
        topn_levels=_parse_topn_levels(args.topn_levels),
        box_padding_angstrom=args.box_padding_angstrom,
        box_min_size_angstrom=args.box_min_size_angstrom,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        vina_seed=args.vina_seed,
        ligand_seed=args.ligand_seed,
        bootstrap_iter=args.bootstrap_iter,
        bootstrap_seed=args.bootstrap_seed,
        reuse=args.reuse,
        skip_failed=args.skip_failed,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    summary = run_docking_pipeline(cfg)
    write_four_way_reports(out_dir, notes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Four-way report: {out_dir / 'four_way_report.md'}")


if __name__ == "__main__":
    main()
