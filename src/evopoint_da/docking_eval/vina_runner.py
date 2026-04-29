from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

from .chem import DockingBox

STANDARD_POLYMER_RESIDUES = {
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
    "HID",
    "HIE",
    "HIP",
    "ASH",
    "GLH",
    "LYN",
    "CYX",
    "ACE",
    "NME",
    "A",
    "C",
    "G",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
}

EXCLUDED_HETATM_RESNAMES = {
    "HOH",
    "WAT",
    "SOL",
    "DOD",
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "CS",
    "RB",
    "LI",
    "F",
    "BR",
    "I",
    "SO4",
    "PO4",
    "PEG",
    "GOL",
    "EDO",
    "DMS",
    "ACT",
    "ACY",
    "FMT",
    "MES",
    "TRS",
}

VINA_RESULT_RE = re.compile(
    r"^\s*REMARK\s+VINA\s+RESULT:\s+"
    r"(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def current_env_bin_dir() -> Path:
    return Path(sys.executable).resolve().parent


def ensure_current_env_bin_on_path() -> None:
    env_bin = str(current_env_bin_dir())
    entries = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    if env_bin not in entries:
        os.environ["PATH"] = os.pathsep.join([env_bin, *entries]) if entries else env_bin


def find_binary(name: str) -> str | None:
    ensure_current_env_bin_on_path()
    found = shutil.which(name)
    if found:
        return found
    candidate = current_env_bin_dir() / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def require_binary(name: str) -> str:
    path = find_binary(name)
    if path:
        return path
    raise RuntimeError(
        f"Required executable was not found: {name}. "
        f"Checked PATH and active Python environment bin directory: {current_env_bin_dir()}"
    )


def resolve_binary(name: str, *, dry_run: bool = False) -> str:
    path = find_binary(name)
    if path:
        return path
    if dry_run:
        return name
    return require_binary(name)


def run_command(args: list[str], *, log_path: str | Path | None = None, dry_run: bool = False) -> None:
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(f"COMMAND:\n{' '.join(args)}\n\n")
    if dry_run:
        return

    proc = subprocess.run(args, capture_output=True, text=True)
    if log_path is not None:
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _parse_coord_triplet_from_pdb_line(line: str) -> tuple[float, float, float]:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        parts = line.split()
        if len(parts) < 9:
            raise
        return float(parts[6]), float(parts[7]), float(parts[8])


def _iter_pdb_atoms(input_pdb: str | Path):
    with Path(input_pdb).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x, y, z = _parse_coord_triplet_from_pdb_line(line)
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip().upper()
            element = line[76:78].strip().upper() or atom_name[:1]
            yield {
                "record": line[:6].strip().upper(),
                "resname": resname,
                "chain": line[21].strip() or "_",
                "resseq": line[22:26].strip(),
                "icode": line[26].strip(),
                "atom_name": atom_name,
                "element": element,
                "coord": np.array([x, y, z], dtype=float),
            }


def infer_box_from_bound_heterogen_pdb(
    input_pdb: str | Path,
    *,
    padding_angstrom: float = 8.0,
    min_size_angstrom: float = 16.0,
) -> DockingBox:
    ligand_groups: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    for atom in _iter_pdb_atoms(input_pdb):
        if atom["record"] != "HETATM":
            continue
        if atom["resname"] in EXCLUDED_HETATM_RESNAMES:
            continue
        if atom["element"] in {"H", "D"}:
            continue
        key = (atom["resname"], atom["chain"], atom["resseq"], atom["icode"])
        ligand_groups[key].append(atom["coord"])

    if ligand_groups:
        _, coords = max(ligand_groups.items(), key=lambda item: len(item[1]))
        arr = np.vstack(coords)
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

    raise ValueError(
        f"No bound heterogen suitable for docking-box inference was found in {input_pdb}. "
        "Provide explicit center/size columns or a reference ligand SDF."
    )


def _parse_occupancy_from_pdb_line(line: str) -> float:
    try:
        return float(line[54:60])
    except Exception:
        return 0.0


def _is_hydrogen_pdb_atom(line: str) -> bool:
    atom_name = line[12:16].strip().upper()
    element = line[76:78].strip().upper()
    return element in {"H", "D"} or atom_name.startswith(("H", "D"))


def sanitize_receptor_for_docking(input_pdb: str | Path, output_pdb: str | Path) -> Path:
    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {input_path}")
    output_path = Path(output_pdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str, str, str], list[tuple[str, float]]] = defaultdict(list)
    order: list[tuple[str, str, str, str, str]] = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if _is_hydrogen_pdb_atom(line):
                continue
            resname = line[17:20].strip().upper()
            if resname and resname not in STANDARD_POLYMER_RESIDUES:
                continue
            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            atom_name = line[12:16]
            if atom_name.strip().upper() == "OXT":
                continue
            key = (chain, resseq, icode, resname, atom_name)
            if key not in grouped:
                order.append(key)
            grouped[key].append((line, _parse_occupancy_from_pdb_line(line)))

    if not grouped:
        raise ValueError(f"No polymer ATOM records remained after receptor sanitization: {input_path}")

    def choose_record(records: list[tuple[str, float]]) -> str:
        def sort_key(item: tuple[str, float]) -> tuple[int, int, float, str]:
            line, occupancy = item
            altloc = line[16].strip()
            pref_blank = 0 if altloc == "" else 1
            pref_a = 0 if altloc == "A" else 1
            return (pref_blank, pref_a, -occupancy, altloc)

        return sorted(records, key=sort_key)[0][0]

    serial = 1
    last_residue: tuple[str, str, str, str] | None = None
    lines_out: list[str] = []
    for key in order:
        line = choose_record(grouped[key])
        chain, resseq, icode, resname, _atom_name = key
        residue_id = (chain, resseq, icode, resname)
        if last_residue is not None and chain != last_residue[0]:
            prev_chain, prev_resseq, prev_icode, prev_resname = last_residue
            ter = (
                f"TER   {serial:>5d}      {prev_resname:>3s} "
                f"{(prev_chain if prev_chain.strip() else ' ')}{prev_resseq}{prev_icode}\n"
            )
            lines_out.append(ter)
            serial += 1

        chars = list(line.rstrip("\n"))
        if len(chars) < 80:
            chars.extend([" "] * (80 - len(chars)))
        chars[0:6] = list("ATOM  ")
        chars[6:11] = list(f"{serial:>5d}")
        chars[16] = " "
        lines_out.append("".join(chars).rstrip() + "\n")
        serial += 1
        last_residue = residue_id

    if last_residue is not None:
        prev_chain, prev_resseq, prev_icode, prev_resname = last_residue
        ter = (
            f"TER   {serial:>5d}      {prev_resname:>3s} "
            f"{(prev_chain if prev_chain.strip() else ' ')}{prev_resseq}{prev_icode}\n"
        )
        lines_out.append(ter)
    lines_out.append("END\n")
    output_path.write_text("".join(lines_out), encoding="utf-8")
    return output_path


def _has_unrecognized_argument_error(exc: RuntimeError, flag: str) -> bool:
    return f"unrecognized arguments: {flag}" in str(exc)


def prepare_receptor_with_meeko(
    input_pdb: str | Path,
    receptor_pdbqt: str | Path,
    receptor_json: str | Path,
    log_path: str | Path,
    *,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    binary = resolve_binary("mk_prepare_receptor.py", dry_run=dry_run)
    receptor_pdbqt = Path(receptor_pdbqt)
    receptor_json = Path(receptor_json)
    sanitized_pdb = receptor_pdbqt.with_name("receptor_for_meeko.pdb")
    sanitize_receptor_for_docking(input_pdb, sanitized_pdb)
    receptor_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    receptor_json.parent.mkdir(parents=True, exist_ok=True)

    base_args = [
        binary,
        "--read_pdb",
        str(sanitized_pdb),
        "--write_pdbqt",
        str(receptor_pdbqt),
        "--write_json",
        str(receptor_json),
    ]

    for bad_res_flag in ("--allow_bad_res", "--delete_bad_res"):
        try:
            run_command([*base_args, bad_res_flag], log_path=log_path, dry_run=dry_run)
            return receptor_pdbqt, receptor_json
        except RuntimeError as exc:
            if not _has_unrecognized_argument_error(exc, bad_res_flag):
                raise

    raise RuntimeError(
        "mk_prepare_receptor.py rejected both supported bad-residue flags "
        "('--allow_bad_res' and '--delete_bad_res')."
    )


def prepare_ligand_with_meeko(
    input_sdf: str | Path,
    ligand_pdbqt: str | Path,
    log_path: str | Path,
    *,
    dry_run: bool = False,
) -> Path:
    binary = resolve_binary("mk_prepare_ligand.py", dry_run=dry_run)
    ligand_pdbqt = Path(ligand_pdbqt)
    ligand_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_command([binary, "-i", str(input_sdf), "-o", str(ligand_pdbqt)], log_path=log_path, dry_run=dry_run)
    except RuntimeError:
        obabel = find_binary("obabel")
        if obabel is None:
            raise
        run_command([obabel, str(input_sdf), "-O", str(ligand_pdbqt)], log_path=log_path, dry_run=dry_run)
    return ligand_pdbqt


def run_vina(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    box: DockingBox,
    output_pdbqt: str | Path,
    log_path: str | Path,
    *,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 3.0,
    seed: int = 20260408,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    binary = resolve_binary("vina", dry_run=dry_run)
    output_pdbqt = Path(output_pdbqt)
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{box.center_x:.3f}",
        "--center_y",
        f"{box.center_y:.3f}",
        "--center_z",
        f"{box.center_z:.3f}",
        "--size_x",
        f"{box.size_x:.3f}",
        "--size_y",
        f"{box.size_y:.3f}",
        "--size_z",
        f"{box.size_z:.3f}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--energy_range",
        f"{energy_range:.3f}",
        "--seed",
        str(seed),
        "--out",
        str(output_pdbqt),
    ]
    run_command(cmd, log_path=log_path, dry_run=dry_run)
    return output_pdbqt, Path(log_path)


def export_docking_results_with_meeko(
    docking_pdbqt: str | Path,
    output_sdf: str | Path,
    log_path: str | Path,
    *,
    dry_run: bool = False,
) -> Path:
    binary = resolve_binary("mk_export.py", dry_run=dry_run)
    output_sdf = Path(output_sdf)
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_command([binary, str(docking_pdbqt), "-s", str(output_sdf)], log_path=log_path, dry_run=dry_run)
    except RuntimeError:
        obabel = find_binary("obabel")
        if obabel is None:
            raise
        run_command([obabel, str(docking_pdbqt), "-O", str(output_sdf)], log_path=log_path, dry_run=dry_run)
    return output_sdf


def parse_vina_pdbqt_scores(path: str | Path) -> list[float]:
    scores: list[float] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = VINA_RESULT_RE.match(line)
            if match:
                scores.append(float(match.group("score")))
    return scores


def write_box_config(box: DockingBox, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"center_x = {box.center_x:.3f}",
        f"center_y = {box.center_y:.3f}",
        f"center_z = {box.center_z:.3f}",
        f"size_x = {box.size_x:.3f}",
        f"size_y = {box.size_y:.3f}",
        f"size_z = {box.size_z:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
