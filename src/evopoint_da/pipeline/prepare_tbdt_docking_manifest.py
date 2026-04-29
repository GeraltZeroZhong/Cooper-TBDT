from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from evopoint_da.docking_eval.io_utils import write_csv
from evopoint_da.docking_eval.vina_runner import EXCLUDED_HETATM_RESNAMES


DEFAULT_EXCLUDED_LIGANDS = EXCLUDED_HETATM_RESNAMES | {
    "C8E",
    "LDA",
    "OCT",
    "HEX",
    "GOL",
    "MPD",
    "MPG",
    "MSE",
    "MTN",
    "OES",
    "PEG",
    "SO4",
    "PO4",
}


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sample_ids(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _parse_override(raw: str) -> tuple[str, tuple[str, str, str, str]]:
    if "=" not in raw:
        raise ValueError(f"Override must be pair_id=RES:CHAIN:RESSEQ[:ICODE], got {raw!r}")
    pair_id, spec = raw.split("=", 1)
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(f"Override must be pair_id=RES:CHAIN:RESSEQ[:ICODE], got {raw!r}")
    resname, chain, resseq = parts[:3]
    icode = parts[3] if len(parts) == 4 else ""
    return pair_id, (resname.upper(), chain, resseq, icode)


def _pdb_atom_serial(line: str) -> int | None:
    try:
        return int(line[6:11])
    except ValueError:
        return None


def _pdb_atom_coord(line: str) -> tuple[float, float, float] | None:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None


def _format_pdb_coord_line(line: str, xyz: tuple[float, float, float]) -> str:
    chars = list(line.rstrip("\n"))
    if len(chars) < 80:
        chars.extend([" "] * (80 - len(chars)))
    x, y, z = xyz
    chars[30:38] = list(f"{x:8.3f}")
    chars[38:46] = list(f"{y:8.3f}")
    chars[46:54] = list(f"{z:8.3f}")
    return "".join(chars).rstrip() + "\n"


def _parse_resseq(line: str) -> int | None:
    try:
        return int(line[22:26])
    except ValueError:
        return None


def _copy_af2_receptor(af2_pdb: Path, out_pdb: Path) -> None:
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in af2_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(True) if line.startswith(("ATOM", "TER", "END"))]
    out_pdb.write_text("".join(lines), encoding="utf-8")


def _write_cooper_receptor(
    *,
    af2_pdb: Path,
    out_pdb: Path,
    sample: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    af2_indices = sample.get("af2_indices")
    pred_delta = prediction.get("pred_delta")
    if af2_indices is None or pred_delta is None:
        raise ValueError("Sample must contain af2_indices and prediction must contain pred_delta.")
    deltas = {
        int(idx) + 1: tuple(float(x) for x in pred_delta[i].detach().cpu().tolist())
        for i, idx in enumerate(af2_indices.detach().cpu().tolist())
    }

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    for line in af2_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        if line.startswith("ATOM"):
            resseq = _parse_resseq(line)
            coord = _pdb_atom_coord(line)
            if resseq is not None and coord is not None and resseq in deltas:
                dx, dy, dz = deltas[resseq]
                line = _format_pdb_coord_line(line, (coord[0] + dx, coord[1] + dy, coord[2] + dz))
            out_lines.append(line)
        elif line.startswith(("TER", "END")):
            out_lines.append(line)
    if not out_lines or not any(line.startswith("END") for line in out_lines):
        out_lines.append("END\n")
    out_pdb.write_text("".join(out_lines), encoding="utf-8")


def _write_holo_chain_receptor(holo_pdb: Path, chain: str, out_pdb: Path) -> None:
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for line in holo_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        if line.startswith("ATOM") and (line[21].strip() or "_") == chain:
            selected.append(line)
    if not selected:
        raise ValueError(f"No ATOM records for chain {chain!r} in {holo_pdb}")
    selected.append("END\n")
    out_pdb.write_text("".join(selected), encoding="utf-8")


def _ligand_instances(pdb_path: Path) -> dict[tuple[str, str, str, str], list[str]]:
    groups: dict[tuple[str, str, str, str], list[str]] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip().upper()
        chain = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        groups.setdefault((resname, chain, resseq, icode), []).append(line)
    return groups


def _choose_ligand_instance(
    row: dict[str, str],
    groups: dict[tuple[str, str, str, str], list[str]],
    overrides: dict[str, tuple[str, str, str, str]],
    *,
    include_excluded: bool,
) -> tuple[str, str, str, str] | None:
    pair_id = row["pair_id"]
    if pair_id in overrides:
        return overrides[pair_id]
    if row.get("_only_overrides"):
        return None

    allowed_resnames = {
        token.strip().upper()
        for token in str(row.get("ligand_ccd", "")).replace(",", ";").split(";")
        if token.strip()
    }
    candidates = []
    for key, lines in groups.items():
        resname = key[0]
        if not include_excluded and resname in DEFAULT_EXCLUDED_LIGANDS:
            continue
        if allowed_resnames and resname not in allowed_resnames:
            continue
        candidates.append((len(lines), key))
    if not candidates and not allowed_resnames:
        for key, lines in groups.items():
            if include_excluded or key[0] not in DEFAULT_EXCLUDED_LIGANDS:
                candidates.append((len(lines), key))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _write_ligand_pdb(source_pdb: Path, ligand_key: tuple[str, str, str, str], out_pdb: Path) -> int:
    groups = _ligand_instances(source_pdb)
    lines = groups.get(ligand_key, [])
    if not lines:
        raise ValueError(f"Ligand instance {ligand_key} not found in {source_pdb}")
    serials = {serial for line in lines if (serial := _pdb_atom_serial(line)) is not None}
    conect: list[str] = []
    for line in source_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        if not line.startswith("CONECT"):
            continue
        values = [int(raw) for raw in [line[i : i + 5].strip() for i in range(6, len(line), 5)] if raw.isdigit()]
        if values and values[0] in serials:
            kept = [values[0], *[value for value in values[1:] if value in serials]]
            if len(kept) > 1:
                conect.append("CONECT" + "".join(f"{value:5d}" for value in kept) + "\n")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text("".join(lines + conect + ["END\n"]), encoding="utf-8")
    return len(lines)


def _obabel_to_sdf(input_pdb: Path, output_sdf: Path) -> tuple[bool, str]:
    obabel = shutil.which("obabel")
    if obabel is None:
        return False, "obabel executable not found"
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([obabel, str(input_pdb), "-O", str(output_sdf)], capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0 and output_sdf.exists() and output_sdf.stat().st_size > 0, log.strip()


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = {row["pair_id"]: row for row in _read_table(args.manifest)}
    sample_ids = _sample_ids(args.sample_list)
    overrides = dict(_parse_override(raw) for raw in args.ligand_override)
    base_dir = args.manifest.parent
    rows_out: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for pair_id in sample_ids:
        row = manifest_rows.get(pair_id)
        if row is None:
            records.append({"pair_id": pair_id, "status": "skipped", "reason": "pair_id not in manifest"})
            continue
        if args.only_overrides:
            row = {**row, "_only_overrides": True}
        af2_pdb = _resolve(row["af2_pdb"], base_dir)
        holo_pdb = _resolve(row["experimental_pdb"], base_dir)
        chain = str(row.get("chain") or row.get("pdb_chain") or "A").strip() or "A"

        target_dir = args.out_dir / "targets" / pair_id
        receptor_af2 = target_dir / "receptors" / "af2.pdb"
        receptor_cooper = target_dir / "receptors" / "cooper_tbdt.pdb"
        receptor_holo = target_dir / "receptors" / "true_holo_chain.pdb"

        groups = _ligand_instances(holo_pdb)
        ligand_key = _choose_ligand_instance(row, groups, overrides, include_excluded=args.include_excluded_ligands)
        if ligand_key is None:
            records.append({"pair_id": pair_id, "status": "skipped", "reason": "no selected ligand instance"})
            continue

        pair_path = args.pair_dir / f"{pair_id}.pt"
        pred_path = args.prediction_dir / f"{pair_id}.pt"
        if not pair_path.exists() or not pred_path.exists():
            records.append({"pair_id": pair_id, "status": "skipped", "reason": "missing pair or prediction file"})
            continue

        sample = torch.load(pair_path, map_location="cpu", weights_only=False)
        prediction = torch.load(pred_path, map_location="cpu", weights_only=False)
        _copy_af2_receptor(af2_pdb, receptor_af2)
        _write_cooper_receptor(af2_pdb=af2_pdb, out_pdb=receptor_cooper, sample=sample, prediction=prediction)
        _write_holo_chain_receptor(holo_pdb, chain, receptor_holo)

        ligand_pdb = target_dir / "ligand" / f"{pair_id}_{ligand_key[0]}_{ligand_key[1]}{ligand_key[2]}{ligand_key[3]}.pdb"
        ligand_sdf = ligand_pdb.with_suffix(".sdf")
        n_ligand_atoms = _write_ligand_pdb(holo_pdb, ligand_key, ligand_pdb)
        ok, obabel_log = _obabel_to_sdf(ligand_pdb, ligand_sdf)
        record = {
            "pair_id": pair_id,
            "pdb_id": row.get("pdb_id", ""),
            "ligand_resname": ligand_key[0],
            "ligand_chain": ligand_key[1],
            "ligand_resseq": ligand_key[2],
            "ligand_icode": ligand_key[3],
            "n_ligand_atoms": n_ligand_atoms,
            "ligand_pdb": str(ligand_pdb),
            "ligand_sdf": str(ligand_sdf),
            "obabel_log": obabel_log,
        }
        if not ok:
            record.update({"status": "skipped", "reason": "ligand PDB to SDF conversion failed"})
            records.append(record)
            continue

        rows_out.append(
            {
                "target_id": pair_id,
                "receptor_af2": str(receptor_af2.resolve()),
                "receptor_cooper_tbdt": str(receptor_cooper.resolve()),
                "receptor_holo": str(receptor_holo.resolve()),
                "ligand_sdf": str(ligand_sdf.resolve()),
                "reference_ligand_sdf": str(ligand_sdf.resolve()),
                "box_source_pdb": str(ligand_pdb.resolve()),
                "pdb_id": row.get("pdb_id", ""),
                "pdb_chain": chain,
                "ligand_ccd": ligand_key[0],
                "state_label": row.get("state_label", ""),
                "substrate_class": row.get("substrate_class", ""),
            }
        )
        record.update({"status": "included", "reason": ""})
        records.append(record)

    args.docking_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.docking_manifest, rows_out)
    report = {
        "manifest": str(args.docking_manifest),
        "n_samples": len(sample_ids),
        "n_included": len(rows_out),
        "n_skipped": len(sample_ids) - len(rows_out),
        "records": records,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare held-out TBDT receptors and ligand SDFs for docking evaluation.")
    p.add_argument("--manifest", type=Path, default=Path("data/tbdt_gold_training_manifest.csv"))
    p.add_argument("--sample-list", type=Path, default=Path("artifacts/tbdt_v1/test_graph_files.txt"))
    p.add_argument("--pair-dir", type=Path, default=Path("data/processed_tbdt_gold_pairs"))
    p.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/tbdt_v1/docking_inputs"))
    p.add_argument("--docking-manifest", type=Path, default=Path("data/tbdt_v1/docking_manifest.csv"))
    p.add_argument("--report-path", type=Path, default=Path("artifacts/tbdt_v1/docking_inputs/prepare_report.json"))
    p.add_argument(
        "--ligand-override",
        action="append",
        default=[],
        help="Force a ligand instance: pair_id=RES:CHAIN:RESSEQ[:ICODE]. Repeatable.",
    )
    p.add_argument("--include-excluded-ligands", action="store_true")
    p.add_argument("--only-overrides", action="store_true", help="Include only rows named by --ligand-override.")
    return p.parse_args()


def main() -> None:
    report = build_inputs(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
