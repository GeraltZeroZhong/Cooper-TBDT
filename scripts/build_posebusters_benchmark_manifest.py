#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-target docking/diagnostic manifests from local PoseBusters data.")
    p.add_argument(
        "--posebusters-root",
        type=Path,
        default=Path("data/docking_benchmarks/posebusters/posebusters_benchmark_set"),
    )
    p.add_argument("--pdb-uniprot-map", type=Path, default=Path("data/metadata/pdb_uniprot_mapping.json"))
    p.add_argument("--holoshift-root", type=Path, default=Path("outputs/holoshift_posebusters"))
    p.add_argument(
        "--predicted-dir-name",
        default="posebusters_benchmark_holo_aligned_predicted_structures",
        help="Directory containing holo-aligned predicted proteins used as apo/predicted baseline.",
    )
    p.add_argument(
        "--esmfold-dir-name",
        default="posebusters_benchmark_holo_aligned_esmfold_predicted_structures",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/negative_diagnostic_benchmark/posebusters"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--min-targets", type=int, default=50)
    return p.parse_args()


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
        writer.writerows(rows)


def split_for_family(family_id: str) -> str:
    value = int(hashlib.sha1(family_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.7:
        return "train"
    if value < 0.8:
        return "val"
    if value < 0.9:
        return "calib"
    return "test"


def first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = args.posebusters_root.resolve()
    predicted_dir = root / args.predicted_dir_name
    esmfold_dir = root / args.esmfold_dir_name
    mapping = json.loads(args.pdb_uniprot_map.read_text(encoding="utf-8")) if args.pdb_uniprot_map.exists() else {}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    target_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and "_" in d.name and not d.name.startswith("posebusters_")
    )
    for target_dir in target_dirs:
        target_id = target_dir.name
        pdb_id = target_id.split("_", 1)[0].upper()
        ligand_code = target_id.split("_", 1)[1] if "_" in target_id else ""
        true_holo = first_existing(
            [
                target_dir / f"{target_id}_protein_bs_cropped.pdb",
                target_dir / f"{target_id}_protein.pdb",
            ]
        )
        ligand_sdf = first_existing(
            [
                target_dir / f"{target_id}_ligand_start_conf.sdf",
                target_dir / f"{target_id}_ligand.sdf",
            ]
        )
        reference_ligand_sdf = first_existing([target_dir / f"{target_id}_ligand.sdf"])
        if true_holo is None or ligand_sdf is None or reference_ligand_sdf is None:
            skipped.append({"target_id": target_id, "reason": "missing true_holo or ligand/reference ligand"})
            continue

        predicted = first_existing(
            [
                predicted_dir / f"{target_id}_holo_aligned_predicted_protein.pdb",
                predicted_dir / f"{target_id}.pdb",
                predicted_dir / f"{target_id.lower()}_model.cif",
            ]
        )
        holo_aligned_pred = first_existing([predicted_dir / f"{target_id}_holo_aligned_predicted_protein.pdb"])
        esmfold = first_existing([esmfold_dir / f"{target_id}_holo_aligned_predicted_protein.pdb"])

        holoshift_unrelaxed = first_existing(
            [
                args.holoshift_root / target_id / f"{target_id}_holoshift_predicted_unrelaxed.pdb",
                args.holoshift_root / target_id / f"{target_id}_holoshift_predict_openmm_relax_predicted_unrelaxed.pdb",
            ]
        )
        holoshift_relax = first_existing(
            [
                args.holoshift_root / target_id / f"{target_id}_holoshift_openmm_relax.pdb",
                args.holoshift_root / target_id / f"{target_id}_holoshift_predict_openmm_relax.pdb",
            ]
        )

        uniprot = mapping.get(pdb_id, "")
        family_id = uniprot or pdb_id
        row: dict[str, Any] = {
            "target_id": target_id,
            "pdb_id": pdb_id,
            "ligand_code": ligand_code,
            "protein_family": family_id,
            "split": split_for_family(family_id),
            "receptor_true_holo": str(true_holo),
            "ligand_sdf": str(ligand_sdf),
            "reference_ligand_sdf": str(reference_ligand_sdf),
            "receptor_raw_af2": str(predicted or ""),
            "receptor_predicted": str(predicted or ""),
            "receptor_holo_aligned_predicted": str(holo_aligned_pred or predicted or ""),
            "receptor_esmfold": str(esmfold or ""),
            "receptor_holoshift_unrelaxed": str(holoshift_unrelaxed or ""),
            "receptor_holoshift": str(holoshift_relax or ""),
            "has_predicted_baseline": int(predicted is not None),
            "has_esmfold_baseline": int(esmfold is not None),
            "has_holoshift": int(holoshift_unrelaxed is not None and holoshift_relax is not None),
            "apo_chain": "A",
            "holo_chain": "A",
            "uniprot_id": uniprot,
            "source": "PoseBusters",
        }
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    return rows, skipped


def main() -> None:
    args = parse_args()
    rows, skipped = build_rows(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "posebusters_multitarget_manifest.csv", rows)
    write_csv(args.out_dir / "posebusters_redocking_manifest.csv", [
        {
            "target_id": row["target_id"],
            "receptor_true_holo": row["receptor_true_holo"],
            "ligand_sdf": row["ligand_sdf"],
            "reference_ligand_sdf": row["reference_ligand_sdf"],
            "protein_family": row["protein_family"],
            "split": row["split"],
        }
        for row in rows
    ])
    write_csv(args.out_dir / "posebusters_predicted_manifest.csv", [
        row for row in rows if row.get("has_predicted_baseline") == 1
    ])
    write_csv(args.out_dir / "posebusters_holoshift_available_manifest.csv", [
        row for row in rows if row.get("has_holoshift") == 1
    ])
    write_csv(args.out_dir / "posebusters_manifest_skipped.csv", skipped)

    if len(rows) < args.min_targets:
        raise SystemExit(
            f"Only {len(rows)} targets were available, below --min-targets={args.min_targets}. "
            f"See {args.out_dir / 'posebusters_manifest_skipped.csv'}."
        )
    print(f"Wrote {len(rows)} PoseBusters targets to {args.out_dir / 'posebusters_multitarget_manifest.csv'}")
    print(f"HoloShift-complete targets: {sum(int(row['has_holoshift']) for row in rows)}")


if __name__ == "__main__":
    main()
