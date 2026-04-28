#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a local PocketMiner cryptic-pocket benchmark manifest.")
    p.add_argument(
        "--pocketminer-root",
        type=Path,
        default=Path("outputs/pocketminer_holoshift/external/pocketminer_gvp"),
        help="Clone of Mickdub/gvp on the pocket_pred branch.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift"))
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


def canonical_id(raw: str) -> str:
    return str(raw).strip().lower()


def id_with_chain(pdb_id: str, chain_id: str) -> str:
    return f"{str(pdb_id).strip().lower()}{str(chain_id).strip()}"


def find_case_insensitive(directory: Path, candidates: list[str]) -> Path | None:
    existing = {path.name.lower(): path for path in directory.glob("*")}
    for candidate in candidates:
        hit = existing.get(candidate.lower())
        if hit is not None:
            return hit.resolve()
    return None


def load_split_ids(pm_data: Path, split: str) -> np.ndarray:
    return np.load(pm_data / f"{split}_apo_ids_with_chainids.npy", allow_pickle=True)


def load_labels(pm_data: Path, split: str) -> dict[str, np.ndarray]:
    return np.load(pm_data / f"{split}_label_dictionary.npy", allow_pickle=True).item()


def first_chain_id(path: Path | None) -> str:
    if path is None:
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                return line[21].strip() or "_"
    return ""


def resolve_label_key(apo_id_chain: str, label_dict: dict[str, np.ndarray]) -> tuple[str, str]:
    keys = {str(key).lower(): str(key) for key in label_dict}
    raw = canonical_id(apo_id_chain)
    first4 = raw[:4]
    if first4 in keys:
        return keys[first4], raw[4:]
    if raw in keys:
        suffix = raw[4:] if len(raw) > 4 else ""
        return keys[raw], suffix
    raise KeyError(raw)


def load_cryptic_table(pm_data: Path) -> dict[str, dict[str, Any]]:
    table_path = pm_data / "supplementary-tables.xlsx"
    if not table_path.exists():
        return {}
    df = pd.read_excel(table_path, sheet_name="validation_and_test_sets", header=2)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        apo_pdb = row.get("PDB ID")
        apo_chain = row.get("chain ID")
        if pd.isna(apo_pdb) or pd.isna(apo_chain):
            continue
        target = canonical_id(id_with_chain(apo_pdb, apo_chain))
        rows[target] = {
            "target_id": target,
            "apo_pdb_id": str(apo_pdb).strip(),
            "apo_chain": str(apo_chain).strip(),
            "holo_pdb_id": "" if pd.isna(row.get("PDB ID.1")) else str(row.get("PDB ID.1")).strip(),
            "holo_chain": "" if pd.isna(row.get("chain ID.1")) else str(row.get("chain ID.1")).strip(),
            "ligand_code": "" if pd.isna(row.get("ligand code")) else str(row.get("ligand code")).strip(),
            "cryptic_ligand_lining_residue_count": row.get("number of resolved cryptic ligand lining residues", ""),
            "cath_code": "" if pd.isna(row.get("CATH code")) else str(row.get("CATH code")).strip(),
            "motion_type": "" if pd.isna(row.get("motion type")) else str(row.get("motion type")).strip(),
            "dominant_motion": "" if pd.isna(row.get("dominant motion")) else str(row.get("dominant motion")).strip(),
            "pocket_direction": (
                ""
                if pd.isna(row.get("pocket direction (manually assigned)"))
                else str(row.get("pocket direction (manually assigned)")).strip()
            ),
            "apo_holo_all_ca_rmsd_nm": row.get("apo-holo all-Cα RMSD", ""),
            "ligand_lining_heavy_rmsd_nm": row.get("ligand lining residue all-heavy-atom RMSD", ""),
            "structure_source": "" if pd.isna(row.get("structure source")) else str(row.get("structure source")).strip(),
            "table_set": "" if pd.isna(row.get("set")) else str(row.get("set")).strip(),
            "notes": "" if pd.isna(row.get("notes")) else str(row.get("notes")).strip(),
        }
    return rows


def build_rows(pm_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pm_data = pm_root / "data" / "pm-dataset"
    apo_dir = pm_data / "apo-structures"
    all_dir = pm_data / "all-structures"
    table = load_cryptic_table(pm_data)

    manifest_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_key = "val" if split == "validation" else "test"
        label_dict = load_labels(pm_data, split_key)
        for apo_id_chain in load_split_ids(pm_data, split_key):
            apo_id_chain = str(apo_id_chain)
            label_key, chain_id = resolve_label_key(apo_id_chain, label_dict)
            pdb_id = label_key
            target = canonical_id(apo_id_chain)
            labels = np.asarray(label_dict[label_key], dtype=int)
            apo_pdb = find_case_insensitive(apo_dir, [f"{target}_clean_h.pdb"])
            original_chain_id = chain_id
            cleaned_chain_id = first_chain_id(apo_pdb)
            if cleaned_chain_id:
                chain_id = cleaned_chain_id
            meta = table.get(target, {})
            holo_target = (
                id_with_chain(meta["holo_pdb_id"], meta["holo_chain"])
                if meta.get("holo_pdb_id") and meta.get("holo_chain")
                else ""
            )
            holo_pdb = (
                find_case_insensitive(all_dir, [f"{holo_target}_clean_h.pdb"])
                if holo_target
                else None
            )
            ligand_pdb = (
                find_case_insensitive(
                    all_dir,
                    [
                        f"{holo_target}_clean_ligand_h.pdb",
                        f"{holo_target}_clean_ligand_h_ligand.pdb",
                        f"{holo_target}_clean_ligand_h_doubleligand.pdb",
                    ],
                )
                if holo_target
                else None
            )
            row: dict[str, Any] = {
                **meta,
                "target_id": target,
                "split": split,
                "apo_pdb_id": pdb_id,
                "apo_chain": chain_id,
                "apo_chain_from_id": original_chain_id,
                "apo_pdb": str(apo_pdb or ""),
                "label_source": str(pm_data / f"{split_key}_label_dictionary.npy"),
                "n_residues": int(len(labels)),
                "n_positive": int((labels == 1).sum()),
                "n_negative": int((labels == 0).sum()),
                "n_uncertain": int((labels == 2).sum()),
                "has_cryptic_holo": int(bool(holo_pdb and ligand_pdb)),
                "holo_pdb": str(holo_pdb or ""),
                "holo_ligand_pdb": str(ligand_pdb or ""),
            }
            manifest_rows.append(row)
            for idx, label in enumerate(labels):
                label_rows.append(
                    {
                        "target_id": target,
                        "split": split,
                        "apo_pdb_id": pdb_id,
                        "apo_chain": chain_id,
                        "residue_index": idx,
                        "label": int(label),
                        "is_eval": int(label in {0, 1}),
                    }
                )
    return manifest_rows, label_rows


def main() -> None:
    args = parse_args()
    manifest_rows, label_rows = build_rows(args.pocketminer_root.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pocketminer_cryptic_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "pocketminer_residue_labels.csv", label_rows)
    status = {
        "n_targets": len(manifest_rows),
        "n_cryptic_holo_targets": sum(int(row["has_cryptic_holo"]) for row in manifest_rows),
        "n_eval_residues": sum(int(row["is_eval"]) for row in label_rows),
        "n_positive_residues": sum(1 for row in label_rows if int(row["label"]) == 1),
        "n_negative_residues": sum(1 for row in label_rows if int(row["label"]) == 0),
    }
    (args.out_dir / "pocketminer_manifest_status.json").write_text(
        __import__("json").dumps(status, indent=2), encoding="utf-8"
    )
    print(__import__("json").dumps(status, indent=2))


if __name__ == "__main__":
    main()
