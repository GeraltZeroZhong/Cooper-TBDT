from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evopoint_da.data.alignment import compute_displacement_target
from evopoint_da.data.structure import StructureParser
from evopoint_da.data.tbdt import (
    REGION_BARREL_CORE,
    build_region_features,
    family_id,
    get_region_residue_ids,
    load_region_annotation,
    state_id,
    substrate_id,
)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TBDT AF2->experimental state displacement targets.")
    p.add_argument("--manifest", default="data/tbdt_state_manifest.csv")
    p.add_argument("--out_dir", default="data/processed_tbdt_state_pairs")
    p.add_argument("--report_path", default="artifacts/build_tbdt_state_dataset_report.json")
    p.add_argument(
        "--require-core-alignment",
        action="store_true",
        help="Skip rows whose annotation has fewer than 3 barrel_core residues for Kabsch alignment.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first row failure instead of recording a skipped row.",
    )
    return p.parse_args()


def _resolve_path(raw_path: str | None, base_dir: Path | None = None) -> Path | None:
    if raw_path is None or str(raw_path).strip() == "":
        return None
    path = Path(str(raw_path).strip())
    return path if path.is_absolute() else (base_dir or Path.cwd()) / path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _filter_chain(
    chains: dict[str, dict[str, Any]],
    chain_id: str | None,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if chain_id is None or str(chain_id).strip() == "":
        return chains
    chain_id = str(chain_id).strip()
    if chain_id not in chains:
        raise ValueError(f"{label} chain {chain_id!r} not found; available chains: {sorted(chains)}")
    return {chain_id: chains[chain_id]}


def _row_value(row: dict[str, str], key: str, default: str = "", *aliases: str) -> str:
    for candidate in (key, *aliases):
        value = row.get(candidate)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _row_pair_id(row: dict[str, str], experimental_pdb: Path) -> str:
    pair_id = _row_value(row, "pair_id")
    if pair_id:
        return pair_id
    return experimental_pdb.stem


def _tensor_bool(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.bool)


def _build_output_record(
    *,
    row: dict[str, str],
    pair_id: str,
    af2_path: Path,
    experimental_path: Path,
    annotation_path: Path | None,
    annotation: dict[str, Any],
    delta_r: np.ndarray,
    residue_ids: list[str],
    af2_aligned: np.ndarray,
    af2_idx: np.ndarray,
    holo_idx: np.ndarray,
    af2_chain_id: str,
    holo_chain_id: str,
    selected_af2_data: dict[str, Any],
    core_alignment_residue_ids: set[str],
) -> dict[str, Any]:
    region_features = build_region_features(residue_ids, annotation, default_chain=af2_chain_id)
    matched_core_count = sum(1 for rid in residue_ids if rid in core_alignment_residue_ids)

    af2_pos = torch.as_tensor(af2_aligned, dtype=torch.float32)
    y_delta = torch.as_tensor(delta_r, dtype=torch.float32)
    af2_indices = torch.as_tensor(af2_idx, dtype=torch.long)
    holo_indices = torch.as_tensor(holo_idx, dtype=torch.long)

    metadata = {
        "source": "tbdt_state_manifest",
        "manifest_row": dict(row),
        "af2_pdb": str(af2_path),
        "experimental_pdb": str(experimental_path),
        "region_annotation_json": str(annotation_path) if annotation_path else None,
        "af2_chain": af2_chain_id,
        "experimental_chain": holo_chain_id,
        "core_alignment_region": REGION_BARREL_CORE,
        "core_alignment_residue_count": int(len(core_alignment_residue_ids)),
        "core_alignment_matched_count": int(matched_core_count),
        "core_alignment_used": bool(matched_core_count >= 3),
        "annotation_schema_version": annotation.get("schema_version"),
        "protein": annotation.get("protein"),
        "uniprot_id": annotation.get("uniprot_id"),
    }

    return {
        "pair_id": pair_id,
        "residue_ids": residue_ids,
        "af2_pos": af2_pos,
        "holo_pos": af2_pos + y_delta,
        "y_delta": y_delta,
        "af2_indices": af2_indices,
        "holo_indices": holo_indices,
        "plddt": torch.as_tensor(selected_af2_data["plddts"][af2_idx], dtype=torch.float32).unsqueeze(1),
        "sequence": "".join(selected_af2_data["sequence"][i] for i in af2_idx.tolist()),
        "family_id": torch.tensor(family_id(_row_value(row, "family")), dtype=torch.long),
        "state_id": torch.tensor(state_id(_row_value(row, "state", "", "state_label", "target_state")), dtype=torch.long),
        "substrate_id": torch.tensor(
            substrate_id(_row_value(row, "substrate", "", "substrate_class")),
            dtype=torch.long,
        ),
        "region_id": torch.as_tensor(region_features["region_id"], dtype=torch.long),
        "barrel_core_mask": _tensor_bool(region_features["barrel_core_mask"]),
        "plug_mask": _tensor_bool(region_features["plug_mask"]),
        "extracellular_loop_mask": _tensor_bool(region_features["extracellular_loop_mask"]),
        "tonb_box_mask": _tensor_bool(region_features["tonb_box_mask"]),
        "substrate_contact_mask": _tensor_bool(region_features["substrate_contact_mask"]),
        "eval_mask": _tensor_bool(region_features["eval_mask"]),
        "loss_weight": torch.as_tensor(region_features["loss_weight"], dtype=torch.float32),
        "metadata": metadata,
    }


def build_tbdt_state_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_path(args.manifest)
    if manifest_path is None:
        raise ValueError("--manifest is required.")
    rows = _read_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    out_dir = _resolve_path(args.out_dir)
    if out_dir is None:
        raise ValueError("--out_dir is required.")
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = StructureParser()
    built = 0
    residue_counts: list[int] = []
    rmsd_values: list[float] = []
    skip_reasons: Counter[str] = Counter()

    for row_idx, row in enumerate(rows, start=1):
        af2_path = _resolve_path(_row_value(row, "af2_pdb"), manifest_dir)
        experimental_path = _resolve_path(
            _row_value(row, "experimental_pdb", "", "holo_pdb", "target_pdb"),
            manifest_dir,
        )
        annotation_path = _resolve_path(_row_value(row, "region_annotation_json"), manifest_dir)

        try:
            if af2_path is None or experimental_path is None:
                raise ValueError("manifest row must define af2_pdb and experimental_pdb")
            pair_id = _row_pair_id(row, experimental_path)

            missing_paths = [str(path) for path in (af2_path, experimental_path, annotation_path) if path and not path.exists()]
            if missing_paths:
                raise FileNotFoundError(f"missing input path(s): {missing_paths}")

            annotation = load_region_annotation(annotation_path)
            chain = _row_value(row, "chain", "", "pdb_chain") or None
            af2_chain = _row_value(row, "af2_chain") or chain
            holo_chain = _row_value(row, "holo_chain") or _row_value(row, "experimental_chain") or chain

            af2_chains_all = parser.parse_ca_structure(str(af2_path), strict=True)
            holo_chains_all = parser.parse_ca_structure(str(experimental_path), strict=True)
            if not af2_chains_all or not holo_chains_all:
                raise ValueError("parse failed or all chains are shorter than 15 residues")

            af2_chains = _filter_chain(af2_chains_all, af2_chain, label="AF2")
            holo_chains = _filter_chain(holo_chains_all, holo_chain, label="experimental")

            core_residue_ids = get_region_residue_ids(
                annotation,
                REGION_BARREL_CORE,
                default_chain=af2_chain or next(iter(af2_chains)),
            )
            if args.require_core_alignment and len(core_residue_ids) < 3:
                raise ValueError(f"barrel_core annotation has {len(core_residue_ids)} residues; at least 3 required")
            core_alignment_selector = core_residue_ids if len(core_residue_ids) >= 3 else None

            (
                delta_r,
                residue_ids,
                af2_aligned,
                af2_idx,
                holo_idx,
                best_af2_chain_id,
                best_holo_chain_id,
            ) = compute_displacement_target(
                af2_chains,
                holo_chains,
                alignment_residue_ids=core_alignment_selector,
            )

            selected_af2_data = af2_chains[best_af2_chain_id]
            out = _build_output_record(
                row=row,
                pair_id=pair_id,
                af2_path=af2_path,
                experimental_path=experimental_path,
                annotation_path=annotation_path,
                annotation=annotation,
                delta_r=delta_r,
                residue_ids=residue_ids,
                af2_aligned=af2_aligned,
                af2_idx=af2_idx,
                holo_idx=holo_idx,
                af2_chain_id=best_af2_chain_id,
                holo_chain_id=best_holo_chain_id,
                selected_af2_data=selected_af2_data,
                core_alignment_residue_ids=core_residue_ids,
            )

            torch.save(out, out_dir / f"{pair_id}.pt")
            built += 1
            residue_counts.append(len(residue_ids))
            rmsd = float(np.sqrt(np.mean(np.sum(np.square(delta_r), axis=1))))
            rmsd_values.append(rmsd)
            print(f"[debug] Built {pair_id}: chain {best_af2_chain_id}->{best_holo_chain_id}, res={len(residue_ids)}, rmsd={rmsd:.4f} A")
        except Exception as exc:
            if args.strict:
                raise
            skip_reasons[f"{type(exc).__name__}: {exc}"] += 1
            print(f"[debug] Skip manifest row {row_idx}: {exc}")

    report = {
        "manifest": str(manifest_path),
        "rows_total": len(rows),
        "built_pairs": built,
        "skipped_pairs": len(rows) - built,
        "skip_reason_counts": dict(skip_reasons),
        "rmsd_stats": {
            "mean": float(np.mean(rmsd_values)) if rmsd_values else None,
            "median": float(np.median(rmsd_values)) if rmsd_values else None,
            "p90": float(np.percentile(rmsd_values, 90)) if rmsd_values else None,
        },
        "residue_count_stats": {
            "mean": float(np.mean(residue_counts)) if residue_counts else None,
            "median": float(np.median(residue_counts)) if residue_counts else None,
            "p90": float(np.percentile(residue_counts, 90)) if residue_counts else None,
        },
    }
    return report


def main() -> None:
    args = args_parser()
    report = build_tbdt_state_dataset(args)

    report_path = _resolve_path(args.report_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Built {report['built_pairs']} TBDT paired samples.")


if __name__ == "__main__":
    main()
