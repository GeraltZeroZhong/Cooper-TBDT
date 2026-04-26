from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_table
from .pipeline import (
    DockingPipelineConfig,
    infer_default_structure_specs,
    parse_structure_specs,
    run_docking_pipeline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a batch Meeko/Vina docking pipeline and generate pose RMSD and score tables."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/docking_eval"))
    p.add_argument(
        "--structure",
        action="append",
        default=[],
        help="Structure label and manifest receptor column, e.g. af2=receptor_af2. Repeat for multiple structures.",
    )
    p.add_argument("--target-col", default="target_id")
    p.add_argument("--ligand-col", default="ligand_sdf")
    p.add_argument("--reference-ligand-col", default="reference_ligand_sdf")
    p.add_argument("--box-source-pdb-col", default="box_source_pdb")
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
    p.add_argument("--reuse", action="store_true", help="Reuse existing intermediate files when present.")
    p.add_argument("--skip-failed", action="store_true", help="Record failures and keep processing other targets.")
    p.add_argument("--dry-run", action="store_true", help="Write planned command logs without running Vina.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _parse_topn_levels(raw: str) -> list[int]:
    levels = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("--topn-levels must contain positive integers.")
    return levels


def main() -> None:
    args = parse_args()
    topn_levels = _parse_topn_levels(args.topn_levels)
    rows = read_table(args.manifest)
    structures = parse_structure_specs(args.structure) if args.structure else infer_default_structure_specs(rows)

    cfg = DockingPipelineConfig(
        manifest=args.manifest,
        output_dir=args.out_dir,
        structures=structures,
        target_col=args.target_col,
        ligand_col=args.ligand_col,
        reference_ligand_col=args.reference_ligand_col,
        box_source_pdb_col=args.box_source_pdb_col,
        rmsd_threshold=args.rmsd_threshold,
        topn_levels=topn_levels,
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
