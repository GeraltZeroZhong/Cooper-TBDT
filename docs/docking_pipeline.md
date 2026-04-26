# Docking Evaluation Pipeline

This repo has two docking-evaluation layers:

- `python -m evopoint_da.docking_eval` summarizes already-computed pose RMSD and Vina score tables.
- `python -m evopoint_da.docking_eval.pipeline_cli` runs Meeko/Vina, computes ligand heavy-atom pose RMSD against a crystal reference ligand, and writes those tables.

## Dependencies

The batch runner expects these executables on `PATH`:

- `mk_prepare_receptor.py`
- `mk_prepare_ligand.py`
- `mk_export.py`
- `vina`

It also imports RDKit for ligand SDF preparation and heavy-atom RMSD.

## Manifest

Use CSV or TSV. Relative paths are resolved relative to the manifest file.

Required columns for the default AF2 vs HoloShift comparison:

```csv
target_id,receptor_af2,receptor_holoshift,ligand_sdf,reference_ligand_sdf
1ABC,structures/1ABC_af2.pdb,structures/1ABC_holoshift.pdb,ligands/1ABC_ligand.sdf,ligands/1ABC_crystal_ligand.sdf
```

The runner can infer the docking box from `reference_ligand_sdf`. Alternatively,
provide explicit columns:

```text
center_x,center_y,center_z,size_x,size_y,size_z
```

or a PDB with a bound ligand:

```text
box_source_pdb
```

Custom receptor columns can be supplied with repeated `--structure LABEL=COLUMN`.

## Run

```bash
python -m evopoint_da.docking_eval.pipeline_cli \
  --manifest data/docking_manifest.csv \
  --out-dir outputs/docking_eval/run_001 \
  --structure af2=receptor_af2 \
  --structure holoshift=receptor_holoshift \
  --rmsd-threshold 2.0 \
  --exhaustiveness 8 \
  --num-modes 9
```

Useful development flags:

```bash
python -m evopoint_da.docking_eval.pipeline_cli \
  --manifest data/docking_manifest.csv \
  --out-dir outputs/docking_eval/dry_run \
  --dry-run \
  --limit 2
```

## Outputs

The pipeline writes:

- `poses_all.csv`: one row per target, structure, and Vina pose.
- `poses_af2.csv`, `poses_holoshift.csv`: filtered pose tables compatible with the existing summarizer.
- `scores.csv`: Top-1 Vina scores and `delta_score = score_holoshift - score_af2`.
- `summary.json`: success rates, Top-N hit rates, first-hit rank, and delta score statistics.
- `summary.md`: compact human-readable report.
- `targets/<target_id>/...`: receptor/ligand PDBQT files, Vina logs, docking box files, exported pose SDFs.

Success is computed as Top-1 ligand heavy-atom RMSD `< 2.0 A` by default.

