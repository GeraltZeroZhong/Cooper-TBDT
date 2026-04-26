# Docking Evaluation Pipeline

This repo has two docking-evaluation layers:

- `python -m evopoint_da.docking_eval` summarizes already-computed pose RMSD and Vina score tables.
- `python -m evopoint_da.docking_eval.pipeline_cli` runs Meeko/Vina, computes ligand heavy-atom pose RMSD against a crystal reference ligand, and writes those tables.

## Scientific Contract

HoloShift prediction is currently a ligand-agnostic C-alpha displacement task. It is not a ligand-aware holo pocket reconstruction model and it does not repack side chains. The full-atom receptor written for docking applies the predicted C-alpha displacement as a rigid per-residue translation, then optionally runs restrained OpenMM minimization.

For prediction, `run_Predict.py` uses a `--feature_pt` graph built with the same feature schema used for training: ESMC embeddings reduced by the training PCA model, pLDDT, structural node features, KNN/PAE edge attributes, and GVP graph features. If `--feature_pt` is omitted or points to a missing file, `run_Predict.py` builds that file automatically from `--pdb_file` using `--esm_weights`, `--pca_path`, and optional `--pae_path`; this is still the strict training-compatible feature path, not a placeholder fallback. Residue IDs, sequence, feature dimension, and graph indices are validated against the exact PDB chain being predicted.

Build inference features explicitly with:

```bash
python scripts/build_prediction_features.py \
  --pdb-file prepared/raw_af2_aligned_cropped.pdb \
  --output-pt prepared/target_training_graph_features.pt \
  --esm-weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca-path data/pca_esmc_128.pkl \
  --pae-path prepared/AF-UNIPROT.json \
  --chain-id A \
  --require-pae
```

When running docking as evidence, use ligand heavy-atom pose RMSD as the primary endpoint. Vina score is reported as an auxiliary diagnostic only; a lower Vina score across different receptor conformations is not treated as pose correctness. Include a true-holo redocking arm whenever possible. If true-holo redocking cannot recover the crystal ligand pose in the requested Top-N window, pose-power claims from the benchmark should be considered failed for that run.

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
For four-way reports, `publishability.pose_power_claims_allowed` is true only when the true-holo redocking sanity gate passes.
