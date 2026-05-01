# Cooper-TBDT

Cooper-TBDT builds, trains, and evaluates a state-labeled C-alpha displacement
benchmark for TonB-dependent transporters (TBDTs). It starts from an
AFDB/AlphaFold-like structure, aligns it to a paired experimental structure on
the conserved beta-barrel core, and learns residue-level displacement vectors
for functional regions such as the plug, extracellular loops, substrate-contact
residues, and TonB box.

The name follows a cooper, a traditional barrel maker: the beta-barrel is kept
as the reference scaffold, while the benchmark asks how local transporter
regions should move for a target state.

## Quick Start

Create the development environment and install the package:

```bash
conda env create -f environment.yml
conda activate evopoint_da
python -m pip install -e ".[dev]"
```

Inspect available pipeline stages:

```bash
python main.py --step publication_report -- --help
python main.py --step build_pairs -- --help
python main.py --step build_features -- --help
python main.py --step eval_regions -- --help
```

A minimal Gold workflow is:

```bash
python main.py --step download_assets -- \
  --manifest data/tbdt_mixed_manifest.csv \
  --tier gold \
  --download-pae \
  --sync-tier-manifests

python main.py --step build_pairs -- \
  --manifest data/tbdt_gold_training_manifest.csv \
  --out_dir data/processed_tbdt_gold_pairs \
  --require-core-alignment

python main.py --step build_features -- \
  --pair_dir data/processed_tbdt_gold_pairs \
  --output_dir data/processed_tbdt_gold_graphs \
  --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128.pkl \
  --fit_pca

python train.py \
  data=tbdt_state \
  model=gvp_tbdt_module \
  data.data_dir=data/processed_tbdt_gold_graphs \
  study_name=tbdt_gold_scaffold_prior \
  trainer.max_epochs=40
```

## Scientific Scope

Cooper-TBDT is not a de novo structure-prediction benchmark and does not use
full-chain RMSD as its main success criterion. The raw AFDB structure is the
zero-displacement baseline. A method improves the task only when its predicted
local displacement field lowers region-level error relative to leaving the AFDB
coordinates unchanged.

For each matched residue `i`, the target is:

```text
delta_i = x_i(exp) - x_i(AFDB)
x_i(corrected) = x_i(AFDB) + delta_hat_i
```

where the AFDB and experimental structures have already been placed in a shared
barrel-core frame. The main reporting units are functional regions: barrel core,
plug, extracellular loops, substrate-contact residues, TonB box, and their
evaluation masks.

## Installation

The repository is developed against Python 3.10, PyTorch, PyTorch Geometric,
Hydra, PyTorch Lightning, Biopython, FreeSASA, RDKit, Meeko, and Vina. The
recommended setup is the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate evopoint_da
python -m pip install -e ".[dev]"
```

For CPU-only development, install a CPU-compatible PyTorch/PyG stack first, then
install the package:

```bash
python -m pip install -e ".[dev]"
```

The Conda environment includes the docking stack used by this repository. In a
pip-managed environment with compatible wheels, docking extras can be installed
with:

```bash
python -m pip install -e ".[dev,docking]"
```

## Command Line

`main.py` provides stable aliases for reproducible pipeline stages:

```bash
python main.py --step <stage> -- <stage-specific arguments>
```

Common stages:

- `download_assets`: download structures and PAE files referenced by manifests.
- `build_pairs`: build AFDB-to-experimental displacement target `.pt` files.
- `build_features`: build PyG graph files with sequence, structure, and PAE features.
- `train`: run the Hydra training entry point.
- `predict_graphs`: export displacement predictions from a checkpoint.
- `eval_regions`: compute raw or predicted region-level displacement metrics.
- `blend_predictions`: compose prediction sources by structural region.
- `coordinate_baselines`, `template_baselines`, `external_baselines`: run coordinate controls.
- `eval_classification`: evaluate residue-shift localization curves.
- `publication_report`: assemble publication reporting tables from artifacts.
- `figure_all`: build configured figure panels.
- `prepare_docking_manifest` and `docking_eval`: run the secondary docking endpoint.

Use `--help` after the forwarded `--` to inspect a stage:

```bash
python main.py --step predict_graphs -- --help
```

## Data

Tracked data files define the benchmark:

- `data/tbdt_mixed_manifest.csv`: combined Gold/Silver/Bronze manifest.
- `data/tbdt_gold_manifest.csv` and `data/tbdt_gold_training_manifest.csv`: Gold TBDT records used for supervised displacement experiments.
- `data/tbdt_silver_manifest.csv`: auxiliary beta-barrel structural records.
- `data/tbdt_bronze_manifest.csv`: AFDB-only TBDT homolog records.
- `data/tbdt_region_annotations/`: JSON masks for barrel core, plug, TonB box, and related regions.

Large local assets are intentionally ignored by Git:

```text
data/raw_af2/
data/raw_pdb/
data/processed_tbdt_*_pairs/
data/processed_tbdt_*_graphs/
checkpoints/
logs/
val_metrics/
outputs/
artifacts/
```

Publication graph builds should use real AFDB-v6 structures, PAE files, and
ESMC/PCA features. `--smoke-test-features` and `--allow-missing-pae` are for
debugging only.

## Training

The training entry point is `train.py`, configured through Hydra:

```bash
python train.py \
  data=tbdt_state \
  model=gvp_tbdt_module \
  data.data_dir=data/processed_tbdt_gold_graphs \
  study_name=tbdt_gold_scaffold_prior \
  trainer.max_epochs=40
```

The default model is a scaffold-prior GVP network. It uses residue graph
geometry, ESMC/PCA sequence features, AFDB confidence features, PAE-aware edges,
region/family/state/substrate conditioning, region-weighted displacement loss,
and a high-confidence barrel-core anchor.

CPU smoke test:

```bash
python train.py \
  data=tbdt_state \
  model=gvp_tbdt_module \
  data.data_dir=data/processed_tbdt_gold_graphs_smoke \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  model.hidden_dim=16 \
  model.num_layers=1 \
  +run_post_train_tests=false
```

Training writes checkpoints, logger output, and validation metrics under:

```text
checkpoints/<study_name>/<timestamp>/
logs/
val_metrics/<study_name>/<timestamp>/
```

## Prediction And Evaluation

Export predictions:

```bash
python main.py --step predict_graphs -- \
  --ckpt checkpoints/tbdt_gold_scaffold_prior/<timestamp>/best-selection-*.ckpt \
  --data-dir data/processed_tbdt_gold_graphs \
  --split test \
  --split-source metadata \
  --output-dir artifacts/tbdt_v1/predictions/scaffold_prior_test
```

Evaluate raw AFDB as the zero-displacement baseline:

```bash
python main.py --step eval_regions -- \
  data/processed_tbdt_gold_graphs \
  --output-json artifacts/tbdt_v1/gold_test_zero_region_metrics.json \
  --output-csv artifacts/tbdt_v1/gold_test_zero_region_metrics.csv
```

Evaluate model predictions:

```bash
python main.py --step eval_regions -- \
  data/processed_tbdt_gold_graphs \
  --predictions artifacts/tbdt_v1/predictions/scaffold_prior_test \
  --output-json artifacts/tbdt_v1/gold_test_scaffold_prior_region_metrics.json \
  --output-csv artifacts/tbdt_v1/gold_test_scaffold_prior_region_metrics.csv \
  --paired-delta-csv artifacts/tbdt_v1/gold_test_scaffold_prior_paired_delta.csv \
  --tonb-metrics-csv artifacts/tbdt_v1/gold_test_scaffold_prior_tonb_metrics.csv
```

Core evaluation fields include `target_displacement_rms`,
`prediction_error_rms`, `mse_improvement_vs_zero_fraction`,
`sample_improvement_rate`, `direction_cosine_mean`, and `magnitude_mae`. TonB
results should be interpreted with centroid, direction, and exposure-state
diagnostics, not coordinate RMSD alone.

## Outputs

Main generated outputs are:

- Pair files: `data/processed_tbdt_*_pairs/*.pt`
- Graph files: `data/processed_tbdt_*_graphs/*.pt`
- Checkpoints: `checkpoints/<study_name>/<timestamp>/*.ckpt`
- Validation metrics: `val_metrics/<study_name>/<timestamp>/*.csv`
- Region metrics: `artifacts/tbdt_v1/**/*.json` and `*.csv`
- Figure panels and publication tables: `artifacts/tbdt_v1/`

The publication report collector does not rerun training. It gathers manifests,
metrics, baselines, file hashes, and quality-control summaries from existing
artifacts:

```bash
python main.py --step publication_report -- \
  --out-dir artifacts/tbdt_v1/publication_report \
  --mixed-manifest data/tbdt_mixed_manifest.csv \
  --gold-manifest data/tbdt_gold_training_manifest.csv \
  --gold-pair-dir data/processed_tbdt_gold_pairs \
  --gold-graph-dir data/processed_tbdt_gold_graphs
```

## Docking Endpoint

Docking is secondary evidence, not the primary Cooper-TBDT endpoint. Pose
correctness is measured by ligand heavy-atom RMSD against the reference ligand;
Vina score is only an auxiliary diagnostic. Any pose-rescue claim should be
gated by true-holo redocking.

```bash
python main.py --step docking_eval -- \
  --manifest data/tbdt_v1/docking_manifest.csv \
  --out-dir outputs/tbdt_v1/docking_eval \
  --structure af2=receptor_af2 \
  --structure cooper_tbdt=receptor_cooper_tbdt \
  --structure true_holo=receptor_holo \
  --rmsd-threshold 2.0 \
  --exhaustiveness 8 \
  --num-modes 9
```

## Repository Layout

```text
.
|-- configs/                  Hydra configs
|-- data/                     Versioned manifests and region annotations
|-- src/evopoint_da/data/     Structure parsing, alignment, graph, and dataset code
|-- src/evopoint_da/models/   GVP backbone and Lightning module
|-- src/evopoint_da/pipeline/ Dataset, baseline, evaluation, and report builders
|-- src/evopoint_da/figures/  Publication figure builders
|-- src/evopoint_da/docking_eval/
|                             Secondary docking evaluation code
|-- tests/                    Unit and smoke tests
|-- main.py                   Pipeline alias entry point
|-- train.py                  Hydra/PyTorch Lightning training entry point
```

## Development

Run tests with:

```bash
pytest
```

The test suite covers graph alignment, dataset processing, model helper
behavior, TBDT pipeline utilities, docking metrics, and region/structure
handling.

Reproducibility conventions:

- Use metadata-defined splits grouped by UniProt accession.
- Do not use Gold test metrics for checkpoint selection, blend calibration, or baseline tuning.
- Treat missing PAE and smoke-test features as non-publication settings.
- Report region-resolved metrics as the main endpoint; all-residue summaries are diagnostic.
- Interpret TonB-box results together with direction and exposure-state diagnostics.

## License

Cooper-TBDT is released under the MIT License. See `LICENSE`.
