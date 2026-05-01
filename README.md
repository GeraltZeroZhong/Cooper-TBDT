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

### Download Benchmark Assets Only

If you only want the Cooper-TBDT benchmark assets, you do not need the full
PyTorch/PyG training environment. A minimal Python environment with `requests`
is enough:

```bash
python -m venv .venv-download
source .venv-download/bin/activate
python -m pip install requests

PYTHONPATH=src python main.py --workflow download_benchmark
```

This reads `data/tbdt_mixed_manifest.csv`, downloads experimental structures to
`data/raw_pdb/`, downloads AFDB-v6 structures and PAE files to `data/raw_af2/`,
syncs the Gold/Silver/Bronze manifests, and writes a download report under
`artifacts/tbdt_v1/`.

To download only the supervised Gold subset:

```bash
PYTHONPATH=src python main.py --workflow download_benchmark -- --tier gold
```

### Run The Provided Baseline Checkpoint

Baseline prediction requires the full Cooper-TBDT environment because it loads
processed graph files with PyTorch/PyG and runs the GVP model. After installing
the environment below, run:

```bash
python main.py --workflow baseline_predict -- \
  --data-dir data/processed_tbdt_gold_graphs \
  --output-dir artifacts/tbdt_v1/predictions/scaffold_prior_test
```

The prediction workflow expects processed graph files in
`data/processed_tbdt_gold_graphs`. If you only have the raw downloaded assets,
use Workflow 3 or the `build_pairs` and `build_features` stages to construct
those graphs first.

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

### ESM-C Weights

The Conda environment installs the `esm` Python package, but Cooper-TBDT expects
the ESM-C checkpoint itself as a local file when building publication-style graph
features. Place the 600M ESM-C weights at the default path or pass your own path
with `--esm-weights`:

```text
esmc_weights/esmc_600m_2024_12_v0.pth
```

One lightweight way to fetch the weights is the Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p esmc_weights
huggingface-cli download EvolutionaryScale/esmc-600m-2024-12 \
  data/weights/esmc_600m_2024_12_v0.pth \
  --local-dir esmc_weights/esmc-600m-2024-12
cp esmc_weights/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth \
  esmc_weights/esmc_600m_2024_12_v0.pth
```

You can also use Git LFS to clone
`https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12` and then copy
`data/weights/esmc_600m_2024_12_v0.pth` to the path above. Review the upstream
model license before downloading or redistributing ESM-C assets.

Use the same PCA file with the matching ESM-C features:

```bash
python main.py --step build_features -- \
  --pair_dir data/processed_tbdt_gold_pairs \
  --output_dir data/processed_tbdt_gold_graphs \
  --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128.pkl \
  --fit_pca
```

For software checks without ESM-C weights, use `--smoke-test-features`; do not
use smoke-test features for publication metrics.

## Three Workflows

`main.py` exposes both low-level pipeline stages and three higher-level
workflows:

```bash
python main.py --workflow download_benchmark -- --help
python main.py --workflow baseline_predict -- --help
python main.py --workflow reproduce_training -- --help
```

Use `--dry-run` before `--workflow` to print the exact commands without running
them:

```bash
python main.py --dry-run --workflow reproduce_training
```

### 1. Download The Cooper-TBDT Benchmark Dataset

This workflow downloads the structures and AFDB/PAE assets referenced by the
versioned manifests. By default it downloads Gold, Silver, and Bronze records,
writes into `data/raw_pdb/` and `data/raw_af2/`, and synchronizes the tier
manifests.

```bash
python main.py --workflow download_benchmark
```

To download only the supervised Gold benchmark assets:

```bash
python main.py --workflow download_benchmark -- --tier gold
```

Main outputs:

- `data/raw_pdb/`: experimental RCSB structures
- `data/raw_af2/`: AFDB structures and PAE files
- `data/tbdt_gold_manifest.csv`, `data/tbdt_silver_manifest.csv`,
  `data/tbdt_bronze_manifest.csv`: synchronized tier manifests
- `artifacts/tbdt_v1/download_tbdt_manifest_assets_report.json`: download report

Useful options:

```bash
python main.py --workflow download_benchmark -- \
  --tier gold \
  --workers 16 \
  --raw-pdb-dir data/raw_pdb \
  --raw-af2-dir data/raw_af2
```

### 2. Run Baseline Prediction From The Provided Checkpoint

Use this workflow when the benchmark graphs already exist and you want baseline
predictions from the Cooper-TBDT checkpoint we provide. It does not train or
evaluate; it downloads the published checkpoint if needed and exports `*.pt`
prediction files.

```bash
python main.py --workflow baseline_predict -- \
  --data-dir data/processed_tbdt_gold_graphs \
  --split test \
  --split-source metadata \
  --output-dir artifacts/tbdt_v1/predictions/scaffold_prior_test
```

The default checkpoint is the seed-404 `best-selection` scaffold-prior baseline.
The workflow stores it at:

```text
checkpoints/cooper_tbdt_baseline/best-selection-seed404.ckpt
```

If the file is missing, the workflow downloads:

```text
https://github.com/GeraltZeroZhong/Cooper-TBDT/releases/download/v0.1.0/cooper_tbdt_baseline_seed404_best-selection.ckpt
```

The expected SHA256 is:

```text
cf7515a8c1634b7a365696d807d03a37ea6fdd483260b4e8b52aa7c7c6daf891
```

Main output:

- `artifacts/tbdt_v1/predictions/scaffold_prior_test/*.pt`
- `artifacts/tbdt_v1/predictions/scaffold_prior_test_report.json`

If the interpretation is binding-site or pocket-oriented, also consider adding
[ProtCross](https://github.com/GeraltZeroZhong/ProtCross) as an external
baseline. Cooper-TBDT predicts residue-level C-alpha displacement vectors,
whereas ProtCross predicts residue-level binding-site probabilities on PDB/AF2
structures. It is therefore a score-only binding-site comparator, not a
replacement for the Cooper-TBDT coordinate endpoint.

```bash
python main.py --step external_baselines -- \
  --data-dir data/processed_tbdt_gold_graphs \
  --split test \
  --split-source metadata \
  --baseline protcross_pocket_score \
  --output-root artifacts/tbdt_v1/external_score_baselines \
  --classification-out-dir artifacts/tbdt_v1/external_baseline_curves
```

To use a locally supplied copy or a mirror instead of the default release asset:

```bash
python main.py --workflow baseline_predict -- \
  --ckpt checkpoints/cooper_tbdt_baseline/best-selection-seed404.ckpt \
  --checkpoint-url "$COOPER_TBDT_BASELINE_CHECKPOINT_URL"
```

### 3. Reproduce Cooper-TBDT Training And Results

This workflow runs the Gold supervised path end to end:

1. download Gold structures and AFDB/PAE assets;
2. build AFDB-to-experimental displacement pair files;
3. build PyG graph files with structure, PAE, SASA, and ESM-C/PCA features;
4. train the scaffold-prior GVP model;
5. export held-out predictions from the selected checkpoint;
6. evaluate raw AFDB and model predictions with region-level metrics.

Publication-style reproduction requires real AFDB-v6 structures, PAE files, and
ESM-C weights:

```bash
python main.py --workflow reproduce_training -- \
  --esm-weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca-path data/pca_esmc_128.pkl \
  --max-epochs 40
```

Main outputs:

- `data/processed_tbdt_gold_pairs/*.pt`
- `data/processed_tbdt_gold_graphs/*.pt`
- `checkpoints/tbdt_gold_scaffold_prior/<timestamp>/*.ckpt`
- `val_metrics/tbdt_gold_scaffold_prior/<timestamp>/*.csv`
- `artifacts/tbdt_v1/predictions/scaffold_prior_test/*.pt`
- `artifacts/tbdt_v1/gold_test_zero_region_metrics.{json,csv}`
- `artifacts/tbdt_v1/gold_test_scaffold_prior_region_metrics.{json,csv}`
- `artifacts/tbdt_v1/gold_test_scaffold_prior_paired_delta.csv`
- `artifacts/tbdt_v1/gold_test_scaffold_prior_tonb_metrics.csv`

For a CPU smoke test, use deterministic lightweight features and a small model.
This is only a software check and is not suitable for publication reporting:

```bash
python main.py --workflow reproduce_training -- \
  --smoke-test-features \
  --allow-missing-pae \
  --accelerator cpu \
  --devices 1 \
  --max-epochs 1 \
  --skip-post-train-tests \
  --train-override model.hidden_dim=16 \
  --train-override model.num_layers=1
```

To reuse already built data or an already trained checkpoint:

```bash
python main.py --workflow reproduce_training -- \
  --skip-download \
  --skip-build-pairs \
  --skip-build-features \
  --skip-train \
  --ckpt checkpoints/tbdt_gold_scaffold_prior/<run>/best-selection-*.ckpt
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

## Command Line

The workflow interface is intended for common use. The original single-stage
interface is still available for debugging and publication artifact assembly:

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

Publication graph builds should use real AFDB-v6 structures, PAE files, and
ESM-C/PCA features. `--smoke-test-features` and `--allow-missing-pae` are for
debugging only.

## Evaluation Notes

Core evaluation fields include `target_displacement_rms`,
`prediction_error_rms`, `mse_improvement_vs_zero_fraction`,
`sample_improvement_rate`, `direction_cosine_mean`, and `magnitude_mae`. TonB
results should be interpreted with centroid, direction, and exposure-state
diagnostics, not coordinate RMSD alone.

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
|-- main.py                   Pipeline and workflow entry point
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

## Release Notes

### v0.1.0 Initial Release

Initial Cooper-TBDT release with versioned benchmark manifests, benchmark asset
download workflow, provided-checkpoint baseline prediction, Gold training
reproduction workflow, region-resolved evaluation utilities, and publication
figure/report builders.

## License

Cooper-TBDT is released under the MIT License. See `LICENSE`.
