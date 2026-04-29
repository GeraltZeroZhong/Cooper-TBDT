# Cooper-TBDT v1

## Project Definition

Cooper-TBDT v1 is a state-conditioned displacement benchmark for TonB-dependent transporters. The model target is local C-alpha displacement from an AF2-like starting state to an experimentally observed target state, conditioned by family, target state, substrate class, and residue region. Reporting is region based: barrel core, plug, TonB box, substrate-contact residues, extracellular loops, and any seed-specific regions defined in the manifest or processed sample. Full-chain RMSD is not the primary endpoint for this project.

The v1 baseline is raw AF2, represented as zero predicted displacement. A learned model improves the baseline only when region-level prediction error RMS is lower than the region target-displacement RMS.

## FecA Seed

FecA is the seed system for TBDT v1 because it is a canonical TBDT with a beta-barrel, plug domain, extracellular loops, and ligand-coupled conformational state changes. Seed records should preserve the AF2 chain, matched experimental chain, residue IDs, and barrel-core annotation so the evaluation can compare equivalent C-alpha positions by region.

Recommended seed naming:

```text
target_id: FecA
family: TBDT
seed: true
state_source: paired_af2_holo
```

## Data Sources

TBDT v1 uses paired structure records:

- AF2 or AF2-like predicted structures for the starting conformation.
- Experimental PDB structures for target-state coordinates.
- PAE JSON files for final fixed-PAE graph edges. Missing PAE may be allowed only in explicitly labeled smoke or limitation analyses.
- Region annotations for barrel core and TBDT functional regions.
- Optional docking manifests for secondary ligand-pose evaluation.

Processed samples are `.pt` files with displacement targets. The evaluation script expects at least one target field such as `y_delta`; it also accepts `af2_pos` plus `holo_pos` and computes `holo_pos - af2_pos`.

## Manifest Fields

Use CSV or TSV. Relative paths should be resolved relative to the manifest location by the builder that consumes it.

Required fields:

```text
target_id
af2_pdb
experimental_pdb
chain
region_annotation_json
family
state_label
substrate_class
split
```

Recommended fields:

```text
uniprot_id
pae_json
sequence
residue_id_schema
barrel_core_residue_ids
plug_residue_ids
tonb_box_residue_ids
substrate_contact_residue_ids
extracellular_loop_residue_ids
pdb_id
pdb_chain
ligand_ccd
resolution
method
notes
```

Docking secondary endpoint fields, when available:

```text
ligand_sdf
reference_ligand_sdf
receptor_af2
receptor_cooper_tbdt
receptor_holo
center_x,center_y,center_z
size_x,size_y,size_z
```

## Build Command

Download seed structures first. AFDB inputs should use v6 naming.

```bash
python -m evopoint_da.pipeline.fetch_tbdt_structures download-manifest \
  --manifest data/tbdt_state_manifest.csv \
  --download-pae \
  --report-path artifacts/tbdt_v1/download_manifest_report.json
```

Build paired displacement targets with the TBDT state builder. It aligns AF2 and experimental coordinates by annotated barrel-core residues, then writes region masks and condition IDs into each pair file.

```bash
python -m evopoint_da.pipeline.build_tbdt_state_dataset \
  --manifest data/tbdt_state_manifest.csv \
  --out_dir data/processed_tbdt_state_pairs \
  --report_path artifacts/tbdt_v1/build_pairs_report.json

python -m evopoint_da.pipeline.build_features_with_sasa \
  --pair_dir data/processed_tbdt_state_pairs \
  --output_dir data/processed_tbdt_state_graphs \
  --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128.pkl
```

The feature builder preserves TBDT fields from the pair files, so training graph samples retain `family_id`, `state_id`, `substrate_id`, `region_id`, masks, and `loss_weight`.

For smoke tests only, avoid the large ESMC dependency with deterministic 144-dim stand-in features:

```bash
python -m evopoint_da.pipeline.build_features_with_sasa \
  --pair_dir data/processed_tbdt_state_pairs \
  --output_dir data/processed_tbdt_state_graphs \
  --af2_structure_dir data/raw_af2 \
  --pae_dir data/raw_af2 \
  --allow-missing-pae \
  --smoke-test-features \
  --report_path artifacts/tbdt_v1/build_smoke_graphs_report.json
```

## Training Command

Use the Hydra data config for TBDT state data:

```bash
python train.py data=tbdt_state \
  model=gvp_tbdt_module \
  trainer.max_epochs=100
```

Smoke-test training command:

```bash
python train.py data=tbdt_state model=gvp_tbdt_module \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  data.batch_size=1 \
  model.hidden_dim=16 \
  model.num_layers=1
```

## Prediction Smoke Test

The old `run_Predict.py` interface has been removed from the public-code boundary. Use the current pipeline aliases for graph prediction and downstream evaluation:

```bash
python main.py --step predict_graphs -- --help
python main.py --step blend_predictions -- --help
python main.py --step eval_regions -- --help
```

Historical smoke-test prediction flows used AFDB v6 FecA, state/family/substrate conditioning, and optional smoke-test features. New production prediction should use the ESMC/PCA feature path and the current TBDT graph prediction stage rather than the removed `run_Predict.py` script.

## Evaluation Command

Evaluate raw AF2 / zero baseline:

```bash
python -m evopoint_da.pipeline.eval_tbdt_state \
  $(cat artifacts/tbdt_v1/test_graph_files.txt) \
  --output-json artifacts/tbdt_v1/gold_real_test_zero_region_metrics.json \
  --output-csv artifacts/tbdt_v1/gold_real_test_zero_region_metrics.csv
```

Evaluate model predictions:

```bash
python -m evopoint_da.pipeline.eval_tbdt_state \
  $(cat artifacts/tbdt_v1/test_graph_files.txt) \
  --predictions artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test \
  --output-json artifacts/tbdt_v1/report_models/metrics/validation_calibrated_region_blend_test.json \
  --output-csv artifacts/tbdt_v1/report_models/metrics/validation_calibrated_region_blend_test.csv
```

Required metrics include `target_displacement_rms`, `prediction_error_rms`, `mse_improvement_vs_zero_fraction`, `better_than_zero_rate`, `sample_improvement_rate`, `sample_improvement_median`, `direction_cosine_mean`, `magnitude_mae`, and barrel-core predicted displacement magnitude. Interpret positive `improvement_vs_zero` as lower error than raw AF2 in that region. Use `sample_improvement_rate` and `sample_improvement_median` to avoid a few large structures dominating the residue-level aggregate.

## Docking Secondary Endpoint

Docking is secondary evidence. Use ligand heavy-atom pose RMSD against the crystal ligand as the docking endpoint. Vina score can be reported as an auxiliary diagnostic, but it is not pose correctness and should not override ligand pose RMSD.

Use the existing docking runner on a manifest with AF2, Cooper-TBDT, and preferably true-holo receptors:

```bash
python -m evopoint_da.docking_eval.pipeline_cli \
  --manifest data/tbdt_v1/docking_manifest.csv \
  --out-dir outputs/tbdt_v1/docking_eval \
  --structure af2=receptor_af2 \
  --structure cooper_tbdt=receptor_cooper_tbdt \
  --structure true_holo=receptor_holo \
  --rmsd-threshold 2.0 \
  --exhaustiveness 8 \
  --num-modes 9
```

For the current Gold test split, build docking inputs from the held-out graph list and the validation-calibrated Cooper-TBDT coordinate predictions:

```bash
python main.py --step prepare_docking_manifest -- \
  --only-overrides \
  --ligand-override tbdt_c5i2d9_3qlb_a=EFE:A:701 \
  --out-dir artifacts/tbdt_v1/docking_inputs_efe_only \
  --docking-manifest data/tbdt_v1/docking_manifest_efe_only.csv \
  --report-path artifacts/tbdt_v1/docking_inputs_efe_only/prepare_report.json

python main.py --step docking_eval -- \
  --manifest data/tbdt_v1/docking_manifest_efe_only.csv \
  --out-dir outputs/tbdt_v1/docking_eval_efe_only \
  --structure af2=receptor_af2 \
  --structure cooper_tbdt=receptor_cooper_tbdt \
  --structure true_holo=receptor_holo \
  --rmsd-threshold 2.0 \
  --exhaustiveness 8 \
  --num-modes 9 \
  --skip-failed
```

## Redocking Gate

Any pose-power claim must pass a true-holo redocking gate. If docking into the experimental holo receptor cannot recover the reference ligand pose within the declared Top-N window and RMSD threshold, the target is not valid evidence for Cooper-TBDT docking improvement. In that case, report the docking run as diagnostic only and do not claim AF2-to-Cooper-TBDT pose rescue for that target.

Current held-out docking result is negative and limited. Among 21 Gold test targets, only one substrate-like target was cleanly runnable with standard Vina/Meeko/RDKit and a true-holo redocking gate: `tbdt_c5i2d9_3qlb_a` with ligand `EFE` from PDB `3QLB`. The true-holo redocking gate passes for this target, but AF2 and Cooper-TBDT both fail the pose-success endpoint:

| Receptor | Top-1 pose RMSD A | Pose success, RMSD < 2 A |
|---|---:|---:|
| true holo | 0.258 | 1/1 |
| raw AF2 | 8.699 | 0/1 |
| Cooper-TBDT scaffold blend | 8.699 | 0/1 |

Therefore there is currently no evidence that Cooper-TBDT significantly improves ligand docking success rate over raw AF2. The observed success-rate delta is `0.0` on the only gate-passing runnable substrate-like target. This should be reported as a limitation, not as a downstream application win.

Publication placement reminder: move the docking evidence to the supplement. In the main text, docking should be mentioned only as a secondary feasibility endpoint that currently does not support a ligand pose-rescue claim. The supplement should contain the full redocking-gated protocol, the EFE-only result table, per-target ligand inclusion/exclusion reasons, command lines, output artifacts, and the cobalamin/cofactor chemistry limitation.

The main feasibility bottleneck is ligand chemistry, not the absence of docking code. Most held-out ligand-bound candidates are cobalamin/cofactor-like or metal-chelated systems. The BtuB cyanocobalamin targets contain `CNC`, a large Co-containing ligand that is not well suited to the standard Vina/Meeko small-molecule workflow; other held-out HET groups are ions, detergents, spin labels, lipids, or crystallization additives. Cobalamin docking would require a separate cofactor-aware docking protocol before it can be used as fair evidence for pose rescue.

## Expansion Discovery

Discover candidate experimental structures through RCSB Search API, filter to X-ray/cryo-EM structures at 3.5 A or better, retain mapped TBDT-like polymer entities, and download matching AFDB v6 models:

```bash
python -m evopoint_da.pipeline.fetch_tbdt_structures discover \
  --out-manifest data/tbdt_expansion_manifest.csv \
  --download \
  --download-pae \
  --require-af2 \
  --max-search-results 80 \
  --max-retained 25 \
  --report-path artifacts/tbdt_v1/rcsb_discovery_report.json
```

The resulting expansion manifest is a candidate table. Rows with empty `region_annotation_json` can be converted into a smoke-trainable supervised manifest with RCSB/SIFTS/Pfam-derived region annotations:

```bash
python -m evopoint_da.pipeline.prepare_tbdt_training_manifest \
  --input-manifest data/tbdt_expansion_manifest.csv \
  --out-manifest data/tbdt_training_manifest.csv \
  --annotation-dir data/tbdt_region_annotations/auto \
  --report-path artifacts/tbdt_v1/prepare_training_manifest_report.json
```

Build the expanded training pairs and smoke-test graphs:

```bash
python -m evopoint_da.pipeline.build_tbdt_state_dataset \
  --manifest data/tbdt_training_manifest.csv \
  --out_dir data/processed_tbdt_training_pairs \
  --report_path artifacts/tbdt_v1/build_training_pairs_report.json \
  --require-core-alignment

python -m evopoint_da.pipeline.build_features_with_sasa \
  --pair_dir data/processed_tbdt_training_pairs \
  --output_dir data/processed_tbdt_training_graphs \
  --af2_structure_dir data/raw_af2 \
  --pae_dir data/raw_af2 \
  --allow-missing-pae \
  --smoke-test-features \
  --report_path artifacts/tbdt_v1/build_training_smoke_graphs_report.json
```

Current training config `configs/data/tbdt_state.yaml` points to `data/processed_tbdt_gold_graphs` and reads split labels from graph metadata. Automatic annotations are suitable for pipeline validation and early ablations; publication-grade region metrics still require manual curation of extracellular loops, TonB boxes, and substrate-contact residues.

## Mixed Corpus Discovery

Build the broader Gold/Silver/Bronze corpus. Gold is supervised TBDT AFDB-v6 to experimental displacement data; Silver is experimental beta-barrel membrane protein displacement data for auxiliary pretraining; Bronze is AFDB-only TBDT homolog coverage from UniProt PF00593/PF07715 and must be used only for pseudo/self-supervised objectives.

```bash
python -m evopoint_da.pipeline.build_tbdt_mixed_manifest \
  --out-manifest data/tbdt_mixed_manifest.csv \
  --out-gold-manifest data/tbdt_gold_manifest.csv \
  --out-silver-manifest data/tbdt_silver_manifest.csv \
  --out-bronze-manifest data/tbdt_bronze_manifest.csv \
  --report-path artifacts/tbdt_v1/tbdt_mixed_manifest_download_gold_report.json \
  --max-gold 220 \
  --max-silver 320 \
  --max-bronze 600 \
  --download-gold
```

The current run produced 1064 manifest rows: 144 Gold, 320 Silver, and 600 Bronze. Gold downloads are complete locally, with 129 RCSB legacy PDB files and 15 mmCIF fallbacks.

Convert Gold rows into supervised displacement pairs:

```bash
python -m evopoint_da.pipeline.prepare_tbdt_training_manifest \
  --input-manifest data/tbdt_gold_manifest.csv \
  --out-manifest data/tbdt_gold_training_manifest.csv \
  --annotation-dir data/tbdt_region_annotations/gold_auto \
  --report-path artifacts/tbdt_v1/prepare_gold_training_manifest_report.json

python -m evopoint_da.pipeline.build_tbdt_state_dataset \
  --manifest data/tbdt_gold_training_manifest.csv \
  --out_dir data/processed_tbdt_gold_pairs \
  --report_path artifacts/tbdt_v1/build_gold_pairs_report.json \
  --require-core-alignment
```

Current Gold result: 134/134 supervised pairs built after excluding 10 uncertain-state rows. Mean pair RMSD is 1.30 A and median pair RMSD is 0.96 A. Use Gold for held-out evaluation; keep Silver/Bronze out of primary metrics.

Build real ESMC/PCA Gold graphs:

```bash
python -m evopoint_da.pipeline.build_features_with_sasa \
  --pair_dir data/processed_tbdt_gold_pairs \
  --output_dir data/processed_tbdt_gold_graphs \
  --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
  --pca_path data/pca_esmc_128.pkl \
  --af2_structure_dir data/raw_af2 \
  --pae_dir data/raw_af2 \
  --report_path artifacts/tbdt_v1/build_gold_real_graphs_report.json
```

Current real graph result after rerunning Gold PAE downloads and invalidating stale PyG graph caches: 134/134 graphs built, feature dimension 144, median node count 673, median edge count 10768, and no PAE fallback. The earlier 93/134 PAE gap came from building Gold graphs after only the seed PAE files had been downloaded; Gold now has PAE JSON coverage for all 46 training-set UniProt IDs. New graph rebuilds are strict by default; use `--allow-missing-pae` only when the missing-PAE count is reported as a limitation.

The publication report now performs an independent strict input audit before table assembly. It verifies that every Gold pair and clean Silver pair maps all processed residue IDs back to the AFDB-v6 structure and that each PAE matrix can be strict-aligned by AF2 residue indices or residue IDs. Current status: Gold 134/134 passed and clean Silver 205/205 passed.

Train the recommended GVP TBDT model:

```bash
python train.py data=tbdt_state model=gvp_tbdt_module \
  trainer.max_epochs=40 data.batch_size=1 \
  study_name=tbdt_gold_gvp_scaffold_prior_v2 \
  logger.save_dir=logs/tbdt_gold_gvp_scaffold_prior_v2
```

The recommended default uses the slim GVP-TBDT loss: smooth-L1 displacement fitting with sample weights, TBDT region weights, and a weak high-pLDDT scaffold anchor. Current training uses the final single-model recipe in `configs/model/gvp_tbdt_module.yaml`: `lr=3e-4`, no LR warmup, `coord_init_gain=0.01`, `output_scale=2.0`, and `gvp_dropout=0.05`.

Strict split counts are train/val/test = 87/26/21 by UniProt group; `calib` reuses val because Gold has no separate calibration split. The Lightning logs keep the original training MSE/bin metrics; standalone TBDT reporting should use vector C-alpha region metrics from `eval_tbdt_state.py`.

### Article-facing model hierarchy

Use two explicitly separated neural reporting tiers in the paper:

1. **Primary neural baseline: single scaffold-prior model family.** This is one fixed GVP-TBDT training recipe with no post-hoc region switching and no test-set calibration, reported across five fixed training seeds. For the final internal freeze, use the validation-only `best-selection` checkpoint rule, where `val/disp_selection_mse = 0.7 * val/disp_1to2_mse + 0.3 * val/disp_1to5_mse`. The primary article-facing aggregate is `artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv`. For per-target plots that require one checkpoint, use the seed with median validation selector score, currently seed 404: `artifacts/tbdt_v1/seed_stability_best_selection/metrics/seed_404_best-selection_test.json`.
2. **Secondary validation-calibrated region blend.** This is the best reportable coordinate candidate, but it must be described as a validation-calibrated composition, not as the primary single neural model. It uses a scaffold-prior base, a plug-region source, and a TonB-box source with only validation-derived scale choices.

Checkpoint selection uses validation data only. For scaffold-prior sweeps, each candidate checkpoint is scored on Gold validation by the region-vector evaluator: the score is the mean MSE-improvement-vs-raw-AF2 over `eval`, `plug`, and `tonb_box`, minus `0.25 * max(0, barrel_core_predicted_displacement_mean - 0.05)`. Test metrics are reported only after the validation-selected checkpoint or validation-calibrated blend is fixed.

Current held-out Gold vector baselines are eval 1.811 A, plug 1.232 A, TonB box 6.881 A, and all residues 1.895 A for raw AFDB/zero displacement.

The final secondary blend is:

```bash
python main.py --step report_models -- --force
```

This runner trains the fixed report-facing Gold-only specialists and Silver pretrain/fine-tune model on the strict fixed-PAE graph features, exports validation/test predictions, and builds the blend. The blend uses the median-validation primary single-model checkpoint as base (`seed_404_best-selection`), a Gold plug/eval specialist for plug residues, and a Gold TonB specialist for TonB residues. It uses no test-set scale fitting. The plug multiplier is fit on Gold validation only (`scale=1.413`, 2195 validation residues). TonB is left unscaled (`scale=1.0`) because the validation TonB calibration set has too few residues for a stable scalar: 32 validation TonB residues, below the 100-residue minimum. The base model is also left unscaled.

Current test vector-region results for the two article-facing neural tiers:

| Tier | Method | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---|---:|---:|---:|---:|
| baseline | raw AFDB / zero displacement | 1.811 | 1.232 | 6.881 | 0.000 |
| primary neural baseline | single scaffold-prior model family, `best-selection`, 5-seed mean +/- std | 1.797 +/- 0.005 | 1.223 +/- 0.006 | 6.822 +/- 0.027 | 0.015 +/- 0.005 |
| secondary validation-calibrated candidate | scaffold-prior region blend | 1.793 | 1.214 | 6.834 | 0.023 |

The single model is the cleaner primary neural baseline for the main Methods and main-result claim. The region blend may be reported as a validation-calibrated coordinate candidate that improves eval/plug RMSD but does not improve TonB versus the primary single-model family. Historical Gold-only specialists, gate variants, and sweep variants should be framed as ablations or sensitivity analyses, not as independently test-selected model choices.

### Scientific reporting design

Do not reduce Cooper-TBDT to one model row and one scalar. The scientifically clean report should have four layers:

1. **Primary single-model family.** Fixed scaffold-prior training recipe, five fixed seeds, validation-only `best-selection` checkpoint rule, Gold test only. Report mean +/- std across seeds for eval, plug, TonB box, barrel-core predicted displacement, all-residue diagnostic RMSD, sample improvement rate, and paired per-target Delta RMSD confidence intervals. This is the main neural baseline.
2. **Endpoint families.** Report region masks and displacement bins separately. Region endpoints answer "where in the transporter"; displacement bins answer "how large is the motion." Use `<1 A` as small/noise/scaffold-dominated diagnostic, `1-2 A` as the main small functional displacement band for plug/eval, `2-5 A` as the broad functional/large local displacement band, and `>5 A` as a rare large-motion/TonB diagnostic. Do not use all-residue RMSD as a primary endpoint.
3. **Selector sensitivity.** Report `best-selection`, `best-disp1to5`, `best-disp1to2`, and `best-flex` as validation checkpoint selector sensitivity. This is not post-hoc model shopping; it documents whether the scientific conclusion depends on choosing a small-motion, broad-motion, or large-motion selector.
4. **Secondary validation-calibrated candidates.** Region blends and specialist sources are allowed as "best validation-calibrated coordinate candidates," but only after all component models are rerun on fixed-PAE Gold graphs and all calibration scalars are fit on Gold validation only.

This structure lets the paper say: the single neural family is stable, the small functional plug/eval signal is robust, TonB large-motion remains hard and should be reported with dedicated direction/centroid metrics, and any blend improvement is secondary rather than the core claim.

### Single-model seed stability

The final single scaffold-prior configuration was rerun with five fixed training seeds to show sensitivity to random initialization/training order while keeping the Gold metadata split fixed (`split_seed=42`, `split_source=metadata`). Training and prediction used CUDA, `num_workers=0`, `seed_everything(seed, workers=True)`, and deterministic Trainer settings. This run is not a sweep: the architecture, loss weights, max epochs, split, and checkpoint selector are fixed before final Gold test reporting.

```bash
python main.py --step seed_stability -- \
  --seeds 42,101,202,303,404 \
  --max-epochs 40 \
  --out-dir artifacts/tbdt_v1/seed_stability \
  --study-prefix tbdt_single_scaffold_prior_seed_stability
```

Fixed configuration: `lr=3e-4`, `barrel_core_loss_weight=0.05`, `eval_region_loss_weight=2.0`, `plug_loss_weight=2.5`, `tonb_box_loss_weight=4.0`, `substrate_contact_loss_weight=4.0`, `scaffold_anchor_weight=0.2`, `scaffold_anchor_plddt_min=80`, `coord_init_gain=0.01`, `output_scale=2.0`, and `gvp_dropout=0.05`. Primary checkpoint selection is `best-selection` on validation only.

Primary `best-selection` test-set seed stability:

| Seed | selected epoch | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 1.795 | 1.228 | 6.788 | 0.008 |
| 101 | 3 | 1.795 | 1.219 | 6.826 | 0.015 |
| 202 | 7 | 1.794 | 1.214 | 6.838 | 0.015 |
| 303 | 5 | 1.806 | 1.228 | 6.857 | 0.015 |
| 404 | 4 | 1.796 | 1.225 | 6.803 | 0.023 |
| mean +/- std | - | 1.797 +/- 0.005 | 1.223 +/- 0.006 | 6.822 +/- 0.027 | 0.015 +/- 0.005 |

The model is stable on the primary eval, plug, and barrel-core endpoints. TonB remains consistently improved versus raw AFDB/zero displacement but should still be framed as a hard, sparse-region endpoint. Full primary-selector artifacts are in `artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_report.md`, `seed_stability_summary.csv`, and `seed_stability_aggregate.csv`.

Checkpoint-selector sensitivity was evaluated after fixing Gold PAE, using the same five trained seed runs and no retraining. This is a sensitivity analysis, not a new primary-model selection step.

| Checkpoint selector | eval RMSD A mean +/- std | plug RMSD A mean +/- std | TonB RMSD A mean +/- std | barrel-core predicted mean A | selection score mean +/- std |
|---|---:|---:|---:|---:|---:|
| best-disp1to5 | 1.803 +/- 0.012 | 1.234 +/- 0.023 | 6.813 +/- 0.033 | 0.018 +/- 0.004 | 0.0086 +/- 0.0151 |
| best-disp1to2 | 1.797 +/- 0.005 | 1.223 +/- 0.006 | 6.822 +/- 0.027 | 0.015 +/- 0.005 | 0.0155 +/- 0.0054 |
| best-selection | 1.797 +/- 0.005 | 1.223 +/- 0.006 | 6.822 +/- 0.027 | 0.015 +/- 0.005 | 0.0155 +/- 0.0054 |
| best-flex | 1.811 +/- 0.035 | 1.251 +/- 0.048 | 6.791 +/- 0.128 | 0.016 +/- 0.009 | -0.0022 +/- 0.0407 |

`best-disp1to2` and `best-selection` selected the same checkpoints in these five runs and are slightly better than `best-disp1to5` on eval/plug/all-residue RMSD, but slightly worse on TonB RMSD. For the internal final-freeze design, use `best-selection` as the primary selector because it explicitly balances the small functional plug/eval band (`1-2 A`) with the broader functional band (`1-5 A`). Keep `best-disp1to5` as broad-band selector sensitivity and `best-flex` as a large-motion/TonB-biased sensitivity check.

The early selected epochs are expected under the current Gold-only setup, but they should be reported and controlled. Across five seeds, `best-selection` picks epochs 3-7 and `best-disp1to5` picks epochs 3-10. Later epochs increase predicted displacement magnitude and validation MSE, especially on flexible residues. This is consistent with small-data overfitting and displacement over-prediction, not with undertraining. Warmup or lower learning rate may move the selected epoch later, but the decisive test is whether it improves Gold validation/test under the same selector. Recommended optimizer sensitivity for the final freeze: `lr in {1e-4, 2e-4, 3e-4}` crossed with `lr_warmup_epochs in {0, 3, 5}`, all selected on Gold validation and reported as sensitivity rather than silently replacing the main recipe.

## Loss Gate Ablation

These are historical ablations that explain why the old global loss-gate machinery was removed from the active model. They are retained as design history only: the numeric table below was generated before the strict fixed-PAE graph/cache rebuild and is not part of the current publication reporting set. Current report-facing Gold-edge neural models are the fixed primary seed-stability family and the Gold-only specialist ablations rerun by `python main.py --step report_models -- --force`.

| Variant | all RMSD A | eval RMSD A | plug RMSD A | TonB RMSD A | Notes |
|---|---:|---:|---:|---:|---|
| gates off | 1.883 | 1.798 | 1.222 | 6.832 | Default baseline. |
| all gates | 1.891 | 1.806 | 1.223 | 6.882 | Bad; pLDDT/clash/aux/focus together over-regularize. |
| focus only | 1.882 | 1.798 | 1.222 | 6.832 | Neutral to tiny positive; not enough to replace default. |
| direction aux only | 1.884 | 1.798 | 1.219 | 6.850 | Useful signal for plug, but hurts TonB. |
| pLDDT L2 only | 1.892 | 1.806 | 1.227 | 6.868 | Bad; shrinks predictions too aggressively. |

No loss gate is kept in the active training module. A test blend using `focus_only` as the base, `direction_aux_only` for plug, and the existing TonB specialist gave all/eval/plug/TonB RMSD = 1.881/1.790/1.219/6.798 A, essentially tied with the previous region blend. That result was not strong enough to justify preserving the gate code path.

The old ablation artifacts remain in `artifacts/tbdt_v1/gate_ablation/loss_gate_ablation_summary.json` and `loss_gate_ablation_region_summary.csv`, but they should not be cited as current fixed-PAE results.

## Scaffold Prior

The current scientific default is a region-aware scaffold prior: AlphaFold/AFDB confidence metrics support treating high-pLDDT structured regions as reliable local backbones, while TBDT structural literature places the functional state signal mainly in plug, extracellular-loop, substrate-facing, switch/TonB-box regions rather than in a freely deforming barrel scaffold. Therefore the default GVP TBDT config sets:

```yaml
barrel_core_loss_weight: 0.05
eval_region_loss_weight: 2.0
plug_loss_weight: 2.5
tonb_box_loss_weight: 4.0
substrate_contact_loss_weight: 4.0
scaffold_anchor_weight: 0.2
scaffold_anchor_plddt_min: 80.0
```

This down-weights supervised displacement fitting on the barrel core and adds a zero-displacement anchor only on high-confidence barrel-core residues. It avoids the earlier pLDDT L2 mistake: low-confidence flexible regions are not globally shrunk just because their pLDDT is low.

The detailed scaffold-prior sweep is a sensitivity analysis, not the article's model-selection narrative. It decomposes the mechanism into core down-weighting, scaffold-anchor strength, pLDDT anchor threshold, individual region components, and winner-focused optimizer/head/dropout settings. The early sweep directories are design-history artifacts; after the fixed-PAE graph/cache rebuild, use the fixed primary seed-stability run and `report_models` reruns for reportable neural metrics.

```bash
python -m evopoint_da.pipeline.run_tbdt_scaffold_prior_sweep \
  --out-dir artifacts/tbdt_v1/scaffold_prior_sweep \
  --study-prefix tbdt_scaffold_prior_sweep \
  --max-epochs 30 \
  --checkpoint-selector best-disp1to5 \
  --checkpoint-selector best-flex \
  --device cuda

python -m evopoint_da.pipeline.summarize_tbdt_scaffold_prior_sweeps \
  --summary-csv artifacts/tbdt_v1/scaffold_prior_sweep/scaffold_prior_sweep_summary.csv \
  --summary-csv artifacts/tbdt_v1/scaffold_prior_fine_sweep/scaffold_prior_sweep_summary.csv \
  --summary-csv artifacts/tbdt_v1/scaffold_prior_winner_sweep/scaffold_prior_sweep_summary.csv \
  --out-dir artifacts/tbdt_v1/scaffold_prior_sweep_report \
  --top-k 20
```

The highest validation-score mechanism in the historical sweep was TonB-only weighting plus core down-weighting and scaffold anchor (`sp17_tonb_only_anchor`), which provided mechanistic evidence that a scaffold anchor can suppress barrel motion. The balanced validation-selected configuration `sp30_region_anchor_w005_lr1e4` motivated the final fixed recipe, but its old sweep metrics are superseded by the fixed-PAE reruns below. Full historical tables and the tradeoff plot remain in `artifacts/tbdt_v1/scaffold_prior_sweep_report/` for traceability only.

Primary single-model result. The historical no-prior row is retained only as design context; it is not part of the current fixed-PAE reporting set.

| Variant | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---:|---:|---:|---:|
| historical no-prior baseline, pre-fixed-PAE context | 1.798 | 1.222 | 6.832 | 0.152 |
| single scaffold-prior model family, 5-seed mean | 1.797 | 1.223 | 6.822 | 0.015 |

The secondary validation-calibrated scaffold-prior blend uses the median-validation primary checkpoint as the base, the predefined plug/eval specialist for plug residues, and the TonB specialist for TonB residues. The plug scale is validation-fit (`1.413`); TonB and base predictions are identity-scaled:

| Variant | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---:|---:|---:|---:|
| historical blend, pre-fixed-PAE context | 1.790 | 1.218 | 6.798 | 0.152 |
| validation-calibrated scaffold-prior blend | 1.793 | 1.214 | 6.834 | 0.023 |

This is a secondary validation-calibrated reporting candidate, not the primary single neural baseline. It improves eval/plug RMSD while keeping barrel-core deformation small, but it does not improve TonB relative to the primary single-model family. Its all-residue RMSD is not the optimization target and is less meaningful for TBDT state correction because it rewards fitting scaffold differences that should be treated as a reliable frame. Full current artifacts are in `artifacts/tbdt_v1/report_models/` and `artifacts/tbdt_v1/publication_report/`.

## Mechanistic Paired Evaluation

The detailed evaluator now reports the three checks needed before claiming a small region-RMSD improvement is meaningful:

1. Per-target paired delta: `Delta RMSD = RMSD(method) - RMSD(raw AF2)`, with `n_improved`, `n_worsened`, bootstrap 95% CI, and a one-sided Wilcoxon signed-rank test for method RMSD lower than raw AF2.
2. Fine plug regions: `plug_core`, `plug_apical_loop`, and `plug_extension_nt` are reported in addition to plug, TonB box, extracellular loops, substrate contacts, and barrel core. The current automatic split uses a transparent sequence-order heuristic unless explicit masks are provided.
3. TonB state metrics: TonB centroid exposure delta, distance to AF2 barrel/plug reference centroids, centroid displacement cosine, direction-compatible rate, exposure-state classification, and N-terminal plug-extension displacement.

Run the detailed evaluation and report builder:

```bash
python -m evopoint_da.pipeline.eval_tbdt_state \
  $(cat artifacts/tbdt_v1/test_graph_files.txt) \
  --predictions artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test \
  --output-json artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.json \
  --output-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.csv \
  --paired-delta-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_paired_delta.csv \
  --tonb-metrics-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_tonb_state_metrics.csv

python -m evopoint_da.pipeline.build_tbdt_mechanistic_eval_report \
  --metric-json scaffold_blend=artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.json \
  --metric-json primary_single_seed404=artifacts/tbdt_v1/seed_stability_best_selection/metrics/seed_404_best-selection_test.json \
  --out-md artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_report.md \
  --out-csv artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_summary.csv
```

Current scaffold blend per-target readout is directionally consistent for coordinate RMSD: eval improves on 17/20 targets, plug on 16/19, plug apical loop on 14/19, TonB box on 13/15, and all residues on 18/21. Median Delta RMSD is negative for each of those regions and the Wilcoxon one-sided p-values are <=0.002. The signal is stronger in `plug_apical_loop` than in `plug_core`, which confirms that whole-plug RMSD was diluting the flexible apical subregion.

TonB needs separate reporting. The scaffold blend improves TonB coordinate RMSD on 13/15 targets, but exposure-state accuracy is only 20% because most predicted TonB boxes remain `unchanged` while the target labels are buried-like or exposed-like. Direction compatibility is also only 20% for the current blend, with median TonB centroid displacement cosine 0.149. Therefore the current claim should be: TonB coordinate RMSD has a weak positive signal, but TonB direction/exposure-state compatibility is not solved.

Full tables are in `artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_report.md`.

## Coordinate Baselines

Coordinate baselines produce per-residue displacement vectors and enter the primary region RMSD endpoint. They are designed to test whether Cooper-TBDT is doing more than simple priors.

Run interpretable coordinate baselines:

```bash
python main.py --step coordinate_baselines -- \
  --data-dir data/processed_tbdt_gold_graphs \
  --split test \
  --donor-split train \
  --donor-split val \
  --split-source metadata \
  --output-root artifacts/tbdt_v1/coordinate_baselines
```

Implemented coordinate baselines:

- `global_region_mean`: train/val mean displacement vector by region.
- `family_region_mean`: family-specific region mean when available, with global fallback.
- `state_region_mean`: state-specific region mean when available, with global fallback.
- `family_state_region_mean`: family/state-specific region mean when available, with lower-order fallback.
- `region_centroid_shift`: rigid region centroid shift learned from train/val.
- `family_region_centroid_shift`, `state_region_centroid_shift`, `family_state_region_centroid_shift`: grouped rigid-region variants.
- `plug_rigid_shift`: plug-only rigid shift.
- `plug_apical_loop_rigid_shift`: apical-loop-only rigid shift.
- `tonb_box_rigid_shift`: TonB-box-only rigid shift.
- `barrel_frame_ridge`: ridge regression in a TBDT barrel frame, predicting radial/tangential/axial displacement components from region/state/family/substrate/geometry/light structure features.

The barrel-frame baseline defines a local frame from the barrel core: axial direction from PCA, radial direction from the barrel axis to each residue, and tangential direction by cross product. It predicts displacement in that frame, then transforms back to 3D. This is the lightest scientific test of whether graph message passing is needed beyond a geometry-aware linear prior.

Artifacts:

- `artifacts/tbdt_v1/coordinate_baselines/coordinate_baseline_report.json`
- `artifacts/tbdt_v1/coordinate_baselines/coordinate_baseline_assignment_summary.csv`
- Per-baseline prediction directories and `*_region_metrics.json` files.

## Template Baselines

Template baselines are non-neural coordinate baselines. Donor samples come only from train/val pairs; donor AF2 scaffolds are aligned to the target AF2 scaffold, donor AF2-to-experimental displacement vectors are rotated into the target frame, and vectors are copied onto aligned residues.

Internal sequence/template baselines:

```bash
python -m evopoint_da.pipeline.build_tbdt_template_baselines \
  --data-dir data/processed_tbdt_gold_graphs \
  --pair-dir data/processed_tbdt_gold_pairs \
  --split test \
  --donor-split train \
  --donor-split val \
  --split-source metadata \
  --min-identity 0.15 \
  --min-target-coverage 0.05 \
  --average-top-k 10 \
  --output-root artifacts/tbdt_v1/template_baselines
```

External structure-search template baselines:

```bash
python main.py --step structure_template_baselines -- \
  --threads 2 \
  --tool-timeout 1200
```

Implemented external template baselines:

- `foldseek_nearest_template`: Foldseek searches train/val AF2 structures and selects the nearest donor by structure score.
- `usalign_nearest_template`: US-align searches train/val AF2 structures and selects the nearest donor by TM-score.

Important implementation detail: these are not just external selectors followed by sequence alignment. Foldseek uses its `qaln/taln` structure alignment strings; US-align reruns pairwise alignment for the selected donor and parses the structural alignment. Both mappings are converted through processed AF2 residue indices before rotating/transferring donor displacement vectors. Same-UniProt donor leakage is excluded by default.

Per-target external-tool failures abort by default. A zero-displacement failure record is allowed only when `--allow-failed-zero-fallback` is explicitly passed, and such a run should be reported as a limitation rather than mixed silently into baseline metrics.

Current coordinate results on held-out Gold test:

| Method | eval RMSD A | plug RMSD A | TonB RMSD A | MSE improvement vs raw AF2 on eval |
|---|---:|---:|---:|---:|
| raw AF2 / zero | 1.811 | 1.232 | 6.881 | 0.000 |
| Foldseek nearest-template transfer | 2.050 | 1.509 | 7.227 | -0.281 |
| US-align nearest-template transfer | 2.114 | 1.585 | 7.304 | -0.362 |
| nearest template transfer | 2.163 | 1.713 | 6.950 | -0.427 |
| family/state average transfer | 2.110 | 1.646 | 6.926 | -0.357 |
| Cooper-TBDT single scaffold-prior (`best-selection`, 5-seed mean) | 1.797 | 1.223 | 6.822 | 0.016 |
| Cooper-TBDT single scaffold-prior representative seed 404 | 1.796 | 1.225 | 6.803 | 0.017 |
| Cooper-TBDT validation-calibrated blend | 1.793 | 1.214 | 6.834 | 0.020 |

Interpretation: Foldseek and US-align are important negative controls. They show that generic nearest-template structure search does not solve the coordinate endpoint, even with structure-alignment residue mapping. This is a stronger external comparison than only using an internally written template baseline. In article tables, label the 5-seed `Cooper-TBDT single scaffold-prior` aggregate as the primary neural baseline, use the representative seed 404 only for per-target/paired plots, and label `Cooper-TBDT validation-calibrated blend` as the secondary candidate; the blend improves eval/plug but does not improve TonB versus the primary single-model family.

Artifacts:

- `artifacts/tbdt_v1/template_baselines/template_baseline_report.json`
- `artifacts/tbdt_v1/template_baselines/external_template_baseline_report.json`
- `artifacts/tbdt_v1/template_baselines/external_template_baseline_selection.csv`

## Residue-Shift Classification Curves

ROC/PR reporting uses a score-only task so external baselines and Cooper-TBDT predictions have the same output contract. A residue is positive when its experimental target displacement magnitude is at least 1.0 A. Cooper-TBDT is scored by predicted displacement magnitude, AF2 low-pLDDT by `1 - pLDDT`, and AF2 surface exposure by RSA. This classification endpoint measures whether a method ranks state-shift residues correctly; it does not replace region RMSD as the primary coordinate endpoint.

```bash
python -m evopoint_da.pipeline.eval_tbdt_classification_curves \
  --sample-list artifacts/tbdt_v1/test_graph_files.txt \
  --prediction cooper_tbdt_scaffold_blend=artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test \
  --score-baseline prody_anm_mobility=artifacts/tbdt_v1/external_score_baselines/prody_anm_mobility \
  --score-baseline prody_gnm_mobility=artifacts/tbdt_v1/external_score_baselines/prody_gnm_mobility \
  --score-baseline iupred2a_long=artifacts/tbdt_v1/external_score_baselines/iupred2a_long \
  --score-baseline p2rank_pocket_score=artifacts/tbdt_v1/external_score_baselines/p2rank_pocket_score \
  --score-baseline fpocket_pocket_score=artifacts/tbdt_v1/external_score_baselines/fpocket_pocket_score \
  --score-baseline protcross_pocket_score=artifacts/tbdt_v1/external_score_baselines/protcross_pocket_score \
  --external-baseline af2_low_plddt \
  --external-baseline af2_surface_rsa \
  --region eval \
  --region plug \
  --region tonb_box \
  --region all \
  --positive-threshold 1.0 \
  --out-dir artifacts/tbdt_v1/external_baseline_curves
```

The preferred high-level runner builds the external score baselines and then invokes the same ROC/PR evaluator:

```bash
python main.py --step external_baselines -- \
  --threads 1 \
  --tool-timeout 1200
```

Implemented score-only localization baselines:

- Cooper-TBDT predicted displacement magnitude.
- AF2 low-pLDDT.
- AF2 surface RSA.
- ProDy-style ANM mobility.
- ProDy-style GNM mobility.
- IUPred2A long disorder.
- P2Rank pocket-residue score.
- fpocket pocket-residue score.
- ProtCross pocket-residue score from `/home/zero/ProtCross`.

ProDy-style ANM/GNM is a local elastic-network implementation with the same scientific contract as ProDy-style mobility, not a direct import of the ProDy package. ProtCross uses the local checkpoint/PCA under `/home/zero/ProtCross` and the local ESM-C weights symlinked from `esmc_weights/esmc_600m_2024_12_v0.pth`.

Current held-out test residue-shift localization results:

| Region | Method | AUROC | AP |
|---|---|---:|---:|
| eval | Cooper-TBDT scaffold blend | 0.684 | 0.474 |
| eval | AF2 low pLDDT | 0.813 | 0.676 |
| eval | AF2 surface RSA | 0.645 | 0.428 |
| eval | ProDy-style ANM mobility | 0.727 | 0.517 |
| eval | ProDy-style GNM mobility | 0.608 | 0.480 |
| eval | IUPred2A long disorder | 0.413 | 0.231 |
| eval | P2Rank pocket-residue score | 0.535 | 0.285 |
| eval | fpocket pocket-residue score | 0.488 | 0.293 |
| eval | ProtCross pocket-residue score | 0.562 | 0.380 |
| plug | Cooper-TBDT scaffold blend | 0.661 | 0.380 |
| plug | AF2 low pLDDT | 0.786 | 0.590 |
| plug | AF2 surface RSA | 0.620 | 0.337 |
| plug | ProDy-style ANM mobility | 0.702 | 0.448 |
| plug | ProDy-style GNM mobility | 0.549 | 0.345 |
| plug | IUPred2A long disorder | 0.367 | 0.185 |
| plug | P2Rank pocket-residue score | 0.578 | 0.278 |
| plug | fpocket pocket-residue score | 0.517 | 0.281 |
| plug | ProtCross pocket-residue score | 0.607 | 0.390 |
| TonB box | Cooper-TBDT scaffold blend | 0.073 | 0.971 |
| TonB box | AF2 low pLDDT | 0.128 | 0.976 |
| TonB box | AF2 surface RSA | 0.268 | 0.984 |
| TonB box | ProDy-style ANM mobility | 0.146 | 0.978 |
| TonB box | ProDy-style GNM mobility | 0.183 | 0.980 |
| TonB box | IUPred2A long disorder | 0.829 | 0.997 |
| TonB box | P2Rank pocket-residue score | 0.768 | 0.996 |
| TonB box | fpocket pocket-residue score | 0.573 | 0.990 |
| TonB box | ProtCross pocket-residue score | 0.146 | 0.978 |

The TonB-box PR numbers are inflated because this held-out slice has 82 positives and only 1 negative; AUROC is the safer diagnostic for that region, and the current blend is poor on TonB residue-ranking despite weakly improving TonB coordinate RMSD. For eval and plug residue localization, low AF2 pLDDT and ProDy-style ANM are strong external baselines and should be reported alongside the model. Curve images and raw points are in `artifacts/tbdt_v1/external_baseline_curves/`.

## External Baseline Feasibility Notes

Four external methods were evaluated for feasibility:

- MODELLER nearest-holo-template baseline: not run. MODELLER is not installed and requires a MODELLER license key. The license-free nearest-template and external Foldseek/US-align transfer baselines cover the non-neural template-modeling question without pretending to be MODELLER.
- ProDy ANM/GNM mobility baseline: implemented as ProDy-style elastic-network mobility scores. Direct ProDy dependency is not required for the scientific question.
- DynaMine sequence dynamics baseline: not run. The Bio2Byte service/API requires authenticated access; using an amino-acid propensity proxy would not be a valid DynaMine baseline.
- IUPred2A disorder baseline: implemented through the public IUPred2A REST endpoint by UniProt accession, then aligned back to the processed sample sequence.

Pocket/localization external tools:

- P2Rank 2.5.1 was installed locally under `artifacts/tbdt_v1/external_tools/p2rank_2.5.1`.
- fpocket was built locally under `artifacts/tbdt_v1/external_tools/fpocket`.
- ProtCross was run from `/home/zero/ProtCross`.

All P2Rank, fpocket, and ProtCross score files completed successfully for the 21 held-out Gold test targets.
Per-sample failures in these external score baselines abort by default. A zero-valued score file is allowed only with explicit `--allow-failed-zero-fallback`.

## Publication Report Bundle

The full reporting bundle is generated by:

```bash
python -m evopoint_da.pipeline.build_tbdt_publication_report \
  --out-dir artifacts/tbdt_v1/publication_report \
  --bootstrap-iter 5000 \
  --bootstrap-seed 42
```

It writes dataset summaries, split-leakage checks, graph region distributions, coordinate metrics with bootstrap confidence intervals, ROC/PR summaries, template coverage, quality flags, and file hashes. Start from `artifacts/tbdt_v1/publication_report/publication_report.md`.

Current report bundle status:

- `coordinate_metric_rows`: 115
- `selector_sensitivity_rows`: 20
- `neural_comparison_rows`: 24
- `displacement_bin_rows`: 105
- `classification_metric_rows`: 36
- Split leakage check: passed.
- Strict input checks: Gold 134/134 passed; clean Silver 205/205 passed.
- Quality flags: TonB-box ROC/PR class imbalance, low internal nearest-template target coverage, and AF2 low-pLDDT outperforming model magnitude on eval residue-localization AUROC.

## Silver/Bronze Assets And Pretraining Plan

Silver and Bronze assets are downloaded with the existing manifest, without rerunning discovery:

```bash
python -m evopoint_da.pipeline.download_tbdt_manifest_assets \
  --manifest data/tbdt_mixed_manifest.csv \
  --tier silver \
  --tier bronze \
  --workers 12 \
  --sync-tier-manifests \
  --report-path artifacts/tbdt_v1/download_silver_bronze_assets_report.json
```

Current asset status: Silver has 320/320 usable rows with experimental RCSB structure and AFDB v6 model. Bronze has 598/600 usable AFDB-v6 rows; UniProt accessions `P44523` and `Q8CVJ0` have no AlphaFold model available from the AlphaFold API or AFDB v1-v6 files and should be filtered from AFDB-only pretraining.

Recommended first candidate is a two-stage Silver auxiliary pretrain followed by Gold fine-tuning:

1. Rebuild Silver with the same strict feature policy as Gold, including real PAE rather than zero-filled PAE. This is now complete for the clean Silver subset: `artifacts/tbdt_v1/build_silver_clean_real_graphs_report.json` reports 205/205 processed graphs, zero skips, zero PAE fallback, and 4 invalidated dataset caches.
2. Use Silver as geometric/representation pretraining, not as direct TBDT state supervision. Silver rows have `state_label=unknown`, mostly no substrate, and generic beta-barrel displacement labels. They can teach AF2-to-experimental geometry correction and scaffold regularity, but they do not define plug/TonB state-change biology.
3. Prefer either encoder pretraining or a low-weight multi-task objective (`label_weight` around 0.1-0.3), with Silver loss restricted to scaffold/high-confidence or generic geometry targets. Do not let Silver unknown-state labels train plug/TonB functional displacement directly.
4. Fine-tune on Gold with the current TBDT region weights and metadata split. Select checkpoints only on Gold validation metrics and report only Gold held-out test metrics.

Current Silver preprocessing keeps only conservative auxiliary pairs: from 293 built Silver pairs, 205 pass `pair_rmsd <= 3.0 A` and `n_residues >= 120`. These 205 graphs were rebuilt with real ESMC/PCA, real PAE edge features, and strict AFDB-v6 structure resolution in `data/processed_tbdt_silver_graphs_clean`. A strict fixed-PAE Silver pretrain followed by Gold fine-tuning was rerun in `artifacts/tbdt_v1/report_models/`; it did not beat the Gold-only primary single-model family on held-out Gold metrics:

| Model | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---:|---:|---:|---:|
| primary single scaffold-prior, 5-seed mean | 1.797 | 1.223 | 6.822 | 0.015 |
| Silver pretrain -> Gold fine-tune, seed 42 | 1.803 | 1.231 | 6.829 | 0.008 |

Silver can plausibly reduce early overfitting because Gold has only 87 training graphs, 26 validation graphs, and only 32 validation TonB residues. However, Silver will help only if it regularizes the encoder/scaffold geometry; naive mixing can hurt by teaching generic beta-barrel AF2 residuals instead of TBDT state changes. The clean experiment is: Silver encoder/geometric pretrain -> Gold fine-tune -> Gold validation `best-selection` checkpoint -> Gold test report, compared against the Gold-only 5-seed family with the same seeds and selectors.

Bronze should not be used as ordinary zero-displacement supervision over all residues because that would train away the local state corrections. The safest weak-supervision use is scaffold-only regularization: build AFDB-only graphs, set pseudo `y_delta=0` only on high-confidence barrel/core-like residues, set loop/TonB/low-confidence residues to zero loss weight, and mix this objective at a small weight, e.g. 0.05-0.1. A better second candidate is coordinate-denoising pretraining on Bronze: perturb local coordinates and train the model to reconstruct the original AFDB scaffold only on high-confidence structural regions, then discard that head/objective before Gold fine-tuning.

Do not mix Bronze pseudo labels into Gold validation or test. Treat Bronze as representation/scaffold regularization, not as evidence that functional loop or TonB-state displacements are correct.

## Current Public-Code Entry Points

`main.py` provides stable aliases for reproducible pipeline stages:

```bash
python main.py --step build_mixed_manifest -- --help
python main.py --step build_pairs -- --help
python main.py --step build_features -- --help
python main.py --step train -- --help
python main.py --step predict_graphs -- --help
python main.py --step blend_predictions -- --help
python main.py --step eval_regions -- --help
python main.py --step template_baselines -- --help
python main.py --step structure_template_baselines -- --help
python main.py --step coordinate_baselines -- --help
python main.py --step external_baselines -- --help
python main.py --step eval_classification -- --help
python main.py --step seed_stability -- --help
python main.py --step report_models -- --help
python main.py --step publication_report -- --help
python main.py --step prepare_docking_manifest -- --help
python main.py --step docking_eval -- --help
```

Deleted interfaces:

- `run_Predict.py`
- generic protein displacement dataset builder
- generic prediction feature builder
- generic `eval_run` calibration path
- EGNN/backbone-selection configs
