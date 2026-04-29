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
- Optional PAE JSON files for graph edges.
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
  data/processed_tbdt_state_graphs \
  --output-json artifacts/tbdt_v1/zero_region_metrics.json \
  --output-csv artifacts/tbdt_v1/zero_region_metrics.csv
```

Evaluate model predictions:

```bash
python -m evopoint_da.pipeline.eval_tbdt_state \
  data/processed_tbdt_state_graphs \
  --predictions artifacts/tbdt_v1/predictions \
  --output-json artifacts/tbdt_v1/model_region_metrics.json \
  --output-csv artifacts/tbdt_v1/model_region_metrics.csv
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
  --structure holo=receptor_holo \
  --rmsd-threshold 2.0 \
  --exhaustiveness 8 \
  --num-modes 9
```

## Redocking Gate

Any pose-power claim must pass a true-holo redocking gate. If docking into the experimental holo receptor cannot recover the reference ligand pose within the declared Top-N window and RMSD threshold, the target is not valid evidence for Cooper-TBDT docking improvement. In that case, report the docking run as diagnostic only and do not claim AF2-to-Cooper-TBDT pose rescue for that target.

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
  --allow-missing-pae \
  --report_path artifacts/tbdt_v1/build_gold_real_graphs_report.json
```

Current real graph result: 134/134 graphs built, feature dimension 144, median node count 673, median edge count 10768. PAE was missing for 93 rows and was zero-filled in edge features. New graph rebuilds are strict by default; use `--allow-missing-pae` only when the missing-PAE count is reported as a limitation.

Train the recommended GVP TBDT model:

```bash
python train.py data=tbdt_state model=gvp_tbdt_module \
  trainer.max_epochs=40 data.batch_size=1 \
  study_name=tbdt_gold_gvp_scaffold_prior_v2 \
  logger.save_dir=logs/tbdt_gold_gvp_scaffold_prior_v2
```

The recommended default uses the slim GVP-TBDT loss: smooth-L1 displacement fitting with sample weights, TBDT region weights, and a weak high-pLDDT scaffold anchor. Current training uses `lr=1e-4`, no LR warmup, `coord_init_gain=0.01`, `output_scale=2.0`, and `gvp_dropout=0.05`.

Strict split counts are train/val/test = 87/26/21 by UniProt group; `calib` reuses val because Gold has no separate calibration split. The Lightning logs keep the original training MSE/bin metrics; standalone TBDT reporting should use vector C-alpha region metrics from `eval_tbdt_state.py`.

Current held-out Gold vector baselines are eval 1.811 A, plug 1.232 A, TonB box 6.881 A, and all residues 1.895 A for raw AFDB/zero displacement. The best single Gold-only checkpoints are complementary: `gold_only_balanced` is best for overall/eval, `gold_only_region` is best for plug, and `gold_only_tonb` is best for TonB box but hurts plug. The current reportable post-hoc model is therefore a region-blended model:

```bash
python -m evopoint_da.pipeline.blend_tbdt_predictions \
  --data-dir data/processed_tbdt_gold_graphs \
  --split test \
  --split-source metadata \
  --base-predictions artifacts/tbdt_v1/predictions/gold_only_balanced \
  --region-source plug=artifacts/tbdt_v1/predictions/gold_only_region \
  --region-source tonb_box=artifacts/tbdt_v1/predictions/gold_only_tonb \
  --auto-scale-region plug \
  --calibration-region-source plug=artifacts/tbdt_v1/predictions/gold_only_region_val \
  --min-calibration-residues 100 \
  --output-dir artifacts/tbdt_v1/predictions/region_blend_plugcal_test_scripted \
  --report-path artifacts/tbdt_v1/predictions/region_blend_plugcal_test_scripted_report.json
```

This uses no test-set scale fitting. The plug multiplier is fit on Gold validation only (`scale=1.287`, 2195 validation residues). TonB is left unscaled because the validation TonB calibration set has too few residues for a stable scalar.

Current test vector-region results for the earlier blended model:

| Region | Zero RMSD A | Blend error RMSD A | MSE improvement vs zero | Residue improved rate | Sample improved rate | Median sample delta A |
|---|---:|---:|---:|---:|---:|---:|
| all | 1.895 | 1.881 | 1.44% | 54.2% | 95.2% | 0.024 |
| eval | 1.811 | 1.790 | 2.31% | 52.9% | 75.0% | 0.055 |
| plug | 1.232 | 1.218 | 2.17% | 51.5% | 78.9% | 0.016 |
| TonB box | 6.881 | 6.798 | 2.41% | 89.2% | 86.7% | 0.343 |
| barrel core | 1.921 | 1.904 | 1.68% | 54.9% | 85.7% | 0.022 |

The barrel-core predicted displacement mean is 0.152 A, so the earlier model behaved as a small scaffold correction rather than a large core deformation. Full results are in `artifacts/tbdt_v1/tbdt_test_region_metrics_summary.json` and `.csv`.

## Loss Gate Ablation

These are historical ablations that explain why the old global loss-gate machinery was removed from the active model. They were run on the Gold real graph split with the same GVP architecture, `lr=3e-4`, 40 epochs, batch size 1, and validation-selected `best-disp1to5` checkpoints.

| Variant | all RMSD A | eval RMSD A | plug RMSD A | TonB RMSD A | Notes |
|---|---:|---:|---:|---:|---|
| gates off | 1.883 | 1.798 | 1.222 | 6.832 | Default baseline. |
| all gates | 1.891 | 1.806 | 1.223 | 6.882 | Bad; pLDDT/clash/aux/focus together over-regularize. |
| focus only | 1.882 | 1.798 | 1.222 | 6.832 | Neutral to tiny positive; not enough to replace default. |
| direction aux only | 1.884 | 1.798 | 1.219 | 6.850 | Useful signal for plug, but hurts TonB. |
| pLDDT L2 only | 1.892 | 1.806 | 1.227 | 6.868 | Bad; shrinks predictions too aggressively. |

No loss gate is kept in the active training module. A test blend using `focus_only` as the base, `direction_aux_only` for plug, and the existing TonB specialist gave all/eval/plug/TonB RMSD = 1.881/1.790/1.219/6.798 A, essentially tied with the previous region blend. That result was not strong enough to justify preserving the gate code path.

Full ablation artifacts are in `artifacts/tbdt_v1/gate_ablation/loss_gate_ablation_summary.json` and `loss_gate_ablation_region_summary.csv`.

## Scaffold Prior

The current scientific default is a region-aware scaffold prior: AlphaFold/AFDB confidence metrics support treating high-pLDDT structured regions as reliable local backbones, while TBDT structural literature places the functional state signal mainly in plug, extracellular-loop, substrate-facing, switch/TonB-box regions rather than in a freely deforming barrel scaffold. Therefore the default GVP TBDT config sets:

```yaml
barrel_core_loss_weight: 0.1
eval_region_loss_weight: 1.5
plug_loss_weight: 2.0
tonb_box_loss_weight: 3.0
substrate_contact_loss_weight: 3.0
scaffold_anchor_weight: 0.05
scaffold_anchor_plddt_min: 80.0
```

This down-weights supervised displacement fitting on the barrel core and adds a zero-displacement anchor only on high-confidence barrel-core residues. It avoids the earlier pLDDT L2 mistake: low-confidence flexible regions are not globally shrunk just because their pLDDT is low.

The detailed scaffold-prior sweep now decomposes the mechanism into core down-weighting, scaffold-anchor strength, pLDDT anchor threshold, individual region components, and winner-focused optimizer/head/dropout settings:

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

The highest validation-score mechanism is TonB-only weighting plus core down-weighting and scaffold anchor (`sp17_tonb_only_anchor`). It is the cleanest mechanistic proof that the scaffold anchor can suppress barrel motion: test barrel-core predicted displacement mean is 0.0095 A. The balanced default keeps the full plug/eval/TonB/substrate-contact prior active and uses the winner-focused validation-selected configuration `sp30_region_anchor_w005_lr1e4`.

Validation-selected held-out test metrics for the balanced default are eval RMSD 1.794 A, plug RMSD 1.217 A, TonB-box RMSD 6.833 A, and barrel-core predicted displacement mean 0.0266 A. Relative to the previous default (`sp06_region_anchor_w01_t80`), eval and plug improve substantially, TonB remains positive versus raw AFDB but is essentially tied/slightly lower, and the aggregate scaffold-prior score improves while keeping core movement well below 0.05 A. Full tables and the tradeoff plot are in `artifacts/tbdt_v1/scaffold_prior_sweep_report/`.

Single-model test result for `tbdt_scaffold_prior_region_weighted` versus the previous no-prior baseline:

| Variant | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---:|---:|---:|---:|
| previous no-prior baseline | 1.798 | 1.222 | 6.832 | 0.152 |
| scaffold prior | 1.794 | 1.221 | 6.815 | 0.034 |

The scaffold-prior blend uses scaffold prior as the base, the previous plug specialist for plug, and the strong scaffold-prior model for TonB:

| Variant | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core predicted mean A |
|---|---:|---:|---:|---:|
| previous blend | 1.790 | 1.218 | 6.798 | 0.152 |
| scaffold-prior blend | 1.787 | 1.218 | 6.778 | 0.034 |

This is the better scientific reporting candidate: it improves functional-region RMSD while sharply reducing barrel-core deformation. Its all-residue RMSD is not the optimization target and is less meaningful for TBDT state correction because it rewards fitting scaffold differences that should be treated as a reliable frame. Full artifacts are in `artifacts/tbdt_v1/scaffold_prior/scaffold_prior_summary.json` and `.csv`.

## Mechanistic Paired Evaluation

The detailed evaluator now reports the three checks needed before claiming a small region-RMSD improvement is meaningful:

1. Per-target paired delta: `Delta RMSD = RMSD(method) - RMSD(raw AF2)`, with `n_improved`, `n_worsened`, bootstrap 95% CI, and a one-sided Wilcoxon signed-rank test for method RMSD lower than raw AF2.
2. Fine plug regions: `plug_core`, `plug_apical_loop`, and `plug_extension_nt` are reported in addition to plug, TonB box, extracellular loops, substrate contacts, and barrel core. The current automatic split uses a transparent sequence-order heuristic unless explicit masks are provided.
3. TonB state metrics: TonB centroid exposure delta, distance to AF2 barrel/plug reference centroids, centroid displacement cosine, direction-compatible rate, exposure-state classification, and N-terminal plug-extension displacement.

Run the detailed evaluation and report builder:

```bash
python -m evopoint_da.pipeline.eval_tbdt_state \
  $(cat artifacts/tbdt_v1/test_graph_files.txt) \
  --predictions artifacts/tbdt_v1/predictions/region_blend_scaffold_prior_test \
  --output-json artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.json \
  --output-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.csv \
  --paired-delta-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_paired_delta.csv \
  --tonb-metrics-csv artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_tonb_state_metrics.csv

python -m evopoint_da.pipeline.build_tbdt_mechanistic_eval_report \
  --metric-json scaffold_blend=artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_detailed_metrics.json \
  --metric-json sp30_balanced_default=artifacts/tbdt_v1/mechanistic_eval/sp30_balanced_default_detailed_metrics.json \
  --out-md artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_report.md \
  --out-csv artifacts/tbdt_v1/mechanistic_eval/mechanistic_eval_summary.csv
```

Current scaffold blend per-target readout is directionally consistent: eval improves on 15/20 targets, plug on 15/19, plug apical loop on 14/19, TonB box on 13/15, and all residues on 18/21. Median Delta RMSD is negative for each of those regions and the Wilcoxon one-sided p-values are <0.002. The signal is stronger in `plug_apical_loop` than in `plug_core`, which confirms that whole-plug RMSD was diluting the flexible apical subregion.

TonB needs separate reporting. The scaffold blend improves TonB coordinate RMSD on 13/15 targets, but exposure-state accuracy is only 20% because most predicted TonB boxes remain `unchanged` while the target labels are buried-like or exposed-like. Direction compatibility is higher at 73%, with median TonB centroid displacement cosine 0.945. Therefore the current claim should be: TonB coordinate/direction signal is present, but TonB exposure-state compatibility is not solved.

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

Current coordinate results on held-out Gold test:

| Method | eval RMSD A | plug RMSD A | TonB RMSD A | MSE improvement vs raw AF2 on eval |
|---|---:|---:|---:|---:|
| raw AF2 / zero | 1.811 | 1.232 | 6.881 | 0.000 |
| Foldseek nearest-template transfer | 2.050 | 1.509 | 7.227 | -0.281 |
| US-align nearest-template transfer | 2.114 | 1.585 | 7.304 | -0.362 |
| nearest template transfer | 2.163 | 1.713 | 6.950 | -0.427 |
| family/state average transfer | 2.110 | 1.646 | 6.926 | -0.357 |
| Cooper-TBDT single scaffold-prior | 1.796 | 1.231 | 6.778 | 0.017 |
| Cooper-TBDT scaffold-prior blend | 1.787 | 1.218 | 6.778 | 0.026 |

Interpretation: Foldseek and US-align are important negative controls. They show that generic nearest-template structure search does not solve the coordinate endpoint, even with structure-alignment residue mapping. This is a stronger external comparison than only using an internally written template baseline.

Artifacts:

- `artifacts/tbdt_v1/template_baselines/template_baseline_report.json`
- `artifacts/tbdt_v1/template_baselines/external_template_baseline_report.json`
- `artifacts/tbdt_v1/template_baselines/external_template_baseline_selection.csv`

## Residue-Shift Classification Curves

ROC/PR reporting uses a score-only task so external baselines and Cooper-TBDT predictions have the same output contract. A residue is positive when its experimental target displacement magnitude is at least 1.0 A. Cooper-TBDT is scored by predicted displacement magnitude, AF2 low-pLDDT by `1 - pLDDT`, and AF2 surface exposure by RSA. This classification endpoint measures whether a method ranks state-shift residues correctly; it does not replace region RMSD as the primary coordinate endpoint.

```bash
python -m evopoint_da.pipeline.eval_tbdt_classification_curves \
  --sample-list artifacts/tbdt_v1/test_graph_files.txt \
  --prediction cooper_tbdt_scaffold_blend=artifacts/tbdt_v1/predictions/region_blend_scaffold_prior_test \
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
| eval | Cooper-TBDT scaffold blend | 0.651 | 0.504 |
| eval | AF2 low pLDDT | 0.813 | 0.676 |
| eval | AF2 surface RSA | 0.645 | 0.428 |
| eval | ProDy-style ANM mobility | 0.727 | 0.517 |
| eval | ProDy-style GNM mobility | 0.608 | 0.480 |
| eval | IUPred2A long disorder | 0.413 | 0.231 |
| eval | P2Rank pocket-residue score | 0.535 | 0.285 |
| eval | fpocket pocket-residue score | 0.488 | 0.293 |
| eval | ProtCross pocket-residue score | 0.562 | 0.380 |
| plug | Cooper-TBDT scaffold blend | 0.593 | 0.328 |
| plug | AF2 low pLDDT | 0.786 | 0.590 |
| plug | AF2 surface RSA | 0.620 | 0.337 |
| plug | ProDy-style ANM mobility | 0.702 | 0.448 |
| plug | ProDy-style GNM mobility | 0.549 | 0.345 |
| plug | IUPred2A long disorder | 0.367 | 0.185 |
| plug | P2Rank pocket-residue score | 0.578 | 0.278 |
| plug | fpocket pocket-residue score | 0.517 | 0.281 |
| plug | ProtCross pocket-residue score | 0.607 | 0.390 |
| TonB box | Cooper-TBDT scaffold blend | 0.756 | 0.997 |
| TonB box | AF2 low pLDDT | 0.128 | 0.976 |
| TonB box | AF2 surface RSA | 0.268 | 0.984 |
| TonB box | ProDy-style ANM mobility | 0.146 | 0.978 |
| TonB box | ProDy-style GNM mobility | 0.183 | 0.980 |
| TonB box | IUPred2A long disorder | 0.829 | 0.997 |
| TonB box | P2Rank pocket-residue score | 0.768 | 0.996 |
| TonB box | fpocket pocket-residue score | 0.598 | 0.990 |
| TonB box | ProtCross pocket-residue score | 0.744 | 0.996 |

The TonB-box PR numbers are inflated because this held-out slice has 82 positives and only 1 negative; AUROC is the safer diagnostic for that region. For eval and plug residue localization, low AF2 pLDDT and ProDy-style ANM are strong external baselines and should be reported alongside the model. Curve images and raw points are in `artifacts/tbdt_v1/external_baseline_curves/`.

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
- `classification_metric_rows`: 45
- Split leakage check: passed.
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

Recommended first candidate is a two-stage Silver-supervised pretrain followed by Gold fine-tuning:

1. Build Silver AFDB-to-experimental displacement pairs without TBDT-specific core alignment. Silver is beta-barrel auxiliary data, so use full-chain alignment or a generic beta-strand scaffold selector later. Keep state/substrate as unknown and use a low global label weight, e.g. 0.2-0.3.
2. Train the same GVP backbone on Silver for geometric correction only. Do not evaluate method claims on Silver; it is an auxiliary initialization source.
3. Fine-tune on Gold with the current TBDT region weights and metadata split. Select checkpoints only on Gold validation metrics and report only Gold held-out test metrics.

Current Silver preprocessing keeps only conservative auxiliary pairs: from 293 built Silver pairs, 205 pass `pair_rmsd <= 3.0 A` and `n_residues >= 120`. These 205 graphs were rebuilt with real ESMC/PCA features in `data/processed_tbdt_silver_graphs_clean`. A 12-epoch Silver pretrain followed by Gold fine-tuning was tested, but it did not beat the Gold-only region-blended result on the held-out Gold vector metrics. Keep this path as an initialization candidate, not as the current reportable model.

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
python main.py --step publication_report -- --help
```

Deleted interfaces:

- `run_Predict.py`
- generic protein displacement dataset builder
- generic prediction feature builder
- generic `eval_run` calibration path
- EGNN/backbone-selection configs
