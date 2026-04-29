"""Print the current HoloShift-TBDT project handoff note.

This is intentionally a script rather than README/docs content. Markdown notes
under docs/ are local-only working material and are ignored by git.
"""

from __future__ import annotations


HANDOFF = r"""
# HoloShift-TBDT Handoff

## Purpose

HoloShift-TBDT is now a focused state-conditioned displacement benchmark for
TonB-dependent beta-barrel transporters. The supervised target is local C-alpha
displacement from an AFDB/AF2-like start structure to an experimental target
state. The primary reporting unit is structural region, not full-chain RMSD.

Primary regions:
- barrel_core
- plug
- plug_core
- plug_apical_loop
- plug_extension_nt
- extracellular_loop
- tonb_box
- substrate_contact
- eval

## Current Project Boundary

Retained:
- TBDT Gold/Silver/Bronze manifest discovery and asset download.
- AFDB-v6 to experimental paired displacement construction.
- Real ESMC/PCA plus structural/GVP graph feature construction.
- GVP-only TBDT model with family/state/substrate/region conditioning.
- Region-weighted smooth-L1 displacement loss plus high-pLDDT barrel scaffold anchor.
- Batch graph prediction, region blending, template baselines, ROC/PR curves, mechanistic evaluation, and publication report bundle.
- Docking evaluation as a secondary endpoint with true-holo redocking gate logic.

Removed:
- EGNN and backbone selection.
- Generic protein displacement dataset builder.
- Generic prediction/PDB-shifting interface.
- Generic prediction feature builder.
- Generic conformal/eval_run calibration path.
- Old PocketMiner/PoseBusters/binding-readiness scripts.
- Publishable docs/ Markdown content.

## Core Commands

Build Gold supervised pairs:

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

Build real graphs:

    python -m evopoint_da.pipeline.build_features_with_sasa \
      --pair_dir data/processed_tbdt_gold_pairs \
      --output_dir data/processed_tbdt_gold_graphs \
      --esm_weights esmc_weights/esmc_600m_2024_12_v0.pth \
      --pca_path data/pca_esmc_128.pkl \
      --af2_structure_dir data/raw_af2 \
      --pae_dir data/raw_af2 \
      --allow-missing-pae \
      --report_path artifacts/tbdt_v1/build_gold_real_graphs_report.json

Train:

    python train.py trainer.max_epochs=40 study_name=tbdt_gold_gvp_scaffold_prior

Predict held-out graph split:

    python -m evopoint_da.pipeline.predict_tbdt_graphs \
      --ckpt checkpoints/<study>/<run>/best-disp1to5-*.ckpt \
      --data-dir data/processed_tbdt_gold_graphs \
      --split test \
      --output-dir artifacts/tbdt_v1/predictions/<run> \
      --report-path artifacts/tbdt_v1/predictions/<run>_report.json

Evaluate:

    python -m evopoint_da.pipeline.eval_tbdt_state \
      data/processed_tbdt_gold_graphs \
      --predictions artifacts/tbdt_v1/predictions/<run> \
      --output-json artifacts/tbdt_v1/model_region_metrics.json \
      --output-csv artifacts/tbdt_v1/model_region_metrics.csv

Publication bundle:

    python -m evopoint_da.pipeline.build_tbdt_publication_report \
      --out-dir artifacts/tbdt_v1/publication_report \
      --bootstrap-iter 5000 \
      --bootstrap-seed 42

## Current Scientific Conclusions

Raw AFDB/zero displacement remains a strong baseline because many barrel-core
coordinates are already reliable. The meaningful signal is local and region
dependent.

Current reportable model family:
- GVP-TBDT with region conditioning.
- Smooth-L1 node displacement loss.
- Region weights emphasize eval/plug/TonB/substrate-contact residues.
- Barrel core is down-weighted and high-pLDDT barrel residues get a weak
  zero-displacement scaffold anchor.

Best reported behavior from the current internal artifacts:
- Functional-region RMSD improves modestly but consistently versus raw AFDB.
- Scaffold prior sharply reduces predicted barrel-core motion.
- Plug apical loops show clearer signal than whole plug.
- TonB coordinate/direction signal is present, but exposure-state classification is not solved.
- Template-transfer baselines underperform raw AFDB on the current held-out split, largely due to low coverage.
- AF2 low-pLDDT is a strong external score baseline for residue shift localization and must be reported.

## Module Map

src/evopoint_da/data/
- alignment.py: sequence/Kabsch alignment utilities, including barrel-core aligned displacement target computation.
- dataset.py: strict PyG dataset loader for processed TBDT graph samples.
- datamodule.py: Lightning data module with train/val/test split leakage checks.
- features.py: ESMC/PCA and structural node feature helpers.
- graph.py: KNN/PAE edges and GVP scalar/vector feature construction.
- structure.py: PDB/mmCIF parsing and residue-id utilities.
- tbdt.py: region vocabulary, state/substrate IDs, annotation parsing, masks, and loss weights.

src/evopoint_da/models/
- backbones/gvp.py: GVP backbone.
- module.py: GVP-only Lightning module for TBDT displacement training and evaluation logging.

src/evopoint_da/pipeline/
- fetch_tbdt_structures.py: seed/expansion structure discovery and download.
- build_tbdt_mixed_manifest.py: Gold/Silver/Bronze corpus manifest construction.
- download_tbdt_manifest_assets.py: asset download for existing manifests.
- prepare_tbdt_training_manifest.py: supervised manifest normalization and auto region annotations.
- build_tbdt_state_dataset.py: AFDB-v6 to experimental state displacement pair builder.
- build_features_with_sasa.py: pair-to-graph feature builder.
- predict_tbdt_graphs.py: strict graph-level batch inference.
- blend_tbdt_predictions.py: validation-calibrated region blending of prediction directories.
- eval_tbdt_state.py: CLI wrapper for region displacement evaluation.
- tbdt_state_eval_core.py: reusable evaluation core.
- build_tbdt_template_baselines.py: non-neural template-transfer baselines.
- eval_tbdt_classification_curves.py: ROC/PR residue shift localization evaluation.
- build_tbdt_mechanistic_eval_report.py: mechanistic paired-delta and TonB report builder.
- run_tbdt_scaffold_prior_sweep.py: scaffold-prior ablation/tuning runner.
- summarize_tbdt_scaffold_prior_sweeps.py: sweep report summarizer.
- build_tbdt_publication_report.py: publication-grade report bundle.

src/evopoint_da/docking_eval/
- Secondary ligand-pose evaluation. Keep docking claims gated by true-holo redocking recovery.

## Git/Cloud Boundary

Commit:
- src/
- configs/
- tests/
- main.py
- train.py
- pyproject.toml
- environment.yml
- LICENSE
- tools/handoff_tbdt.py
- small manifest CSVs and curated region annotations explicitly whitelisted in .gitignore

Do not commit:
- docs/
- local_handoff/
- data/raw, processed pairs, processed graphs, AFDB/PDB/mmCIF assets
- esmc_weights/
- checkpoints/
- logs/
- val_metrics/
- outputs/
- artifacts/
- wandb/

## Known Risks

- Some automatic region annotations are only suitable for pipeline validation. Publication-grade region metrics require manual curation.
- PAE missingness is allowed only through the explicit --allow-missing-pae flag and must be reported.
- Current batch prediction outputs displacement tensors, not shifted PDBs. Reintroduce PDB export only as a TBDT-specific postprocessor if needed.
- Bronze should be used only for scaffold/denoising weak supervision, not as ordinary zero-displacement functional-region supervision.
""".strip()


def main() -> None:
    print(HANDOFF)


if __name__ == "__main__":
    main()
