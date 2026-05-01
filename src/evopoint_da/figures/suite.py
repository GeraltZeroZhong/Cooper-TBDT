"""Registry and audit helpers for the Cooper-TBDT figure suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evopoint_da.figures.io import write_csv_rows


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    module: str
    step: str
    out_name: str
    scientific_role: str
    primary_message: str
    caveat: str
    expected_data_files: tuple[str, ...]
    source_paths: tuple[str, ...]


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        figure_id="figure_1",
        title="Primary Model vs Raw AFDB",
        module="evopoint_da.figures.main_results",
        step="figure_main_results",
        out_name="main_result_primary_vs_raw_afdb",
        scientific_role="Primary result figure.",
        primary_message=(
            "The five-seed scaffold-prior model gives small but paired-consistent held-out "
            "improvements while keeping barrel-core displacement small; plug ROC/PR panels add a "
            "score-only residue-localization view with the Cooper blend and all external score baselines."
        ),
        caveat=(
            "Panel A is descriptive; the paired target-level delta panels are the stronger evidence. "
            "The blend is secondary and validation-calibrated. Plug ROC/PR is not a coordinate-RMSD "
            "endpoint; AF2 low pLDDT and ANM are strong localization baselines."
        ),
        expected_data_files=(
            "main_result_panel_a_values.csv",
            "main_result_primary_paired_delta_values.csv",
            "main_result_barrel_core_values.csv",
            "main_result_plug_localization_metric_values.csv",
            "main_result_plug_localization_seed_metric_values.csv",
            "main_result_plug_localization_curve_points.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/publication_report/coordinate_metrics_summary.csv",
            "artifacts/tbdt_v1/publication_report/primary_model_paired_delta_samples.csv",
            "artifacts/tbdt_v1/publication_report/primary_model_paired_delta_summary.csv",
            "artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv",
            "artifacts/tbdt_v1/seed_stability_best_selection/metrics/seed_*_best-selection_test.json",
            "artifacts/tbdt_v1/report_models/metrics/validation_calibrated_region_blend_test.json",
            "artifacts/tbdt_v1/external_baseline_curves/classification_curve_summary.csv",
            "artifacts/tbdt_v1/external_baseline_curves/classification_curve_points.csv",
            "artifacts/tbdt_v1/test_graph_files.txt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_42_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_101_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_202_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_303_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_404_best-selection_test/*.pt",
        ),
    ),
    FigureSpec(
        figure_id="figure_2",
        title="Gold Test Displacement Landscape",
        module="evopoint_da.figures.gold_test_displacement_landscape",
        step="figure_gold_test_displacement",
        out_name="gold_test_displacement_landscape",
        scientific_role="Dataset endpoint context figure.",
        primary_message=(
            "Evaluation and plug endpoints are dominated by small displacements, whereas TonB is a "
            "sparse larger-motion endpoint."
        ),
        caveat=(
            "This figure describes target-displacement scale; it does not measure model performance."
        ),
        expected_data_files=(
            "gold_test_raw_afdb_rmsd_values.csv",
            "gold_test_displacement_bin_values.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/publication_report/coordinate_metrics_summary.csv",
            "artifacts/tbdt_v1/publication_report/displacement_bin_summary.csv",
        ),
    ),
    FigureSpec(
        figure_id="figure_3",
        title="Corpus Workflow And Composition",
        module="evopoint_da.figures.corpus_workflow",
        step="figure_corpus_workflow",
        out_name="corpus_workflow",
        scientific_role="Dataset construction and tier-role figure.",
        primary_message=(
            "Gold, Silver, and Bronze have distinct evidence roles; only Gold is used for primary "
            "supervised displacement reporting."
        ),
        caveat=(
            "This figure should not be read as performance evidence. It documents corpus assembly and split design."
        ),
        expected_data_files=(
            "corpus_workflow_summary_values.csv",
            "corpus_workflow_gold_distribution_values.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/tbdt_mixed_manifest_download_gold_report.json",
            "artifacts/tbdt_v1/prepare_gold_training_manifest_report.json",
            "artifacts/tbdt_v1/build_silver_clean_real_graphs_report.json",
            "artifacts/tbdt_v1/download_silver_bronze_assets_report.json",
            "data/tbdt_gold_training_manifest.csv",
        ),
    ),
    FigureSpec(
        figure_id="figure_s0",
        title="Task Definition And Positive Coordinate Case",
        module="evopoint_da.figures.task_definition",
        step="figure_task_definition",
        out_name="task_definition_and_positive_case",
        scientific_role="Problem-definition and reusable structure-placeholder figure.",
        primary_message=(
            "Cooper-TBDT predicts local AFDB-to-experimental C-alpha displacement and evaluates it by "
            "biological region; the positive case exports aligned coordinates for a manual structural inset."
        ),
        caveat=(
            "The structure panel is intentionally a placeholder. The exported CA-only PDB files are for drawing "
            "an illustrative overlay, not for adding a new quantitative claim."
        ),
        expected_data_files=(
            "task_definition_positive_case_metric_values.csv",
            "case_btub_p06129_3m8d_a_summary.csv",
            "case_btub_p06129_3m8d_a_region_counts.csv",
            "case_btub_p06129_3m8d_a_region_centroids.csv",
            "case_btub_p06129_3m8d_a_afdb_aligned_ca.pdb",
            "case_btub_p06129_3m8d_a_experimental_target_aligned_ca.pdb",
            "case_btub_p06129_3m8d_a_cooper_seed404_prediction_aligned_ca.pdb",
        ),
        source_paths=(
            "artifacts/tbdt_v1/publication_report/primary_model_paired_delta_samples.csv",
            "data/processed_tbdt_gold_graphs/btub_p06129_3m8d_a.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_404_best-selection_test/btub_p06129_3m8d_a.pt",
        ),
    ),
    FigureSpec(
        figure_id="figure_4",
        title="Baseline Comparison",
        module="evopoint_da.figures.baseline_comparison",
        step="figure_baseline_comparison",
        out_name="baseline_comparison",
        scientific_role="Non-neural and neural baseline comparison.",
        primary_message=(
            "Template-transfer and frame-aware linear baselines do not solve the held-out coordinate endpoint; "
            "Cooper-TBDT gives the only positive evaluation-region MSE improvement among the listed methods."
        ),
        caveat=(
            "The single-model row is the primary neural result. The blend is a secondary validation-calibrated candidate."
        ),
        expected_data_files=("baseline_comparison_values.csv",),
        source_paths=(
            "artifacts/tbdt_v1/publication_report/coordinate_metrics_summary.csv",
            "artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv",
        ),
    ),
    FigureSpec(
        figure_id="figure_5",
        title="Critical Neural Ablation",
        module="evopoint_da.figures.critical_ablation",
        step="figure_critical_ablation",
        out_name="critical_ablation",
        scientific_role="Mechanistic constraint-setting ablation figure.",
        primary_message=(
            "Region weighting supports the intended endpoint, and the scaffold anchor prevents functional-region "
            "gains from being purchased by barrel-core deformation."
        ),
        caveat=(
            "This is not evidence that every conditioning channel is causally necessary; state and AFDB-confidence "
            "claims must remain cautious."
        ),
        expected_data_files=(
            "critical_ablation_region_values.csv",
            "critical_ablation_paired_counts.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/critical_neural_ablation/critical_region_seed_summary.csv",
            "artifacts/tbdt_v1/critical_neural_ablation/critical_seed_mean_paired_summary.csv",
        ),
    ),
    FigureSpec(
        figure_id="figure_6",
        title="TonB-Box Mechanism Boundary",
        module="evopoint_da.figures.tonb_mechanistic_boundary",
        step="figure_tonb_boundary",
        out_name="tonb_mechanistic_boundary",
        scientific_role="Negative result and mechanistic-boundary figure.",
        primary_message=(
            "TonB coordinate RMSD has weak positive signal, but centroid direction and exposure-state compatibility "
            "are not solved."
        ),
        caveat=(
            "This figure should be framed as a boundary/limitation, not as a TonB mechanism recovery claim."
        ),
        expected_data_files=(
            "tonb_state_values.csv",
            "tonb_state_summary_values.csv",
            "case_btub_p06129_2gsk_a_summary.csv",
            "case_btub_p06129_2gsk_a_tonb_ca_coordinates.csv",
            "case_btub_p06129_2gsk_a_centroids.csv",
            "case_btub_p06129_2gsk_a_afdb_aligned_ca.pdb",
            "case_btub_p06129_2gsk_a_experimental_target_aligned_ca.pdb",
            "case_btub_p06129_2gsk_a_cooper_prediction_aligned_ca.pdb",
        ),
        source_paths=(
            "artifacts/tbdt_v1/mechanistic_eval/scaffold_blend_tonb_state_metrics.csv",
            "data/processed_tbdt_gold_graphs/btub_p06129_2gsk_a.pt",
            "artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test/btub_p06129_2gsk_a.pt",
        ),
    ),
    FigureSpec(
        figure_id="figure_s1",
        title="Residue-Shift Localization ROC/PR",
        module="evopoint_da.figures.residue_shift_localization",
        step="figure_residue_shift_localization",
        out_name="residue_shift_localization",
        scientific_role="Supplemental score-only residue-localization figure.",
        primary_message=(
            "ROC/PR curves test whether scores rank residues with >=1 Å experimental displacement; this is a "
            "localization endpoint and is separate from coordinate RMSD. The panel includes all external score "
            "baselines plus the Cooper validation-calibrated blend."
        ),
        caveat=(
            "AF2 low pLDDT and ANM are strong localization baselines. TonB PR curves are class-imbalanced and "
            "should not be used as evidence that TonB coordinate state is solved."
        ),
        expected_data_files=(
            "residue_shift_localization_metric_values.csv",
            "residue_shift_localization_seed_metric_values.csv",
            "residue_shift_localization_curve_points.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/external_baseline_curves/classification_curve_summary.csv",
            "artifacts/tbdt_v1/external_baseline_curves/classification_curve_points.csv",
            "artifacts/tbdt_v1/test_graph_files.txt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_42_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_101_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_202_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_303_best-selection_test/*.pt",
            "artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_404_best-selection_test/*.pt",
        ),
    ),
    FigureSpec(
        figure_id="figure_s2",
        title="Seed Stability And Selector Sensitivity",
        module="evopoint_da.figures.seed_stability_selector",
        step="figure_seed_stability_selector",
        out_name="seed_stability_selector_sensitivity",
        scientific_role="Supplemental robustness and checkpoint-selection figure.",
        primary_message=(
            "The primary best-selection model family is stable across five seeds, while selector choices mainly "
            "document endpoint sensitivity rather than providing a new model-selection claim."
        ),
        caveat=(
            "Flex-biased selector behavior is sensitivity evidence only. The article-facing primary result remains "
            "the predeclared best-selection family."
        ),
        expected_data_files=(
            "seed_stability_by_seed_values.csv",
            "seed_stability_selector_values.csv",
        ),
        source_paths=(
            "artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_summary.csv",
            "artifacts/tbdt_v1/publication_report/selector_sensitivity_summary.csv",
        ),
    ),
)


def specs_by_out_name() -> dict[str, FigureSpec]:
    return {spec.out_name: spec for spec in FIGURE_SPECS}


def _path_exists(pattern: str) -> bool:
    path = Path(pattern)
    if any(char in pattern for char in "*?[]"):
        return bool(list(path.parent.glob(path.name)))
    return path.exists()


def _missing_sources(spec: FigureSpec) -> list[str]:
    return [path for path in spec.source_paths if not _path_exists(path)]


def _missing_outputs(root: Path, spec: FigureSpec, formats: list[str]) -> list[str]:
    fig_dir = root / spec.out_name
    missing: list[str] = []
    for fmt in formats:
        rendered = fig_dir / f"{spec.out_name}.{fmt}"
        if not rendered.exists():
            missing.append(str(rendered))
    for filename in spec.expected_data_files:
        data_path = fig_dir / filename
        if not data_path.exists():
            missing.append(str(data_path))
    return missing


def write_suite_manifest(root: str | Path, formats: list[str]) -> Path:
    out_root = Path(root)
    rows = []
    for spec in FIGURE_SPECS:
        figure_dir = out_root / spec.out_name
        rendered = [str(figure_dir / f"{spec.out_name}.{fmt}") for fmt in formats]
        rows.append(
            {
                "figure_id": spec.figure_id,
                "title": spec.title,
                "step": spec.step,
                "module": spec.module,
                "out_name": spec.out_name,
                "figure_dir": str(figure_dir),
                "rendered_files": ";".join(rendered),
                "data_files": ";".join(str(figure_dir / name) for name in spec.expected_data_files),
                "scientific_role": spec.scientific_role,
                "primary_message": spec.primary_message,
                "caveat": spec.caveat,
            }
        )
    out_path = out_root / "figure_suite_manifest.csv"
    write_csv_rows(
        out_path,
        rows,
        [
            "figure_id",
            "title",
            "step",
            "module",
            "out_name",
            "figure_dir",
            "rendered_files",
            "data_files",
            "scientific_role",
            "primary_message",
            "caveat",
        ],
    )
    return out_path


def write_scientific_audit(root: str | Path, formats: list[str]) -> Path:
    out_root = Path(root)
    rows = []
    for spec in FIGURE_SPECS:
        missing_sources = _missing_sources(spec)
        missing_outputs = _missing_outputs(out_root, spec, formats)
        status = "pass" if not missing_sources and not missing_outputs else "needs_attention"
        scientific_status = "fit_for_stated_claim"
        if spec.figure_id in {"figure_4", "figure_5", "figure_6", "figure_s1", "figure_s2"}:
            scientific_status = "fit_for_stated_claim_with_caveat"
        rows.append(
            {
                "figure_id": spec.figure_id,
                "title": spec.title,
                "technical_status": status,
                "scientific_status": scientific_status,
                "missing_sources": ";".join(missing_sources),
                "missing_outputs": ";".join(missing_outputs),
                "scientific_role": spec.scientific_role,
                "primary_message": spec.primary_message,
                "caveat": spec.caveat,
            }
        )

    csv_path = out_root / "figure_scientific_audit.csv"
    write_csv_rows(
        csv_path,
        rows,
        [
            "figure_id",
            "title",
            "technical_status",
            "scientific_status",
            "missing_sources",
            "missing_outputs",
            "scientific_role",
            "primary_message",
            "caveat",
        ],
    )

    md_path = out_root / "figure_scientific_audit.md"
    lines = [
        "# Cooper-TBDT Figure Scientific Audit",
        "",
        "This audit records the scientific contract for each generated figure. A `fit_for_stated_claim` status means the figure is appropriate for the claim listed here, not that the underlying result is broadly solved.",
        "",
        "| Figure | Technical status | Scientific status | Claim boundary |",
        "|---|---|---|---|",
    ]
    for row in rows:
        boundary = f"{row['primary_message']} Caveat: {row['caveat']}"
        lines.append(
            f"| {row['figure_id']}: {row['title']} | {row['technical_status']} | "
            f"{row['scientific_status']} | {boundary} |"
        )
    lines.append("")
    lines.append("Companion CSV: `figure_scientific_audit.csv`.")
    md_path.write_text("\n".join(lines) + "\n")
    return md_path
