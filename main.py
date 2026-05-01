"""CLI aliases and workflows for reproducible Cooper-TBDT pipeline modules.

Each stage still owns its own arguments. This file maps stable stage names and
common multi-stage workflows to ``python -m ...`` commands so docs and reports
stay short.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class StepSpec:
    name: str
    module: str


PIPELINE_STEPS: tuple[StepSpec, ...] = (
    StepSpec("fetch_seed_structures", "evopoint_da.pipeline.fetch_tbdt_structures"),
    StepSpec("build_mixed_manifest", "evopoint_da.pipeline.build_tbdt_mixed_manifest"),
    StepSpec("download_assets", "evopoint_da.pipeline.download_tbdt_manifest_assets"),
    StepSpec("prepare_training_manifest", "evopoint_da.pipeline.prepare_tbdt_training_manifest"),
    StepSpec("build_pairs", "evopoint_da.pipeline.build_tbdt_state_dataset"),
    StepSpec("build_features", "evopoint_da.pipeline.build_features_with_sasa"),
    StepSpec("train", "train"),
    StepSpec("predict_graphs", "evopoint_da.pipeline.predict_tbdt_graphs"),
    StepSpec("blend_predictions", "evopoint_da.pipeline.blend_tbdt_predictions"),
    StepSpec("eval_regions", "evopoint_da.pipeline.eval_tbdt_state"),
    StepSpec("template_baselines", "evopoint_da.pipeline.build_tbdt_template_baselines"),
    StepSpec("structure_template_baselines", "evopoint_da.pipeline.build_tbdt_structure_template_baselines"),
    StepSpec("coordinate_baselines", "evopoint_da.pipeline.build_tbdt_coordinate_baselines"),
    StepSpec("external_baselines", "evopoint_da.pipeline.build_tbdt_external_baselines"),
    StepSpec("eval_classification", "evopoint_da.pipeline.eval_tbdt_classification_curves"),
    StepSpec("seed_stability", "evopoint_da.pipeline.run_tbdt_seed_stability"),
    StepSpec("report_models", "evopoint_da.pipeline.run_tbdt_report_models"),
    StepSpec("publication_report", "evopoint_da.pipeline.build_tbdt_publication_report"),
    StepSpec("prepare_docking_manifest", "evopoint_da.pipeline.prepare_tbdt_docking_manifest"),
    StepSpec("docking_eval", "evopoint_da.docking_eval.pipeline_cli"),
    StepSpec("figure_main_results", "evopoint_da.figures.main_results"),
    StepSpec("figure_gold_test_displacement", "evopoint_da.figures.gold_test_displacement_landscape"),
    StepSpec("figure_corpus_workflow", "evopoint_da.figures.corpus_workflow"),
    StepSpec("figure_task_definition", "evopoint_da.figures.task_definition"),
    StepSpec("figure_baseline_comparison", "evopoint_da.figures.baseline_comparison"),
    StepSpec("figure_critical_ablation", "evopoint_da.figures.critical_ablation"),
    StepSpec("figure_tonb_boundary", "evopoint_da.figures.tonb_mechanistic_boundary"),
    StepSpec("figure_residue_shift_localization", "evopoint_da.figures.residue_shift_localization"),
    StepSpec("figure_seed_stability_selector", "evopoint_da.figures.seed_stability_selector"),
    StepSpec("figure_all", "evopoint_da.figures.build_all"),
    StepSpec("figure_panel_archive", "evopoint_da.figures.panel_archive"),
)

WORKFLOWS: dict[str, str] = {
    "download_benchmark": "Download benchmark structures and AFDB/PAE assets from the versioned manifests.",
    "baseline_predict": "Run prediction with the provided Cooper-TBDT baseline checkpoint.",
    "reproduce_training": "Build Gold data, train the baseline model, export predictions, and evaluate regions.",
}

DEFAULT_BASELINE_CHECKPOINT = Path("checkpoints/cooper_tbdt_baseline/best-selection-seed404.ckpt")
DEFAULT_BASELINE_CHECKPOINT_URL = (
    "https://github.com/GeraltZeroZhong/Cooper-TBDT/releases/download/v0.1.0/"
    "cooper_tbdt_baseline_seed404_best-selection.ckpt"
)
DEFAULT_BASELINE_CHECKPOINT_SHA256 = "cf7515a8c1634b7a365696d807d03a37ea6fdd483260b4e8b52aa7c7c6daf891"


def _run_module(module: str, extra_args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print("[pipeline] " + " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _workflow_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python main.py --workflow {name} --",
        description=WORKFLOWS[name],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if name == "download_benchmark":
        parser.add_argument("--manifest", default="data/tbdt_mixed_manifest.csv")
        parser.add_argument(
            "--tier",
            action="append",
            choices=["gold", "silver", "bronze"],
            help="Tier to download. Repeat to select multiple tiers. Defaults to all tiers.",
        )
        parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
        parser.add_argument("--raw-af2-dir", default="data/raw_af2")
        parser.add_argument("--af-version", type=int, default=6)
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--out-manifest", default=None)
        parser.add_argument("--report-path", default="artifacts/tbdt_v1/download_tbdt_manifest_assets_report.json")
        parser.add_argument("--overwrite", action="store_true")
        parser.set_defaults(download_pae=True, sync_tier_manifests=True)
        parser.add_argument("--download-pae", dest="download_pae", action="store_true")
        parser.add_argument("--no-download-pae", dest="download_pae", action="store_false")
        parser.add_argument("--sync-tier-manifests", dest="sync_tier_manifests", action="store_true")
        parser.add_argument("--no-sync-tier-manifests", dest="sync_tier_manifests", action="store_false")
        return parser

    if name == "baseline_predict":
        parser.add_argument(
            "--ckpt",
            default=str(DEFAULT_BASELINE_CHECKPOINT),
            help="Local path for the provided Cooper-TBDT baseline checkpoint.",
        )
        parser.add_argument(
            "--checkpoint-url",
            default=os.environ.get("COOPER_TBDT_BASELINE_CHECKPOINT_URL", DEFAULT_BASELINE_CHECKPOINT_URL),
            help="URL used to download the provided baseline checkpoint when --ckpt is missing.",
        )
        parser.add_argument(
            "--checkpoint-sha256",
            default=os.environ.get("COOPER_TBDT_BASELINE_CHECKPOINT_SHA256"),
            help="Optional SHA256 expected for the downloaded checkpoint. Defaults to the published baseline hash.",
        )
        parser.add_argument(
            "--download-checkpoint",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Download the provided checkpoint if --ckpt is missing.",
        )
        parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
        parser.add_argument("--output-dir", default="artifacts/tbdt_v1/predictions/scaffold_prior_test")
        parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
        parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
        parser.add_argument("--split-seed", type=int, default=42)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--device", default=None, help="Prediction device. Omit to let the prediction module choose.")
        parser.add_argument("--report-path", default="artifacts/tbdt_v1/predictions/scaffold_prior_test_report.json")
        return parser

    if name == "reproduce_training":
        parser.add_argument("--asset-manifest", default="data/tbdt_mixed_manifest.csv")
        parser.add_argument("--training-manifest", default="data/tbdt_gold_training_manifest.csv")
        parser.add_argument("--pair-dir", default="data/processed_tbdt_gold_pairs")
        parser.add_argument("--graph-dir", default="data/processed_tbdt_gold_graphs")
        parser.add_argument("--raw-pdb-dir", default="data/raw_pdb")
        parser.add_argument("--raw-af2-dir", default="data/raw_af2")
        parser.add_argument("--af-version", type=int, default=6)
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--esm-weights", default="esmc_weights/esmc_600m_2024_12_v0.pth")
        parser.add_argument("--pca-path", default="data/pca_esmc_128.pkl")
        parser.add_argument("--study-name", default="tbdt_gold_scaffold_prior")
        parser.add_argument("--max-epochs", type=int, default=40)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--checkpoint-selector",
            default="best-selection",
            choices=["best-selection", "best-disp1to2", "best-disp1to5", "best-flex", "last"],
        )
        parser.add_argument("--ckpt", default=None, help="Use this checkpoint for prediction/evaluation instead of resolving the latest trained checkpoint.")
        parser.add_argument("--prediction-dir", default="artifacts/tbdt_v1/predictions/scaffold_prior_test")
        parser.add_argument("--artifact-dir", default="artifacts/tbdt_v1")
        parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
        parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
        parser.add_argument("--split-seed", type=int, default=42)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--device", default=None, help="Prediction device. Omit to let the prediction module choose.")
        parser.add_argument("--accelerator", default=None, help="Hydra trainer.accelerator override, e.g. cpu or gpu.")
        parser.add_argument("--devices", default=None, help="Hydra trainer.devices override.")
        parser.add_argument("--smoke-test-features", action="store_true")
        parser.add_argument("--allow-missing-pae", action="store_true")
        parser.add_argument("--skip-download", action="store_true")
        parser.add_argument("--skip-build-pairs", action="store_true")
        parser.add_argument("--skip-build-features", action="store_true")
        parser.add_argument("--skip-train", action="store_true")
        parser.add_argument("--skip-predict", action="store_true")
        parser.add_argument("--skip-eval", action="store_true")
        parser.add_argument("--skip-zero-eval", action="store_true")
        parser.add_argument("--skip-post-train-tests", action="store_true")
        parser.add_argument(
            "--train-override",
            action="append",
            default=[],
            help="Additional Hydra override forwarded to train.py. Repeat for multiple overrides.",
        )
        return parser

    raise ValueError(f"Unsupported workflow: {name}")


def _download_benchmark_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--manifest",
        str(args.manifest),
        "--raw-pdb-dir",
        str(args.raw_pdb_dir),
        "--raw-af2-dir",
        str(args.raw_af2_dir),
        "--af-version",
        str(args.af_version),
        "--workers",
        str(args.workers),
        "--report-path",
        str(args.report_path),
    ]
    if args.out_manifest:
        cmd.extend(["--out-manifest", str(args.out_manifest)])
    for tier in args.tier or ["gold", "silver", "bronze"]:
        cmd.extend(["--tier", tier])
    if args.download_pae:
        cmd.append("--download-pae")
    if args.sync_tier_manifests:
        cmd.append("--sync-tier-manifests")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _prediction_args(args: argparse.Namespace, ckpt: str) -> list[str]:
    report_path = getattr(args, "report_path", None)
    if report_path is None:
        report_path = Path(args.artifact_dir) / "predict_tbdt_graphs_report.json"
    cmd = [
        "--ckpt",
        str(ckpt),
        "--data-dir",
        str(args.data_dir if hasattr(args, "data_dir") else args.graph_dir),
        "--output-dir",
        str(args.output_dir if hasattr(args, "output_dir") else args.prediction_dir),
        "--split",
        str(args.split),
        "--split-source",
        str(args.split_source),
        "--split-seed",
        str(args.split_seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--report-path",
        str(report_path),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    return cmd


def _checkpoint_sha256(args: argparse.Namespace) -> str | None:
    value = getattr(args, "checkpoint_sha256", None)
    if value:
        return str(value)
    if getattr(args, "checkpoint_url", None) == DEFAULT_BASELINE_CHECKPOINT_URL:
        return DEFAULT_BASELINE_CHECKPOINT_SHA256
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual = _file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}. "
            "Delete the file and rerun, or pass --checkpoint-sha256 for a custom checkpoint."
        )


def _download_checkpoint(url: str, output_path: Path, expected_sha256: str | None, dry_run: bool) -> None:
    print(f"[pipeline] download {shlex.quote(url)} -> {shlex.quote(str(output_path))}")
    if dry_run:
        return
    if not url:
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {output_path}. "
            "Pass --checkpoint-url, set COOPER_TBDT_BASELINE_CHECKPOINT_URL, or pass --ckpt."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urlopen(url) as response, tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        _verify_checkpoint(tmp_path, expected_sha256)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _ensure_baseline_checkpoint(args: argparse.Namespace, dry_run: bool) -> str:
    ckpt = Path(args.ckpt)
    expected_sha256 = _checkpoint_sha256(args)
    if ckpt.exists():
        _verify_checkpoint(ckpt, expected_sha256)
        return str(ckpt)
    if not args.download_checkpoint:
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {ckpt}. "
            "Rerun without --no-download-checkpoint, or pass --ckpt to an existing checkpoint."
        )
    _download_checkpoint(str(args.checkpoint_url), ckpt, expected_sha256, dry_run)
    return str(ckpt)


def _latest_checkpoint(study_name: str, selector: str) -> Path:
    root = Path("checkpoints") / study_name if study_name else Path("checkpoints")
    pattern = "last.ckpt" if selector == "last" else f"{selector}-*.ckpt"
    matches = sorted(
        root.glob(f"*/{pattern}"),
        key=lambda path: (path.parent.name, path.stat().st_mtime, path.name),
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {root}/*/{pattern}")
    return matches[-1]


def _run_download_benchmark(workflow_args: list[str], dry_run: bool) -> None:
    args = _workflow_parser("download_benchmark").parse_args(workflow_args)
    _run_module("evopoint_da.pipeline.download_tbdt_manifest_assets", _download_benchmark_args(args), dry_run)


def _run_baseline_predict(workflow_args: list[str], dry_run: bool) -> None:
    args = _workflow_parser("baseline_predict").parse_args(workflow_args)
    ckpt = _ensure_baseline_checkpoint(args, dry_run)
    _run_module("evopoint_da.pipeline.predict_tbdt_graphs", _prediction_args(args, ckpt), dry_run)


def _run_reproduce_training(workflow_args: list[str], dry_run: bool) -> None:
    args = _workflow_parser("reproduce_training").parse_args(workflow_args)
    artifact_dir = Path(args.artifact_dir)

    if not args.skip_download:
        download_args = argparse.Namespace(
            manifest=args.asset_manifest,
            tier=["gold"],
            raw_pdb_dir=args.raw_pdb_dir,
            raw_af2_dir=args.raw_af2_dir,
            af_version=args.af_version,
            workers=args.workers,
            out_manifest=None,
            report_path=str(artifact_dir / "tbdt_mixed_manifest_download_gold_report.json"),
            overwrite=False,
            download_pae=True,
            sync_tier_manifests=True,
        )
        _run_module("evopoint_da.pipeline.download_tbdt_manifest_assets", _download_benchmark_args(download_args), dry_run)

    if not args.skip_build_pairs:
        _run_module(
            "evopoint_da.pipeline.build_tbdt_state_dataset",
            [
                "--manifest",
                str(args.training_manifest),
                "--out_dir",
                str(args.pair_dir),
                "--report_path",
                str(artifact_dir / "build_gold_pairs_report.json"),
                "--require-core-alignment",
            ],
            dry_run,
        )

    if not args.skip_build_features:
        feature_args = [
            "--pair_dir",
            str(args.pair_dir),
            "--output_dir",
            str(args.graph_dir),
            "--pca_path",
            str(args.pca_path),
            "--pae_dir",
            str(args.raw_af2_dir),
            "--af2_structure_dir",
            str(args.raw_af2_dir),
            "--report_path",
            str(artifact_dir / "build_gold_graphs_report.json"),
        ]
        if args.smoke_test_features:
            feature_args.append("--smoke-test-features")
        else:
            feature_args.extend(["--esm_weights", str(args.esm_weights), "--fit_pca"])
        if args.allow_missing_pae:
            feature_args.append("--allow-missing-pae")
        _run_module("evopoint_da.pipeline.build_features_with_sasa", feature_args, dry_run)

    if not args.skip_train:
        train_args = [
            "data=tbdt_state",
            "model=gvp_tbdt_module",
            f"data.data_dir={args.graph_dir}",
            f"study_name={args.study_name}",
            f"trainer.max_epochs={args.max_epochs}",
            f"seed={args.seed}",
        ]
        if args.accelerator:
            train_args.append(f"trainer.accelerator={args.accelerator}")
        if args.devices:
            train_args.append(f"trainer.devices={args.devices}")
        if args.skip_post_train_tests:
            train_args.append("+run_post_train_tests=false")
        train_args.extend(str(item) for item in args.train_override)
        _run_module("train", train_args, dry_run)

    ckpt = args.ckpt
    if not ckpt and (not args.skip_predict or not args.skip_eval):
        if dry_run:
            pattern = "last.ckpt" if args.checkpoint_selector == "last" else f"{args.checkpoint_selector}-*.ckpt"
            ckpt = f"checkpoints/{args.study_name}/<latest-run>/{pattern}"
        else:
            ckpt = str(_latest_checkpoint(args.study_name, args.checkpoint_selector))

    if not args.skip_predict:
        _run_module("evopoint_da.pipeline.predict_tbdt_graphs", _prediction_args(args, str(ckpt)), dry_run)

    if not args.skip_eval and not args.skip_zero_eval:
        _run_module(
            "evopoint_da.pipeline.eval_tbdt_state",
            [
                str(args.graph_dir),
                "--output-json",
                str(artifact_dir / "gold_test_zero_region_metrics.json"),
                "--output-csv",
                str(artifact_dir / "gold_test_zero_region_metrics.csv"),
            ],
            dry_run,
        )

    if not args.skip_eval:
        _run_module(
            "evopoint_da.pipeline.eval_tbdt_state",
            [
                str(args.graph_dir),
                "--predictions",
                str(args.prediction_dir),
                "--output-json",
                str(artifact_dir / "gold_test_scaffold_prior_region_metrics.json"),
                "--output-csv",
                str(artifact_dir / "gold_test_scaffold_prior_region_metrics.csv"),
                "--paired-delta-csv",
                str(artifact_dir / "gold_test_scaffold_prior_paired_delta.csv"),
                "--tonb-metrics-csv",
                str(artifact_dir / "gold_test_scaffold_prior_tonb_metrics.csv"),
            ],
            dry_run,
        )


def _run_workflow(name: str, workflow_args: list[str], dry_run: bool) -> None:
    if name == "download_benchmark":
        _run_download_benchmark(workflow_args, dry_run)
    elif name == "baseline_predict":
        _run_baseline_predict(workflow_args, dry_run)
    elif name == "reproduce_training":
        _run_reproduce_training(workflow_args, dry_run)
    else:
        raise ValueError(f"Unsupported workflow: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a named Cooper-TBDT pipeline module")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--step",
        choices=[s.name for s in PIPELINE_STEPS],
        default=None,
        help="Pipeline stage alias to run. Defaults to publication_report when no workflow is selected.",
    )
    mode.add_argument(
        "--workflow",
        choices=sorted(WORKFLOWS),
        default=None,
        help="Higher-level workflow to run. Put workflow-specific options after '--'.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument(
        "step_args",
        nargs=argparse.REMAINDER,
        help=(
            "Args forwarded to the target stage or workflow after '--'. Example: "
            "python main.py --step build_pairs -- --manifest data/tbdt_gold_training_manifest.csv"
        ),
    )
    args = p.parse_args()
    if args.step is None and args.workflow is None:
        args.step = "publication_report"
    return args


def _normalize_step_args(step_args: list[str]) -> list[str]:
    if step_args and step_args[0] == "--":
        return step_args[1:]
    return step_args


def main() -> None:
    args = parse_args()
    forwarded = _normalize_step_args(args.step_args)

    if args.workflow:
        _run_workflow(args.workflow, forwarded, args.dry_run)
    else:
        selected = next(spec for spec in PIPELINE_STEPS if spec.name == args.step)
        _run_module(selected.module, forwarded, args.dry_run)


if __name__ == "__main__":
    main()
