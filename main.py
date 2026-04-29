"""Thin CLI aliases for reproducible Cooper-TBDT pipeline modules.

Each stage still owns its own arguments. This file only maps a stable stage
name to ``python -m ...`` so commands used in docs and reports stay short.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass


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
)


def _run_module(module: str, extra_args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print("[pipeline] " + " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a named Cooper-TBDT pipeline module")
    p.add_argument(
        "--step",
        choices=[s.name for s in PIPELINE_STEPS],
        default="publication_report",
        help="Pipeline stage alias to run.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument(
        "step_args",
        nargs=argparse.REMAINDER,
        help=(
            "Args forwarded to target module after '--'. Example: "
            "python main.py --step build_pairs -- --af2_dir data/raw_af2"
        ),
    )
    return p.parse_args()


def _normalize_step_args(step_args: list[str]) -> list[str]:
    if step_args and step_args[0] == "--":
        return step_args[1:]
    return step_args


def main() -> None:
    args = parse_args()
    forwarded = _normalize_step_args(args.step_args)

    selected = next(spec for spec in PIPELINE_STEPS if spec.name == args.step)
    _run_module(selected.module, forwarded, args.dry_run)


if __name__ == "__main__":
    main()
