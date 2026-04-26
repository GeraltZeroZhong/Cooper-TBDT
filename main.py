"""Unified pipeline orchestrator.

This entrypoint provides a single CLI to run one pipeline stage or execute the
full sequence end-to-end.
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


PIPELINE_STEPS: list[StepSpec] = [
    StepSpec("build_pairs", "evopoint_da.pipeline.build_displacement_dataset"),
    StepSpec("build_features", "evopoint_da.pipeline.build_features_with_sasa"),
    StepSpec("train", "train"),
    StepSpec("calibrate", "evopoint_da.pipeline.eval_run"),
    StepSpec("predict", "evopoint_da.pipeline.predict"),
]


def _run_module(module: str, extra_args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print("[pipeline] " + " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified entrypoint for EvoPoint pipeline")
    p.add_argument(
        "--step",
        choices=["full", *[s.name for s in PIPELINE_STEPS]],
        default="full",
        help="Run one stage or run the full pipeline in order",
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

    if args.step == "full":
        if forwarded:
            raise ValueError("`--step full` does not support forwarded args. Run per-step for custom params.")
        for step in PIPELINE_STEPS:
            _run_module(step.module, [], args.dry_run)
        return

    selected = next(spec for spec in PIPELINE_STEPS if spec.name == args.step)
    _run_module(selected.module, forwarded, args.dry_run)


if __name__ == "__main__":
    main()
