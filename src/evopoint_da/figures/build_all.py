"""Build and audit the full Cooper-TBDT figure suite."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from evopoint_da.figures.style import parse_formats
from evopoint_da.figures.suite import FIGURE_SPECS, FigureSpec, write_scientific_audit, write_suite_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all Cooper-TBDT publication/report figures")
    parser.add_argument(
        "--out-dir",
        default="artifacts/tbdt_v1/figures",
        help="Root output directory for all figure subdirectories and suite-level manifests.",
    )
    parser.add_argument("--style", default="cooper", help="Named style from evopoint_da.figures.style.")
    parser.add_argument(
        "--formats",
        default="png,svg,pdf",
        help="Comma-separated output formats. The suite default includes PDF for publication layout.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Optional comma-separated subset by figure_id or out_name, e.g. "
            "figure_1,baseline_comparison."
        ),
    )
    parser.add_argument("--skip-build", action="store_true", help="Only refresh suite manifest and audit files.")
    parser.add_argument("--dry-run", action="store_true", help="Print figure commands without running them.")
    return parser.parse_args()


def _selected_specs(only: str) -> list[FigureSpec]:
    tokens = {item.strip() for item in only.split(",") if item.strip()}
    if not tokens:
        return list(FIGURE_SPECS)
    selected = [
        spec
        for spec in FIGURE_SPECS
        if spec.figure_id in tokens or spec.out_name in tokens or spec.step in tokens
    ]
    missing = sorted(tokens - {spec.figure_id for spec in selected} - {spec.out_name for spec in selected} - {spec.step for spec in selected})
    if missing:
        raise ValueError(f"Unknown figure selector(s): {', '.join(missing)}")
    return selected


def _run_spec(spec: FigureSpec, args: argparse.Namespace, formats: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        spec.module,
        "--out-dir",
        args.out_dir,
        "--style",
        args.style,
        "--formats",
        ",".join(formats),
        "--dpi",
        str(args.dpi),
    ]
    print("[figure-suite] " + " ".join(shlex.quote(part) for part in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_specs(args.only)
    if not args.skip_build:
        for spec in selected:
            _run_spec(spec, args, formats)

    manifest = write_suite_manifest(out_root, formats)
    audit = write_scientific_audit(out_root, formats)
    print(manifest)
    print(audit)


if __name__ == "__main__":
    main()
