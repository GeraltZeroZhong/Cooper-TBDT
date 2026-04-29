"""CLI wrapper for TBDT state-displacement region evaluation."""

from __future__ import annotations

import argparse
import json

from evopoint_da.pipeline.tbdt_state_eval_core import (
    BARREL_CORE_KEYS,
    DIRECT_REGION_MASK_KEYS,
    PREDICTION_KEYS,
    REGION_CONTAINER_KEYS,
    TARGET_KEYS,
    _as_mapping,
    _extract_prediction,
    _extract_regions,
    _extract_target,
    _load_pt,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TBDT per-region displacement metrics from processed .pt samples. "
            "When predictions are omitted, evaluate the raw AF2 / zero-displacement baseline."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more processed .pt files or directories containing processed .pt samples.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Optional prediction .pt file or directory. A directory is matched to samples by file stem. "
            "A single prediction file may be used with one input sample or may contain a dict keyed by sample stem."
        ),
    )
    parser.add_argument("--output-json", default="artifacts/tbdt_state_region_metrics.json")
    parser.add_argument("--output-csv", default=None, help="Optional flat CSV metrics output.")
    parser.add_argument(
        "--region-json",
        default=None,
        help=(
            "Optional JSON mapping of sample stem to region masks/indices/residue_ids. "
            "Use key '*' for regions shared by all samples."
        ),
    )
    parser.add_argument(
        "--include-all-region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an 'all' region row for diagnostics. Primary TBDT reporting should use named regions.",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=1.0,
        help="Only residues with target displacement at least this many Angstrom contribute to direction cosine.",
    )
    parser.add_argument(
        "--add-derived-regions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add derived diagnostic regions when explicit masks are absent. "
            "Currently derives plug_core, plug_apical_loop, and plug_extension_nt from plug/TonB masks."
        ),
    )
    parser.add_argument(
        "--plug-apical-fraction",
        type=float,
        default=0.35,
        help="Fraction of non-TonB plug residues assigned to the sequence-C-terminal plug_apical_loop heuristic.",
    )
    parser.add_argument(
        "--plug-extension-residues",
        type=int,
        default=12,
        help="Number of sequence-N-terminal plug residues assigned to plug_extension_nt when no explicit mask exists.",
    )
    parser.add_argument(
        "--bootstrap-iter",
        type=int,
        default=5000,
        help="Bootstrap iterations for per-target paired delta confidence intervals.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--paired-delta-csv",
        default=None,
        help="Optional CSV of per-target paired delta rows: Delta RMSD = method RMSD - raw AF2 RMSD.",
    )
    parser.add_argument(
        "--tonb-metrics-csv",
        default=None,
        help="Optional CSV of per-target TonB-box geometric/state metrics.",
    )
    parser.add_argument(
        "--tonb-exposure-threshold",
        type=float,
        default=1.0,
        help="Angstrom threshold for TonB exposed-like/buried-like classification from centroid exposure delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print(json.dumps({"output_json": args.output_json, "n_samples": report["n_samples"]}, indent=2))


if __name__ == "__main__":
    main()
