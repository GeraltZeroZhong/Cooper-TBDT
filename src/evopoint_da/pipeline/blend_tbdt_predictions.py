"""Blend TBDT displacement predictions with region-specific sources.

This utility is intentionally post-hoc: it composes already exported
per-residue displacement predictions. It is useful for TBDT state correction
because different checkpoints can specialize in different regions, such as a
plug-focused model and a TonB-box-focused model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evopoint_da.data.dataset import build_split_file_lists
from evopoint_da.pipeline.eval_tbdt_state import (
    _as_mapping,
    _extract_prediction,
    _extract_regions,
    _extract_target,
    _load_pt,
)


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}
DEFAULT_PRIORITY = ["tonb_box", "substrate_contact", "plug", "extracellular_loop", "eval", "barrel_core"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend TBDT graph predictions by structural region.")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--base-predictions", required=True, help="Prediction directory used for all residues by default.")
    parser.add_argument("--base-scale", type=float, default=1.0)
    parser.add_argument(
        "--region-source",
        action="append",
        default=[],
        metavar="REGION=DIR",
        help="Prediction directory for a named region. May be repeated.",
    )
    parser.add_argument(
        "--region-scale",
        action="append",
        default=[],
        metavar="REGION=SCALE",
        help="Explicit multiplier for a named region. May be repeated.",
    )
    parser.add_argument(
        "--auto-scale-region",
        action="append",
        default=[],
        metavar="REGION",
        help="Fit a non-negative scalar for this region on --calibration-split.",
    )
    parser.add_argument("--calibration-split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--calibration-region-source",
        action="append",
        default=[],
        metavar="REGION=DIR",
        help="Prediction directory for fitting REGION scale on the calibration split.",
    )
    parser.add_argument("--max-scale", type=float, default=12.0)
    parser.add_argument("--min-calibration-residues", type=int, default=100)
    parser.add_argument(
        "--priority",
        default=",".join(DEFAULT_PRIORITY),
        help="Comma-separated region override order. Earlier regions win overlaps.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _parse_mapping(items: list[str], *, value_type: type = str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected REGION=VALUE item, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected non-empty REGION=VALUE item, got: {item!r}")
        parsed[key] = value_type(value)
    return parsed


def _split_files(args: argparse.Namespace, split: str) -> list[str]:
    splits = build_split_file_lists(
        root=args.data_dir,
        split_ranges=DEFAULT_SPLIT_RANGES,
        split_seed=args.split_seed,
        split_source=args.split_source,
    )
    return splits.get(split, [])


def _load_prediction(prediction_dir: Path, stem: str, n: int) -> torch.Tensor:
    prediction_path = prediction_dir / f"{stem}.pt"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction for {stem}: {prediction_path}")
    return _extract_prediction(_load_pt(prediction_path), stem, n)


def _fit_scale(
    sample_files: list[str],
    *,
    region_name: str,
    prediction_dir: Path,
    max_scale: float,
) -> dict[str, float | int | str]:
    numerator = 0.0
    denominator = 0.0
    n_residues = 0
    n_samples = 0
    for raw_path in sample_files:
        sample_path = Path(raw_path)
        sample = _as_mapping(_load_pt(sample_path))
        target = _extract_target(sample, sample_path)
        n = int(target.size(0))
        prediction = _load_prediction(prediction_dir, sample_path.stem, n)
        regions = _extract_regions(sample, sample_path.stem, n, {}, True)
        mask = torch.ones(n, dtype=torch.bool) if region_name == "all" else regions.get(region_name)
        if mask is None or not bool(mask.any()):
            continue

        t = target[mask]
        p = prediction[mask]
        numerator += float((p * t).sum().item())
        denominator += float((p * p).sum().item())
        n_residues += int(mask.sum().item())
        n_samples += 1

    raw_scale = 0.0 if denominator <= 1e-12 else numerator / denominator
    clipped_scale = max(0.0, min(float(max_scale), raw_scale))
    return {
        "scale": clipped_scale,
        "raw_scale": raw_scale,
        "n_residues": n_residues,
        "n_samples": n_samples,
        "numerator": numerator,
        "denominator": denominator,
        "status": "fit",
    }


def main() -> None:
    args = parse_args()
    region_sources = {key: Path(value) for key, value in _parse_mapping(args.region_source).items()}
    explicit_region_scales = _parse_mapping(args.region_scale, value_type=float)
    calibration_region_sources = {
        key: Path(value) for key, value in _parse_mapping(args.calibration_region_source).items()
    }
    auto_scale_regions = {str(region) for region in args.auto_scale_region}
    priority = [item.strip() for item in str(args.priority).split(",") if item.strip()]
    priority.extend(region for region in region_sources if region not in priority)

    base_dir = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_files = _split_files(args, args.calibration_split)
    region_scale_reports: dict[str, dict[str, Any]] = {}
    region_scales: dict[str, float] = {}
    for region_name in region_sources:
        if region_name in explicit_region_scales:
            region_scales[region_name] = float(explicit_region_scales[region_name])
            region_scale_reports[region_name] = {"scale": region_scales[region_name], "status": "explicit"}
            continue
        if region_name not in auto_scale_regions:
            region_scales[region_name] = 1.0
            region_scale_reports[region_name] = {"scale": 1.0, "status": "identity"}
            continue

        calibration_dir = calibration_region_sources.get(region_name, region_sources[region_name])
        report = _fit_scale(
            calibration_files,
            region_name=region_name,
            prediction_dir=calibration_dir,
            max_scale=float(args.max_scale),
        )
        if int(report["n_residues"]) < int(args.min_calibration_residues):
            report["status"] = "identity_too_few_calibration_residues"
            report["scale"] = 1.0
        region_scales[region_name] = float(report["scale"])
        region_scale_reports[region_name] = report

    output_files = []
    for raw_path in _split_files(args, args.split):
        sample_path = Path(raw_path)
        sample = _as_mapping(_load_pt(sample_path))
        target = _extract_target(sample, sample_path)
        n = int(target.size(0))
        regions = _extract_regions(sample, sample_path.stem, n, {}, True)

        blended = _load_prediction(base_dir, sample_path.stem, n) * float(args.base_scale)
        assigned = torch.zeros(n, dtype=torch.bool)
        applied_regions: list[str] = []
        for region_name in priority:
            if region_name not in region_sources:
                continue
            mask = regions.get(region_name)
            if mask is None:
                continue
            mask = mask & ~assigned
            if not bool(mask.any()):
                continue
            region_prediction = _load_prediction(region_sources[region_name], sample_path.stem, n)
            blended[mask] = region_prediction[mask] * float(region_scales[region_name])
            assigned[mask] = True
            applied_regions.append(region_name)

        output_path = output_dir / f"{sample_path.stem}.pt"
        torch.save(
            {
                "pair_id": sample_path.stem,
                "pred_delta": blended,
                "metadata": {
                    "base_predictions": str(base_dir),
                    "base_scale": float(args.base_scale),
                    "applied_regions": applied_regions,
                    "region_scales": region_scales,
                },
            },
            output_path,
        )
        output_files.append(str(output_path))

    report = {
        "data_dir": args.data_dir,
        "split": args.split,
        "split_source": args.split_source,
        "base_predictions": str(base_dir),
        "base_scale": float(args.base_scale),
        "region_sources": {region: str(path) for region, path in region_sources.items()},
        "region_scale_reports": region_scale_reports,
        "priority": priority,
        "output_dir": str(output_dir),
        "n_predictions": len(output_files),
        "output_files": output_files,
    }
    report_path = Path(args.report_path) if args.report_path else output_dir.with_name(f"{output_dir.name}_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"report_path": str(report_path), "n_predictions": len(output_files)}, indent=2))


if __name__ == "__main__":
    main()
