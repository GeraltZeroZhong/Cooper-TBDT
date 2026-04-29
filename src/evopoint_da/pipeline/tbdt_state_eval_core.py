"""Core evaluation loop for TBDT state-displacement region metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from evopoint_da.pipeline.tbdt_state_eval_metrics import (
    _json_safe,
    _paired_delta_summary,
    _region_metrics,
    _rms_from_vectors,
    _sample_region_summary,
    _tonb_state_metrics,
    _tonb_summary,
    _write_csv,
    _write_generic_csv,
)
from evopoint_da.pipeline.tbdt_state_eval_utils import (
    BARREL_CORE_KEYS,
    DIRECT_REGION_MASK_KEYS,
    PREDICTION_KEYS,
    REGION_CONTAINER_KEYS,
    TARGET_KEYS,
    _as_mapping,
    _collect_pt_files,
    _derive_plug_regions,
    _extract_prediction,
    _extract_regions,
    _extract_target,
    _first_present,
    _load_pt,
    _load_region_json,
    _prediction_index,
    _vector_tensor,
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples = _collect_pt_files(args.inputs)
    pred_paths, shared_prediction = _prediction_index(args.predictions, samples)
    region_json = _load_region_json(args.region_json)
    direction_threshold = float(getattr(args, "direction_threshold", 1.0))
    add_derived_regions = bool(getattr(args, "add_derived_regions", True))
    plug_apical_fraction = float(getattr(args, "plug_apical_fraction", 0.35))
    plug_extension_residues = int(getattr(args, "plug_extension_residues", 12))
    bootstrap_iter = int(getattr(args, "bootstrap_iter", 5000))
    bootstrap_seed = int(getattr(args, "bootstrap_seed", 42))
    tonb_exposure_threshold = float(getattr(args, "tonb_exposure_threshold", 1.0))

    sample_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    tonb_rows: list[dict[str, Any]] = []
    aggregate_vectors: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"target": [], "prediction": []})

    for sample_path in samples:
        sample = _as_mapping(_load_pt(sample_path))
        target = _extract_target(sample, sample_path)
        n = int(target.size(0))

        pred_obj = shared_prediction
        if pred_obj is None and pred_paths:
            pred_path = pred_paths.get(sample_path.stem)
            if pred_path is None:
                raise FileNotFoundError(f"No prediction .pt found for sample stem: {sample_path.stem}")
            pred_obj = _load_pt(pred_path)
        prediction = _extract_prediction(pred_obj, sample_path.stem, n)

        regions = _extract_regions(sample, sample_path.stem, n, region_json, args.include_all_region)
        if add_derived_regions:
            _derive_plug_regions(
                regions,
                n,
                plug_apical_fraction=plug_apical_fraction,
                plug_extension_residues=plug_extension_residues,
            )
            regions = {name: mask for name, mask in regions.items() if bool(mask.any())}
        barrel_mask = regions.get("barrel_core")
        barrel_core_target_rms = _rms_from_vectors(target[barrel_mask]) if barrel_mask is not None else float("nan")

        region_blocks = {}
        for region_name, mask in sorted(regions.items()):
            metrics = _region_metrics(
                target,
                prediction,
                mask,
                direction_threshold=direction_threshold,
            )
            metrics["barrel_core_target_rms"] = barrel_core_target_rms
            region_blocks[region_name] = metrics
            aggregate_vectors[region_name]["target"].append(target[mask])
            aggregate_vectors[region_name]["prediction"].append(prediction[mask])
            csv_rows.append(
                {
                    "scope": "sample",
                    "sample_id": sample_path.stem,
                    "region": region_name,
                    **metrics,
                }
            )

        pos_value = _first_present(sample, ("pos", "af2_pos"))
        pos = _vector_tensor(pos_value, f"pos in {sample_path}") if pos_value is not None else None
        tonb_metrics = _tonb_state_metrics(
            pos=pos,
            target=target,
            prediction=prediction,
            regions=regions,
            exposure_threshold=tonb_exposure_threshold,
        )
        if tonb_metrics is not None:
            tonb_row = {"sample_id": sample_path.stem, **tonb_metrics}
            tonb_rows.append(tonb_row)

        sample_results.append(
            {
                "sample_id": sample_path.stem,
                "path": str(sample_path),
                "n_residues": n,
                "barrel_core_target_rms": barrel_core_target_rms,
                "regions": region_blocks,
                "tonb_state_metrics": tonb_metrics,
            }
        )

    aggregate_by_region = {}
    for region_name, tensors in sorted(aggregate_vectors.items()):
        target = torch.cat(tensors["target"], dim=0)
        prediction = torch.cat(tensors["prediction"], dim=0)
        mask = torch.ones(target.size(0), dtype=torch.bool)
        metrics = _region_metrics(
            target,
            prediction,
            mask,
            direction_threshold=direction_threshold,
        )
        aggregate_by_region[region_name] = metrics

    sample_level_by_region = _sample_region_summary(sample_results)
    paired_delta_by_region, paired_delta_rows = _paired_delta_summary(
        sample_results,
        bootstrap_iter=bootstrap_iter,
        bootstrap_seed=bootstrap_seed,
    )

    aggregate_barrel_core_target_rms = aggregate_by_region.get("barrel_core", {}).get(
        "target_displacement_rms",
        float("nan"),
    )
    for region_name, metrics in aggregate_by_region.items():
        metrics["barrel_core_target_rms"] = (
            metrics["target_displacement_rms"] if region_name == "barrel_core" else aggregate_barrel_core_target_rms
        )
        metrics.update(sample_level_by_region.get(region_name, {}))
        csv_rows.append(
            {
                "scope": "aggregate",
                "sample_id": "",
                "region": region_name,
                **metrics,
            }
        )

    report = {
        "task": "tbdt_state_displacement",
        "prediction_source": args.predictions or "zero_displacement_baseline",
        "metric_contract": {
            "primary_unit": "region_residue_ca_displacement",
            "full_chain_rmsd_primary": False,
            "improvement_vs_zero": "zero_error_rms - prediction_error_rms, in Angstrom",
            "better_than_zero_rate": "fraction of residues whose prediction error norm is lower than raw AF2/zero displacement",
            "mse_improvement_vs_zero_fraction": "(zero vector MSE - prediction MSE) / zero vector MSE",
            "sample_improvement_rate": "fraction of samples whose region RMSD is improved versus zero displacement",
            "paired_delta_rmsd": "method prediction_error_rms - raw AF2/zero_error_rms at sample level; negative is better",
            "paired_delta_wilcoxon_less": "one-sided Wilcoxon signed-rank test for method RMSD < raw AF2 RMSD",
            "derived_regions": "plug_core, plug_apical_loop, and plug_extension_nt are sequence-order heuristics unless explicit masks exist",
            "tonb_state_metrics": "TonB centroid exposure/distance/vector metrics use AF2 barrel/plug centroids as reference points",
            "direction_cosine_threshold_A": direction_threshold,
            "tonb_exposure_threshold_A": tonb_exposure_threshold,
        },
        "n_samples": len(samples),
        "aggregate_by_region": aggregate_by_region,
        "paired_delta_by_region": paired_delta_by_region,
        "tonb_state_summary": _tonb_summary(tonb_rows),
        "tonb_state_samples": tonb_rows,
        "samples": sample_results,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, allow_nan=False)
    if args.output_csv:
        _write_csv(csv_rows, Path(args.output_csv))
    if getattr(args, "paired_delta_csv", None):
        _write_generic_csv(paired_delta_rows, Path(args.paired_delta_csv))
    if getattr(args, "tonb_metrics_csv", None):
        _write_generic_csv(tonb_rows, Path(args.tonb_metrics_csv))
    return report
