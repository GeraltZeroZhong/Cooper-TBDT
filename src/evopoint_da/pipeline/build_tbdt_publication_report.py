"""Assemble publication-grade TBDT v1 reporting tables.

The script does not rerun training. It collects reproducible dataset statistics,
coordinate metrics, baseline comparisons, classification curves, quality flags,
and file hashes from the current artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, wilcoxon

from evopoint_da.pipeline.eval_tbdt_state import _as_mapping, _extract_regions, _extract_target, _load_pt
from evopoint_da.pipeline.build_features_with_sasa import (
    _metadata_af2_path,
    _metadata_uniprot_id,
    _resolve_pae_path,
    build_uniprot_to_af2_path,
)
from evopoint_da.data.graph import parse_pae_matrix_for_indices, parse_pae_matrix_for_residue_ids
from evopoint_da.data.structure import StructureParser


REGIONS = ("eval", "plug", "tonb_box", "barrel_core", "all")
PRIMARY_REGIONS = ("eval", "plug", "tonb_box", "barrel_core")
DISPLACEMENT_BINS = (
    ("lt_0p5", 0.0, 0.5),
    ("0p5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("2_to_5", 2.0, 5.0),
    ("ge_5", 5.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TBDT v1 publication reporting tables.")
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/publication_report")
    parser.add_argument("--mixed-manifest", default="data/tbdt_mixed_manifest.csv")
    parser.add_argument("--gold-manifest", default="data/tbdt_gold_training_manifest.csv")
    parser.add_argument("--silver-manifest", default="data/tbdt_silver_manifest.csv")
    parser.add_argument("--bronze-manifest", default="data/tbdt_bronze_manifest.csv")
    parser.add_argument("--gold-pair-dir", default="data/processed_tbdt_gold_pairs")
    parser.add_argument("--silver-pair-dir", default="data/processed_tbdt_silver_pairs_clean")
    parser.add_argument("--gold-graph-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--silver-graph-dir", default="data/processed_tbdt_silver_graphs_clean")
    parser.add_argument("--af2-structure-dir", default="data/raw_af2")
    parser.add_argument("--pae-dir", default="data/raw_af2")
    parser.add_argument("--test-sample-list", default="artifacts/tbdt_v1/test_graph_files.txt")
    parser.add_argument("--bootstrap-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _numeric(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            if str(value).strip() == "":
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _stats(values: list[float]) -> dict[str, float | int | None]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "p10": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _counter_rows(rows: list[dict[str, str]], field: str, scope: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(field) or "").strip() or "missing" for row in rows)
    return [
        {"scope": scope, "field": field, "value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _manifest_summary(rows: list[dict[str, str]], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rows": len(rows),
        "unique_uniprot": len({row.get("uniprot_id", "") for row in rows if row.get("uniprot_id")}),
        "unique_pdb": len({row.get("pdb_id", "") for row in rows if row.get("pdb_id")}),
        "family_counts": dict(Counter(row.get("family", "") or "missing" for row in rows)),
        "state_counts": dict(Counter(row.get("state_label", "") or row.get("state", "") or "missing" for row in rows)),
        "substrate_counts": dict(Counter(row.get("substrate_class", "") or row.get("substrate", "") or "missing" for row in rows)),
        "split_counts": dict(Counter(row.get("split", "") or "missing" for row in rows)),
        "method_counts": dict(Counter(row.get("method", "") or "missing" for row in rows)),
        "evidence_counts": dict(Counter(row.get("evidence_level", "") or "missing" for row in rows)),
        "resolution": _stats(_numeric([row.get("resolution", "") for row in rows])),
        "sequence_length": _stats(_numeric([row.get("sequence_length", "") for row in rows])),
        "reference_coverage": _stats(_numeric([row.get("reference_coverage", "") for row in rows])),
        "mutation_count": _stats(_numeric([row.get("mutation_count", "") for row in rows])),
        "deletion_count": _stats(_numeric([row.get("deletion_count", "") for row in rows])),
        "af_version_counts": dict(Counter(row.get("af_version", "") or "missing" for row in rows)),
    }


def _split_leakage(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "").strip()
        if split not in {"train", "val", "test"}:
            continue
        group = str(row.get("split_group_id") or row.get("uniprot_id") or "").strip()
        if group:
            by_group[group].add(split)
    overlaps = {
        group: sorted(splits)
        for group, splits in sorted(by_group.items())
        if len(splits.intersection({"train", "val", "test"})) > 1
    }
    return {
        "group_field": "split_group_id_or_uniprot_id",
        "n_groups": len(by_group),
        "n_overlapping_groups": len(overlaps),
        "overlapping_groups": overlaps,
        "status": "passed" if not overlaps else "failed",
    }


def _graph_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.pt")) if path.exists() else []


def _pair_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.pt")) if path.exists() else []


def _strict_input_check(pair_dir: Path, af2_dir: Path, pae_dir: Path, scope: str) -> dict[str, Any]:
    files = _pair_files(pair_dir)
    parser = StructureParser()
    af2_by_uniprot = build_uniprot_to_af2_path(str(af2_dir))
    af2_by_lower = {key.lower(): value for key, value in af2_by_uniprot.items()}
    residue_cache: dict[str, set[str]] = {}
    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _example(kind: str, payload: dict[str, Any]) -> None:
        if len(examples[kind]) < 8:
            examples[kind].append(payload)

    for path in files:
        sample = _as_mapping(_load_pt(path))
        stem = str(sample.get("pair_id") or path.stem)
        uniprot_id = _metadata_uniprot_id(sample)
        af2_path = _metadata_af2_path(sample)
        if uniprot_id:
            af2_path = af2_path or af2_by_uniprot.get(uniprot_id) or af2_by_lower.get(uniprot_id.lower())
        if not af2_path or not Path(af2_path).exists():
            counts["missing_af2"] += 1
            _example("missing_af2", {"pair_id": stem, "uniprot_id": uniprot_id})
            continue

        af2_key = str(Path(af2_path).resolve())
        if af2_key not in residue_cache:
            chains = parser.parse_ca_structure(af2_key, strict=True) or {}
            residue_cache[af2_key] = {
                str(rid)
                for chain_data in chains.values()
                for rid in chain_data.get("residue_ids", [])
            }
        residue_ids = [str(rid) for rid in sample.get("residue_ids", [])]
        missing_residue_ids = [rid for rid in residue_ids if rid not in residue_cache[af2_key]]
        if missing_residue_ids:
            counts["residue_id_mismatch"] += 1
            _example(
                "residue_id_mismatch",
                {
                    "pair_id": stem,
                    "missing_count": len(missing_residue_ids),
                    "missing_preview": missing_residue_ids[:5],
                },
            )

        pae_path = _resolve_pae_path(str(pae_dir), uniprot_id, stem)
        if pae_path is None:
            counts["missing_pae"] += 1
            _example("missing_pae", {"pair_id": stem, "uniprot_id": uniprot_id})
            continue
        try:
            if "af2_indices" in sample:
                indices = [int(value) for value in torch.as_tensor(sample["af2_indices"]).reshape(-1).tolist()]
                parse_pae_matrix_for_indices(pae_path, indices, strict=True)
            else:
                parse_pae_matrix_for_residue_ids(pae_path, residue_ids, strict=True)
        except Exception as exc:
            counts["pae_alignment_error"] += 1
            _example("pae_alignment_error", {"pair_id": stem, "pae_path": pae_path, "error": str(exc)})

    failed_pairs = sum(counts.values())
    return {
        "scope": scope,
        "pair_dir": str(pair_dir),
        "af2_structure_dir": str(af2_dir),
        "pae_dir": str(pae_dir),
        "pair_count": len(files),
        "failed_pair_checks": int(failed_pairs),
        "counts": dict(counts),
        "examples": dict(examples),
        "status": "passed" if failed_pairs == 0 else "failed",
    }


def _sample_split(sample: dict[str, Any]) -> str:
    split = sample.get("split")
    if split:
        return str(split)
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        split = metadata.get("split")
        if split:
            return str(split)
        manifest_row = metadata.get("manifest_row", {})
        if isinstance(manifest_row, dict) and manifest_row.get("split"):
            return str(manifest_row["split"])
    return "unknown"


def _region_graph_summary(files: list[Path], scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate: dict[str, dict[str, list[torch.Tensor] | int]] = {
        region: {"target": [], "count": 0, "samples": 0} for region in REGIONS
    }
    sample_rows: list[dict[str, Any]] = []
    plddt_values: list[float] = []
    for path in files:
        sample = _as_mapping(_load_pt(path))
        target = _extract_target(sample, path)
        n = int(target.size(0))
        regions = _extract_regions(sample, path.stem, n, {}, include_all=True)
        plddt = sample.get("plddt")
        if plddt is not None:
            plddt_tensor = torch.as_tensor(plddt, dtype=torch.float32).reshape(-1)
            if plddt_tensor.numel():
                plddt_values.append(float(plddt_tensor.mean().item()))
        row = {"scope": scope, "sample_id": path.stem, "n_residues": n}
        for region in REGIONS:
            mask = regions.get(region)
            if mask is None or not bool(mask.any()):
                row[f"{region}_residues"] = 0
                continue
            vectors = target[mask]
            aggregate[region]["target"].append(vectors)
            aggregate[region]["count"] = int(aggregate[region]["count"]) + int(mask.sum().item())
            aggregate[region]["samples"] = int(aggregate[region]["samples"]) + 1
            rms = float(torch.sqrt(torch.mean(torch.sum(vectors.square(), dim=-1))).item())
            row[f"{region}_residues"] = int(mask.sum().item())
            row[f"{region}_target_rms"] = rms
        sample_rows.append(row)

    region_rows: list[dict[str, Any]] = []
    for region, payload in aggregate.items():
        vectors_list = payload["target"]
        if vectors_list:
            vectors = torch.cat(vectors_list, dim=0)
            norms = torch.linalg.vector_norm(vectors, dim=-1)
            target_rms = float(torch.sqrt(torch.mean(torch.sum(vectors.square(), dim=-1))).item())
            target_mean = float(norms.mean().item())
        else:
            target_rms = float("nan")
            target_mean = float("nan")
        region_rows.append(
            {
                "scope": scope,
                "region": region,
                "n_samples_with_region": int(payload["samples"]),
                "n_residues": int(payload["count"]),
                "target_displacement_rms": target_rms,
                "target_displacement_mean": target_mean,
            }
        )
    summary = {
        "scope": scope,
        "n_graphs": len(files),
        "node_count": _stats([float(row["n_residues"]) for row in sample_rows]),
        "plddt_mean_raw": _stats(plddt_values),
    }
    return sample_rows + region_rows, summary


def _displacement_bin_rows(files: list[Path], scope: str) -> list[dict[str, Any]]:
    accum: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in files:
        sample = _as_mapping(_load_pt(path))
        split = _sample_split(sample)
        target = _extract_target(sample, path)
        n = int(target.size(0))
        regions = _extract_regions(sample, path.stem, n, {}, include_all=True)
        norms = torch.linalg.vector_norm(target, dim=-1)
        for region in REGIONS:
            mask = regions.get(region)
            if mask is None or not bool(mask.any()):
                continue
            region_norms = norms[mask]
            for label, lo, hi in DISPLACEMENT_BINS:
                if math.isinf(hi):
                    bin_mask = region_norms >= float(lo)
                else:
                    bin_mask = (region_norms >= float(lo)) & (region_norms < float(hi))
                key = (split, region, label)
                row = accum.setdefault(
                    key,
                    {
                        "scope": scope,
                        "split": split,
                        "region": region,
                        "bin": label,
                        "low_angstrom": lo,
                        "high_angstrom": "" if math.isinf(hi) else hi,
                        "n_residues": 0,
                        "n_samples_with_region": 0,
                        "region_residue_total": 0,
                    },
                )
                row["n_residues"] += int(bin_mask.sum().item())
            region_total = int(mask.sum().item())
            for label, _lo, _hi in DISPLACEMENT_BINS:
                row = accum[(split, region, label)]
                row["n_samples_with_region"] += 1
                row["region_residue_total"] += region_total

    rows = []
    for row in accum.values():
        total = int(row["region_residue_total"])
        row = dict(row)
        row["fraction"] = (float(row["n_residues"]) / float(total)) if total else None
        rows.append(row)
    return sorted(rows, key=lambda item: (item["scope"], item["split"], item["region"], item["low_angstrom"]))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bootstrap_ci(
    values: list[float],
    *,
    n_iter: int,
    seed: int,
    statistic: str = "mean",
) -> tuple[float | None, float | None]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_iter), dtype=float)
    for idx in range(int(n_iter)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[idx] = float(np.median(sample) if statistic == "median" else np.mean(sample))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _wilcoxon_delta_stats(deltas: list[float]) -> dict[str, Any]:
    values = np.asarray([float(v) for v in deltas if math.isfinite(float(v)) and abs(float(v)) > 1e-12])
    if values.size == 0:
        return {
            "wilcoxon_n_nonzero": 0,
            "wilcoxon_statistic_less": None,
            "wilcoxon_p_less_method_lt_raw": None,
            "wilcoxon_statistic_two_sided": None,
            "wilcoxon_p_two_sided": None,
            "signed_rank_biserial_effect_method_lt_raw": None,
            "wilcoxon_status": "no_nonzero_deltas",
        }
    ranks = rankdata(np.abs(values), method="average")
    improved_rank_sum = float(np.sum(ranks[values < 0.0]))
    worsened_rank_sum = float(np.sum(ranks[values > 0.0]))
    rank_total = improved_rank_sum + worsened_rank_sum
    effect = (improved_rank_sum - worsened_rank_sum) / rank_total if rank_total else None
    try:
        less = wilcoxon(values, alternative="less", zero_method="wilcox")
        two = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:
        return {
            "wilcoxon_n_nonzero": int(values.size),
            "wilcoxon_statistic_less": None,
            "wilcoxon_p_less_method_lt_raw": None,
            "wilcoxon_statistic_two_sided": None,
            "wilcoxon_p_two_sided": None,
            "signed_rank_biserial_effect_method_lt_raw": effect,
            "wilcoxon_status": str(exc),
        }
    return {
        "wilcoxon_n_nonzero": int(values.size),
        "wilcoxon_statistic_less": float(less.statistic),
        "wilcoxon_p_less_method_lt_raw": float(less.pvalue),
        "wilcoxon_statistic_two_sided": float(two.statistic),
        "wilcoxon_p_two_sided": float(two.pvalue),
        "signed_rank_biserial_effect_method_lt_raw": effect,
        "wilcoxon_status": "ok",
    }


def _coordinate_metric_specs() -> list[tuple[str, str, Path, str]]:
    return [
        (
            "raw_af2_zero",
            "Raw AF2 / zero displacement",
            Path("artifacts/tbdt_v1/gold_real_test_zero_region_metrics.json"),
            "baseline",
        ),
        (
            "foldseek_nearest_template",
            "Foldseek nearest-template transfer",
            Path("artifacts/tbdt_v1/template_baselines/foldseek_nearest_template_region_metrics.json"),
            "external_template_baseline",
        ),
        (
            "usalign_nearest_template",
            "US-align nearest-template transfer",
            Path("artifacts/tbdt_v1/template_baselines/usalign_nearest_template_region_metrics.json"),
            "external_template_baseline",
        ),
        (
            "nearest_template",
            "Nearest template transfer",
            Path("artifacts/tbdt_v1/template_baselines/nearest_template_region_metrics.json"),
            "template_baseline",
        ),
        (
            "family_state_average",
            "Family/state average transfer",
            Path("artifacts/tbdt_v1/template_baselines/family_state_average_region_metrics.json"),
            "template_baseline",
        ),
        (
            "cooper_tbdt_scaffold_single",
            "Cooper-TBDT single scaffold-prior representative seed 404",
            Path("artifacts/tbdt_v1/seed_stability_best_selection/metrics/seed_404_best-selection_test.json"),
            "primary_model",
        ),
        (
            "cooper_tbdt_scaffold_blend",
            "Cooper-TBDT validation-calibrated scaffold-prior blend",
            Path("artifacts/tbdt_v1/report_models/metrics/validation_calibrated_region_blend_test.json"),
            "secondary_validation_calibrated_model",
        ),
        (
            "global_region_mean",
            "Global region-mean displacement",
            Path("artifacts/tbdt_v1/coordinate_baselines/global_region_mean_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "family_region_mean",
            "Family region-mean displacement",
            Path("artifacts/tbdt_v1/coordinate_baselines/family_region_mean_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "state_region_mean",
            "State region-mean displacement",
            Path("artifacts/tbdt_v1/coordinate_baselines/state_region_mean_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "family_state_region_mean",
            "Family/state region-mean displacement",
            Path("artifacts/tbdt_v1/coordinate_baselines/family_state_region_mean_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "region_centroid_shift",
            "Rigid region centroid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/region_centroid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "family_region_centroid_shift",
            "Family rigid region centroid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/family_region_centroid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "state_region_centroid_shift",
            "State rigid region centroid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/state_region_centroid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "family_state_region_centroid_shift",
            "Family/state rigid region centroid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/family_state_region_centroid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "plug_rigid_shift",
            "Plug-only rigid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/plug_rigid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "plug_apical_loop_rigid_shift",
            "Plug apical-loop-only rigid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/plug_apical_loop_rigid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "tonb_box_rigid_shift",
            "TonB-box-only rigid shift",
            Path("artifacts/tbdt_v1/coordinate_baselines/tonb_box_rigid_shift_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "barrel_frame_ridge",
            "Barrel-frame ridge baseline",
            Path("artifacts/tbdt_v1/coordinate_baselines/barrel_frame_ridge_region_metrics.json"),
            "coordinate_baseline",
        ),
        (
            "gold_balanced",
            "Gold-only balanced ablation",
            Path("artifacts/tbdt_v1/report_models/metrics/gold_balanced_test.json"),
            "ablation",
        ),
        (
            "gold_plug_specialist",
            "Gold-only plug/eval specialist",
            Path("artifacts/tbdt_v1/report_models/metrics/gold_plug_specialist_test.json"),
            "ablation",
        ),
        (
            "gold_tonb_specialist",
            "Gold-only TonB specialist",
            Path("artifacts/tbdt_v1/report_models/metrics/gold_tonb_specialist_test.json"),
            "ablation",
        ),
        (
            "silver_pretrain_finetune",
            "Silver pretrain then Gold fine-tune",
            Path("artifacts/tbdt_v1/report_models/metrics/silver_pretrain_gold_finetune_test.json"),
            "ablation",
        ),
    ]


def _coordinate_tables(n_iter: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for method, label, path, category in _coordinate_metric_specs():
        report = _load_json(path)
        if not report:
            continue
        for region, metrics in (report.get("aggregate_by_region") or {}).items():
            if region not in REGIONS:
                continue
            values = []
            improved_flags = []
            for sample in report.get("samples") or []:
                sm = (sample.get("regions") or {}).get(region)
                if not sm:
                    continue
                zero = float(sm.get("zero_error_rms", float("nan")))
                err = float(sm.get("prediction_error_rms", float("nan")))
                if math.isfinite(zero) and math.isfinite(err):
                    delta = zero - err
                    values.append(delta)
                    improved_flags.append(1.0 if delta > 0 else 0.0)
                    sample_rows.append(
                        {
                            "method": method,
                            "label": label,
                            "category": category,
                            "sample_id": sample.get("sample_id", ""),
                            "region": region,
                            "zero_error_rms": zero,
                            "prediction_error_rms": err,
                            "sample_improvement": delta,
                            "mse_improvement_vs_zero_fraction": sm.get("mse_improvement_vs_zero_fraction", ""),
                        }
                    )
            mean_ci = _bootstrap_ci(values, n_iter=n_iter, seed=seed) if values else (None, None)
            rate_ci = _bootstrap_ci(improved_flags, n_iter=n_iter, seed=seed + 17) if improved_flags else (None, None)
            aggregate_rows.append(
                {
                    "method": method,
                    "label": label,
                    "category": category,
                    "region": region,
                    "n_residues": metrics.get("n_residues"),
                    "sample_count": metrics.get("sample_count"),
                    "zero_error_rms": metrics.get("zero_error_rms"),
                    "prediction_error_rms": metrics.get("prediction_error_rms"),
                    "target_displacement_rms": metrics.get("target_displacement_rms"),
                    "predicted_displacement_mean": metrics.get("predicted_displacement_mean"),
                    "mse_improvement_vs_zero_fraction": metrics.get("mse_improvement_vs_zero_fraction"),
                    "better_than_zero_rate": metrics.get("better_than_zero_rate"),
                    "direction_cosine_mean": metrics.get("direction_cosine_mean"),
                    "sample_improvement_rate": metrics.get("sample_improvement_rate"),
                    "sample_improvement_mean": metrics.get("sample_improvement_mean"),
                    "sample_improvement_median": metrics.get("sample_improvement_median"),
                    "sample_improvement_mean_ci95_low": mean_ci[0],
                    "sample_improvement_mean_ci95_high": mean_ci[1],
                    "sample_improvement_rate_ci95_low": rate_ci[0],
                    "sample_improvement_rate_ci95_high": rate_ci[1],
                    "source_json": str(path),
                }
            )
    return aggregate_rows, sample_rows


def _classification_rows(path: Path) -> list[dict[str, Any]]:
    report = _load_json(path)
    rows = []
    for row in report.get("summary") or []:
        rows.append({**row, "source_json": str(path)})
    return rows


def _template_selection_summary(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    identities = _numeric([row.get("nearest_identity", "") for row in rows])
    coverages = _numeric([row.get("nearest_target_coverage", "") for row in rows])
    candidate_counts = _numeric([row.get("candidate_count", "") for row in rows])
    fit_regions = Counter(row.get("nearest_fit_region", "") or "missing" for row in rows)
    return {
        "rows": len(rows),
        "candidate_count": _stats(candidate_counts),
        "nearest_identity": _stats(identities),
        "nearest_target_coverage": _stats(coverages),
        "nearest_fit_region_counts": dict(fit_regions),
        "targets_with_no_candidate": sum(1 for row in rows if str(row.get("candidate_count") or "0") == "0"),
    }


def _hash_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "sha256": _sha256(path),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    selected = rows[:max_rows] if max_rows else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}" if math.isfinite(value) else ""
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _seed_stability_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_csv(path)
    labels = {
        ("test", "eval", "prediction_error_rms"): "eval RMSD A",
        ("test", "plug", "prediction_error_rms"): "plug RMSD A",
        ("test", "tonb_box", "prediction_error_rms"): "TonB RMSD A",
        ("test", "barrel_core", "predicted_displacement_mean"): "barrel-core predicted mean A",
        ("test", "selection", "score"): "validation-style score",
    }
    rows: list[dict[str, Any]] = []
    for source in raw_rows:
        key = (source.get("split", ""), source.get("region", ""), source.get("metric", ""))
        label = labels.get(key)
        if label is None:
            continue
        row = {"endpoint": label}
        for field in ("n_seeds", "mean", "std", "min", "max"):
            value = source.get(field, "")
            if field == "n_seeds":
                row[field] = int(value) if str(value).strip() else ""
            else:
                row[field] = _float_or_none(value)
        rows.append(row)
    return rows


def _paired_delta_summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[
            (
                str(row.get("method", "")),
                str(row.get("label", "")),
                str(row.get("aggregation", "")),
                str(row.get("region", "")),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (method, label, aggregation, region), region_rows in sorted(by_key.items()):
        deltas = [
            float(row["delta_rmsd_method_minus_raw"])
            for row in region_rows
            if _float_or_none(row.get("delta_rmsd_method_minus_raw")) is not None
        ]
        if not deltas:
            continue
        mean_ci = _bootstrap_ci(deltas, n_iter=n_iter, seed=seed, statistic="mean")
        median_ci = _bootstrap_ci(deltas, n_iter=n_iter, seed=seed + 101, statistic="median")
        seed_counts = [
            int(row.get("n_seeds") or 0)
            for row in region_rows
            if str(row.get("n_seeds") or "").strip()
        ]
        summary.append(
            {
                "method": method,
                "label": label,
                "category": "primary_model",
                "aggregation": aggregation,
                "region": region,
                "n_targets": len(deltas),
                "n_seeds": max(seed_counts) if seed_counts else "",
                "n_improved": sum(1 for value in deltas if value < 0.0),
                "n_worsened": sum(1 for value in deltas if value > 0.0),
                "n_tied": sum(1 for value in deltas if value == 0.0),
                "improved_fraction": sum(1.0 for value in deltas if value < 0.0) / float(len(deltas)),
                "median_delta_rmsd_method_minus_raw": float(np.median(deltas)),
                "mean_delta_rmsd_method_minus_raw": float(np.mean(deltas)),
                "mean_delta_ci95_low": mean_ci[0],
                "mean_delta_ci95_high": mean_ci[1],
                "median_delta_ci95_low": median_ci[0],
                "median_delta_ci95_high": median_ci[1],
                **_wilcoxon_delta_stats(deltas),
            }
        )
    return summary


def _primary_model_paired_delta_tables(
    metric_dir: Path,
    *,
    n_iter: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_files = sorted(metric_dir.glob("seed_*_best-selection_test.json"))
    seed_rows: list[dict[str, Any]] = []
    per_target: dict[tuple[str, str], dict[str, Any]] = {}
    for path in seed_files:
        stem_parts = path.stem.split("_")
        seed_id = stem_parts[1] if len(stem_parts) > 1 else ""
        report = _load_json(path)
        for sample in report.get("samples") or []:
            sample_id = str(sample.get("sample_id") or "")
            for region, metrics in (sample.get("regions") or {}).items():
                if region not in REGIONS:
                    continue
                raw = _float_or_none(metrics.get("zero_error_rms"))
                method_rmsd = _float_or_none(metrics.get("prediction_error_rms"))
                if raw is None or method_rmsd is None:
                    continue
                delta = method_rmsd - raw
                seed_rows.append(
                    {
                        "method": f"cooper_tbdt_scaffold_single_seed{seed_id}",
                        "label": f"Cooper-TBDT single scaffold-prior seed {seed_id}",
                        "category": "primary_model_seed",
                        "aggregation": "single_seed",
                        "seed": seed_id,
                        "sample_id": sample_id,
                        "region": region,
                        "raw_af2_rmsd": raw,
                        "method_rmsd": method_rmsd,
                        "delta_rmsd_method_minus_raw": delta,
                        "improved": delta < 0.0,
                        "worsened": delta > 0.0,
                        "n_residues": metrics.get("n_residues", ""),
                        "n_seeds": 1,
                        "source_json": str(path),
                    }
                )
                target_key = (sample_id, region)
                target = per_target.setdefault(
                    target_key,
                    {
                        "sample_id": sample_id,
                        "region": region,
                        "raw_values": [],
                        "method_values": [],
                        "seed_ids": [],
                    },
                )
                target["raw_values"].append(raw)
                target["method_values"].append(method_rmsd)
                target["seed_ids"].append(seed_id)

    family_rows: list[dict[str, Any]] = []
    for (_sample_id, _region), target in sorted(per_target.items()):
        raw_values = [float(value) for value in target["raw_values"]]
        method_values = [float(value) for value in target["method_values"]]
        raw = float(np.mean(raw_values))
        method_rmsd = float(np.mean(method_values))
        delta = method_rmsd - raw
        family_rows.append(
            {
                "method": "cooper_tbdt_scaffold_single_5seed_mean",
                "label": "Cooper-TBDT single scaffold-prior 5-seed family",
                "category": "primary_model",
                "aggregation": "per_target_seed_mean",
                "seed": "all",
                "sample_id": target["sample_id"],
                "region": target["region"],
                "raw_af2_rmsd": raw,
                "method_rmsd": method_rmsd,
                "delta_rmsd_method_minus_raw": delta,
                "improved": delta < 0.0,
                "worsened": delta > 0.0,
                "n_residues": "",
                "n_seeds": len(set(target["seed_ids"])),
                "source_json": str(metric_dir),
            }
        )

    sample_rows = family_rows + seed_rows
    summary_rows = _paired_delta_summary_from_rows(sample_rows, n_iter=n_iter, seed=seed)
    return summary_rows, sample_rows


def _selector_sensitivity_rows() -> list[dict[str, Any]]:
    selectors = [
        ("best-selection", Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv")),
        ("best-disp1to5", Path("artifacts/tbdt_v1/seed_stability/seed_stability_aggregate.csv")),
        ("best-disp1to2", Path("artifacts/tbdt_v1/seed_stability_best_disp1to2/seed_stability_aggregate.csv")),
        ("best-flex", Path("artifacts/tbdt_v1/seed_stability_best_flex/seed_stability_aggregate.csv")),
    ]
    endpoint_labels = {
        ("eval", "prediction_error_rms"): "eval RMSD A",
        ("plug", "prediction_error_rms"): "plug RMSD A",
        ("tonb_box", "prediction_error_rms"): "TonB RMSD A",
        ("barrel_core", "predicted_displacement_mean"): "barrel-core predicted mean A",
        ("selection", "score"): "validation-style score",
    }
    rows: list[dict[str, Any]] = []
    for selector, path in selectors:
        for source in _read_csv(path):
            if source.get("split") != "test":
                continue
            region = source.get("region", "")
            metric = source.get("metric", "")
            label = endpoint_labels.get((region, metric))
            if label is None:
                continue
            rows.append(
                {
                    "selector": selector,
                    "endpoint": label,
                    "region": region,
                    "metric": metric,
                    "n_seeds": int(source.get("n_seeds") or 0),
                    "mean": _float_or_none(source.get("mean")),
                    "std": _float_or_none(source.get("std")),
                    "min": _float_or_none(source.get("min")),
                    "max": _float_or_none(source.get("max")),
                    "source_csv": str(path),
                }
            )
    return rows


def _neural_comparison_rows(coordinate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = {
        "cooper_tbdt_scaffold_single",
        "cooper_tbdt_scaffold_blend",
        "gold_balanced",
        "gold_plug_specialist",
        "gold_tonb_specialist",
        "silver_pretrain_finetune",
    }
    rows = [
        row
        for row in coordinate_rows
        if row.get("method") in methods and row.get("region") in PRIMARY_REGIONS
    ]
    order = {
        "cooper_tbdt_scaffold_single": 0,
        "gold_balanced": 1,
        "gold_plug_specialist": 2,
        "gold_tonb_specialist": 3,
        "silver_pretrain_finetune": 4,
        "cooper_tbdt_scaffold_blend": 5,
    }
    return sorted(rows, key=lambda row: (order.get(str(row.get("method")), 99), str(row.get("region"))))


def _quality_flags(
    *,
    build_reports: dict[str, dict[str, Any]],
    classification_rows: list[dict[str, Any]],
    template_summary: dict[str, Any],
    split_leakage: dict[str, Any],
    strict_input_checks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for name, report in build_reports.items():
        input_pairs = int(report.get("input_pairs") or 0)
        skip_counts = report.get("skip_reason_counts") or {}
        missing_pae = int(skip_counts.get("missing_pae_zero_edge_feature", 0)) + int(
            skip_counts.get("missing_pae_fallback_zero", 0)
        )
        if input_pairs and missing_pae / input_pairs > 0.25:
            flags.append(
                {
                    "severity": "medium",
                    "scope": name,
                    "issue": f"PAE missing for {missing_pae}/{input_pairs} graphs; explicit zero-PAE edge features were used.",
                }
            )
        missing_af2 = int(skip_counts.get("missing_af2_structure", 0))
        if missing_af2:
            flags.append(
                {
                    "severity": "high",
                    "scope": name,
                    "issue": f"AF2 structure missing for {missing_af2}/{input_pairs} graphs.",
                }
            )
    if split_leakage.get("status") != "passed":
        flags.append({"severity": "high", "scope": "split", "issue": "Split-group leakage detected."})
    for check in strict_input_checks:
        if check.get("status") != "passed":
            flags.append(
                {
                    "severity": "high",
                    "scope": str(check.get("scope") or "strict_input_check"),
                    "issue": f"Strict AF2/PAE input check failed: {check.get('counts')}",
                }
            )
    for row in classification_rows:
        if row.get("region") == "tonb_box":
            positives = int(row.get("n_positive") or 0)
            negatives = int(row.get("n_negative") or 0)
            if positives and negatives and min(positives, negatives) < 5:
                flags.append(
                    {
                        "severity": "medium",
                        "scope": "tonb_box_classification",
                        "issue": f"TonB-box ROC/PR is class-imbalanced ({positives} positive, {negatives} negative residues).",
                    }
                )
                break
    coverage = template_summary.get("nearest_target_coverage", {})
    median_coverage = coverage.get("median")
    if isinstance(median_coverage, (int, float)) and median_coverage < 0.15:
        flags.append(
            {
                "severity": "medium",
                "scope": "template_baseline",
                "issue": f"Nearest-template median target coverage is low ({median_coverage:.3f}); report coverage with template results.",
            }
        )
    eval_rows = [row for row in classification_rows if row.get("region") == "eval"]
    model = next((row for row in eval_rows if row.get("method") == "cooper_tbdt_scaffold_blend"), None)
    plddt = next((row for row in eval_rows if row.get("method") == "af2_low_plddt"), None)
    if model and plddt and float(plddt.get("auroc") or 0.0) > float(model.get("auroc") or 0.0):
        flags.append(
            {
                "severity": "medium",
                "scope": "residue_localization",
                "issue": "AF2 low-pLDDT ranks moving residues better than the model on eval ROC/PR; keep ROC/PR secondary.",
            }
        )
    return flags


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_rows = _read_csv(Path(args.mixed_manifest))
    gold_rows = _read_csv(Path(args.gold_manifest))
    silver_rows = _read_csv(Path(args.silver_manifest))
    bronze_rows = _read_csv(Path(args.bronze_manifest))

    manifest_summaries = [
        _manifest_summary(mixed_rows, "mixed"),
        _manifest_summary(gold_rows, "gold_training"),
        _manifest_summary(silver_rows, "silver"),
        _manifest_summary(bronze_rows, "bronze"),
    ]
    manifest_count_rows: list[dict[str, Any]] = []
    for name, rows in [
        ("mixed", mixed_rows),
        ("gold_training", gold_rows),
        ("silver", silver_rows),
        ("bronze", bronze_rows),
    ]:
        for field in ("evidence_level", "family", "state_label", "substrate_class", "split", "method", "af_version"):
            manifest_count_rows.extend(_counter_rows(rows, field, name))

    split_leakage = _split_leakage(gold_rows)
    strict_input_checks = [
        _strict_input_check(
            Path(args.gold_pair_dir),
            Path(args.af2_structure_dir),
            Path(args.pae_dir),
            "gold_pairs",
        ),
        _strict_input_check(
            Path(args.silver_pair_dir),
            Path(args.af2_structure_dir),
            Path(args.pae_dir),
            "silver_clean_pairs",
        ),
    ]
    graph_rows_gold, graph_summary_gold = _region_graph_summary(_graph_files(Path(args.gold_graph_dir)), "gold_graphs")
    graph_rows_silver, graph_summary_silver = _region_graph_summary(
        _graph_files(Path(args.silver_graph_dir)),
        "silver_clean_graphs",
    )
    graph_rows = graph_rows_gold + graph_rows_silver
    displacement_bin_rows = _displacement_bin_rows(
        _graph_files(Path(args.gold_graph_dir)),
        "gold_graphs",
    ) + _displacement_bin_rows(_graph_files(Path(args.silver_graph_dir)), "silver_clean_graphs")

    coordinate_rows, coordinate_sample_rows = _coordinate_tables(int(args.bootstrap_iter), int(args.bootstrap_seed))
    neural_comparison_rows = _neural_comparison_rows(coordinate_rows)
    seed_stability_rows = _seed_stability_rows(
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv")
    )
    primary_paired_delta_rows, primary_paired_delta_sample_rows = _primary_model_paired_delta_tables(
        Path("artifacts/tbdt_v1/seed_stability_best_selection/metrics"),
        n_iter=int(args.bootstrap_iter),
        seed=int(args.bootstrap_seed),
    )
    selector_sensitivity_rows = _selector_sensitivity_rows()
    classification_rows = _classification_rows(Path("artifacts/tbdt_v1/external_baseline_curves/classification_curve_report.json"))
    template_summary = _template_selection_summary(Path("artifacts/tbdt_v1/template_baselines/template_baseline_selection.csv"))

    build_reports = {
        "gold_graphs": _load_json(Path("artifacts/tbdt_v1/build_gold_real_graphs_report.json")),
        "silver_clean_graphs": _load_json(Path("artifacts/tbdt_v1/build_silver_clean_real_graphs_report.json")),
    }
    quality_flags = _quality_flags(
        build_reports=build_reports,
        classification_rows=classification_rows,
        template_summary=template_summary,
        split_leakage=split_leakage,
        strict_input_checks=strict_input_checks,
    )

    hash_paths = [
        Path(args.mixed_manifest),
        Path(args.gold_manifest),
        Path(args.silver_manifest),
        Path(args.bronze_manifest),
        Path("configs/data/tbdt_state.yaml"),
        Path("configs/model/gvp_tbdt_module.yaml"),
        Path("main.py"),
        Path("src/evopoint_da/data/alignment.py"),
        Path("src/evopoint_da/data/datamodule.py"),
        Path("src/evopoint_da/data/dataset.py"),
        Path("src/evopoint_da/data/features.py"),
        Path("src/evopoint_da/data/graph.py"),
        Path("src/evopoint_da/data/structure.py"),
        Path("src/evopoint_da/data/tbdt.py"),
        Path("src/evopoint_da/models/backbones/gvp.py"),
        Path("src/evopoint_da/models/module.py"),
        Path("src/evopoint_da/pipeline/build_features_with_sasa.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_state_dataset.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_template_baselines.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_structure_template_baselines.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_coordinate_baselines.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_external_baselines.py"),
        Path("src/evopoint_da/pipeline/eval_tbdt_state.py"),
        Path("src/evopoint_da/pipeline/tbdt_state_eval_core.py"),
        Path("src/evopoint_da/pipeline/tbdt_state_eval_metrics.py"),
        Path("src/evopoint_da/pipeline/tbdt_state_eval_utils.py"),
        Path("src/evopoint_da/pipeline/eval_tbdt_classification_curves.py"),
        Path("src/evopoint_da/pipeline/run_tbdt_seed_stability.py"),
        Path("src/evopoint_da/pipeline/run_tbdt_report_models.py"),
        Path("src/evopoint_da/pipeline/build_tbdt_publication_report.py"),
        Path(args.test_sample_list),
        *[path for _method, _label, path, _category in _coordinate_metric_specs()],
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_settings.json"),
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_summary.csv"),
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_aggregate.csv"),
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_report.json"),
        Path("artifacts/tbdt_v1/seed_stability_best_selection/seed_stability_report.md"),
        *sorted(Path("artifacts/tbdt_v1/seed_stability_best_selection/metrics").glob("seed_*_best-selection_test.json")),
        Path("artifacts/tbdt_v1/report_models/report_models_report.json"),
        Path("artifacts/tbdt_v1/report_models/report_model_summary.csv"),
        Path("artifacts/tbdt_v1/report_models/prediction_reports/validation_calibrated_region_blend_test.json"),
        Path("artifacts/tbdt_v1/seed_stability/seed_stability_aggregate.csv"),
        Path("artifacts/tbdt_v1/seed_stability_best_disp1to2/seed_stability_aggregate.csv"),
        Path("artifacts/tbdt_v1/seed_stability_best_flex/seed_stability_aggregate.csv"),
        Path("artifacts/tbdt_v1/external_baseline_curves/classification_curve_report.json"),
        Path("artifacts/tbdt_v1/template_baselines/template_baseline_report.json"),
        Path("artifacts/tbdt_v1/template_baselines/external_template_baseline_report.json"),
        Path("artifacts/tbdt_v1/coordinate_baselines/coordinate_baseline_report.json"),
        Path("artifacts/tbdt_v1/external_score_baselines/external_score_baseline_report.json"),
    ]
    reproducibility = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "bootstrap_iter": int(args.bootstrap_iter),
        "bootstrap_seed": int(args.bootstrap_seed),
        "files": _hash_manifest(hash_paths),
    }

    _write_csv(out_dir / "manifest_field_counts.csv", manifest_count_rows)
    _write_csv(out_dir / "graph_region_summary.csv", graph_rows)
    _write_csv(out_dir / "displacement_bin_summary.csv", displacement_bin_rows)
    _write_csv(out_dir / "coordinate_metrics_summary.csv", coordinate_rows)
    _write_csv(out_dir / "coordinate_sample_metrics.csv", coordinate_sample_rows)
    _write_csv(out_dir / "neural_ablation_summary.csv", neural_comparison_rows)
    _write_csv(out_dir / "primary_model_paired_delta_summary.csv", primary_paired_delta_rows)
    _write_csv(out_dir / "primary_model_paired_delta_samples.csv", primary_paired_delta_sample_rows)
    _write_csv(out_dir / "classification_metrics_summary.csv", classification_rows)
    _write_csv(out_dir / "selector_sensitivity_summary.csv", selector_sensitivity_rows)
    _write_json(out_dir / "dataset_summary.json", {"manifests": manifest_summaries, "split_leakage": split_leakage})
    _write_json(out_dir / "strict_input_checks.json", strict_input_checks)
    _write_json(
        out_dir / "graph_summary.json",
        {
            "gold_graphs": graph_summary_gold,
            "silver_clean_graphs": graph_summary_silver,
            "displacement_bins": displacement_bin_rows,
        },
    )
    _write_json(out_dir / "coordinate_metrics_summary.json", coordinate_rows)
    _write_json(out_dir / "neural_ablation_summary.json", neural_comparison_rows)
    _write_json(out_dir / "primary_model_paired_delta_summary.json", primary_paired_delta_rows)
    _write_json(out_dir / "primary_model_paired_delta_samples.json", primary_paired_delta_sample_rows)
    _write_json(out_dir / "classification_metrics_summary.json", classification_rows)
    _write_json(out_dir / "selector_sensitivity_summary.json", selector_sensitivity_rows)
    _write_json(out_dir / "template_selection_summary.json", template_summary)
    _write_json(out_dir / "quality_flags.json", quality_flags)
    _write_json(out_dir / "reproducibility_manifest.json", reproducibility)

    primary_coord = [
        row
        for row in coordinate_rows
        if (
            row["method"]
            in {
                "raw_af2_zero",
                "cooper_tbdt_scaffold_single",
                "cooper_tbdt_scaffold_blend",
                "nearest_template",
                "family_state_average",
            }
            or row.get("category") == "coordinate_baseline"
            or row.get("category") == "external_template_baseline"
        )
        and row["region"] in PRIMARY_REGIONS
    ]
    primary_bins = [
        row
        for row in displacement_bin_rows
        if row.get("scope") == "gold_graphs" and row.get("split") == "test" and row.get("region") in {"eval", "plug", "tonb_box"}
    ]
    primary_class = [row for row in classification_rows if row.get("region") in {"eval", "plug", "tonb_box"}]
    primary_paired_delta_family = [
        row
        for row in primary_paired_delta_rows
        if row.get("aggregation") == "per_target_seed_mean" and row.get("region") in (*PRIMARY_REGIONS, "all")
    ]
    paired_region_order = {region: idx for idx, region in enumerate((*PRIMARY_REGIONS, "all"))}
    primary_paired_delta_family = sorted(
        primary_paired_delta_family,
        key=lambda row: paired_region_order.get(str(row.get("region")), 99),
    )
    markdown = [
        "# Cooper-TBDT v1 Publication Report",
        "",
        "## Dataset",
        "",
        _markdown_table(
            [
                {
                    "name": s["name"],
                    "rows": s["rows"],
                    "unique_uniprot": s["unique_uniprot"],
                    "unique_pdb": s["unique_pdb"],
                    "split_counts": json.dumps(s["split_counts"], sort_keys=True),
                }
                for s in manifest_summaries
            ],
            ["name", "rows", "unique_uniprot", "unique_pdb", "split_counts"],
        ),
        "",
        f"Split leakage check: **{split_leakage['status']}**; overlapping groups: {split_leakage['n_overlapping_groups']}.",
        "",
        "## Strict Input Checks",
        "",
        _markdown_table(strict_input_checks, ["scope", "pair_count", "failed_pair_checks", "status"]),
        "",
        "## Coordinate Metrics",
        "",
        _markdown_table(
            primary_coord,
            [
                "label",
                "region",
                "zero_error_rms",
                "prediction_error_rms",
                "mse_improvement_vs_zero_fraction",
                "sample_improvement_rate",
                "sample_improvement_mean_ci95_low",
                "sample_improvement_mean_ci95_high",
            ],
        ),
        "",
        "## Single-Model Seed Stability",
        "",
        _markdown_table(seed_stability_rows, ["endpoint", "n_seeds", "mean", "std", "min", "max"])
        if seed_stability_rows
        else "Seed stability artifacts not found.",
        "",
        "## Primary Model Paired Delta",
        "",
        "Delta RMSD is `RMSD(model) - RMSD(raw AF2)`. The 5-seed family row averages model RMSD across seeds per target before the paired test, so targets remain the statistical unit.",
        "",
        _markdown_table(
            primary_paired_delta_family,
            [
                "label",
                "region",
                "n_targets",
                "n_improved",
                "n_worsened",
                "median_delta_rmsd_method_minus_raw",
                "median_delta_ci95_low",
                "median_delta_ci95_high",
                "wilcoxon_p_less_method_lt_raw",
                "signed_rank_biserial_effect_method_lt_raw",
            ],
        )
        if primary_paired_delta_family
        else "Primary paired-delta artifacts not found.",
        "",
        "## Selector Sensitivity",
        "",
        _markdown_table(
            selector_sensitivity_rows,
            ["selector", "endpoint", "n_seeds", "mean", "std", "min", "max"],
        )
        if selector_sensitivity_rows
        else "Selector sensitivity artifacts not found.",
        "",
        "## Neural Ablations",
        "",
        _markdown_table(
            neural_comparison_rows,
            [
                "label",
                "category",
                "region",
                "zero_error_rms",
                "prediction_error_rms",
                "mse_improvement_vs_zero_fraction",
                "predicted_displacement_mean",
            ],
        )
        if neural_comparison_rows
        else "Neural ablation artifacts not found.",
        "",
        "## Gold Test Displacement Bins",
        "",
        _markdown_table(
            primary_bins,
            ["region", "bin", "n_residues", "region_residue_total", "fraction"],
        ),
        "",
        "## Residue Localization",
        "",
        _markdown_table(primary_class, ["label", "region", "auroc", "average_precision", "n_positive", "n_negative"]),
        "",
        "## Template Baseline Coverage",
        "",
        "Nearest-template target coverage: "
        + json.dumps(template_summary.get("nearest_target_coverage", {}), sort_keys=True),
        "",
        "## Quality Flags",
        "",
        _markdown_table(quality_flags, ["severity", "scope", "issue"]) if quality_flags else "No quality flags.",
        "",
        "## Artifact Paths",
        "",
        "- Coordinate summary: `coordinate_metrics_summary.csv`",
        "- Seed stability: `../seed_stability_best_selection/seed_stability_report.md`",
        "- Primary model paired delta: `primary_model_paired_delta_summary.csv`",
        "- Selector sensitivity: `selector_sensitivity_summary.csv`",
        "- Displacement bins: `displacement_bin_summary.csv`",
        "- Classification summary: `classification_metrics_summary.csv`",
        "- Dataset summary: `dataset_summary.json`",
        "- Strict input checks: `strict_input_checks.json`",
        "- Reproducibility manifest: `reproducibility_manifest.json`",
    ]
    report_md = out_dir / "publication_report.md"
    report_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    report = {
        "out_dir": str(out_dir),
        "publication_report_md": str(report_md),
        "manifest_summaries": manifest_summaries,
        "split_leakage": split_leakage,
        "strict_input_checks": strict_input_checks,
        "graph_summaries": {"gold_graphs": graph_summary_gold, "silver_clean_graphs": graph_summary_silver},
        "coordinate_metric_rows": len(coordinate_rows),
        "seed_stability_rows": len(seed_stability_rows),
        "primary_paired_delta_rows": len(primary_paired_delta_rows),
        "selector_sensitivity_rows": len(selector_sensitivity_rows),
        "neural_comparison_rows": len(neural_comparison_rows),
        "displacement_bin_rows": len(displacement_bin_rows),
        "classification_metric_rows": len(classification_rows),
        "template_selection_summary": template_summary,
        "quality_flags": quality_flags,
    }
    _write_json(out_dir / "publication_report_index.json", report)
    return report


def main() -> None:
    report = build_report(parse_args())
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
