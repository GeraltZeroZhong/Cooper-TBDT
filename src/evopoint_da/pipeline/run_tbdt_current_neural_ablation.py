"""Run current-code neural ablations for TBDT state displacement.

The experiment is intentionally narrower than the historical scaffold sweep:
each variant changes one interpretable mechanism relative to the final
scaffold-prior recipe, uses the fixed Gold metadata split, selects checkpoints
only on validation ``best-selection``, and evaluates held-out Gold test metrics
with the same region-vector evaluator used by the publication tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evopoint_da.data.dataset import build_split_file_lists
from evopoint_da.pipeline.eval_tbdt_state import evaluate as evaluate_regions
from evopoint_da.pipeline.predict_tbdt_graphs import predict_graphs


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}
REGIONS = ("eval", "plug", "tonb_box", "barrel_core", "all")
SELECTOR = "best-selection"

FINAL_SINGLE_OVERRIDES: dict[str, Any] = {
    "model.lr": "3e-4",
    "model.barrel_core_loss_weight": 0.05,
    "model.eval_region_loss_weight": 2.0,
    "model.plug_loss_weight": 2.5,
    "model.extracellular_loop_loss_weight": 1.0,
    "model.tonb_box_loss_weight": 4.0,
    "model.substrate_contact_loss_weight": 4.0,
    "model.scaffold_anchor_weight": 0.2,
    "model.scaffold_anchor_plddt_min": 80.0,
    "model.coord_init_gain": 0.01,
    "model.output_scale": 2.0,
    "model.gvp_dropout": 0.05,
    "model.zero_node_scalar_feature_indices": [],
    "model.zero_edge_scalar_feature_indices": [],
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    category: str
    question: str
    overrides: dict[str, Any]


def _specs() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec(
            name="full_scaffold_prior",
            label="Full scaffold-prior model",
            category="reference",
            question="Final single-model recipe with all metadata conditioning and AFDB confidence features.",
            overrides=FINAL_SINGLE_OVERRIDES,
        ),
        AblationSpec(
            name="no_state_conditioning",
            label="No state conditioning",
            category="conditioning",
            question="Is the graph-level experimental state label useful beyond sequence, structure, region, family, and substrate labels?",
            overrides={**FINAL_SINGLE_OVERRIDES, "model.state_embedding_dim": 0},
        ),
        AblationSpec(
            name="no_region_conditioning",
            label="No region conditioning",
            category="conditioning",
            question="Is the residue-level region embedding useful beyond region-weighted supervision and structural features?",
            overrides={**FINAL_SINGLE_OVERRIDES, "model.region_embedding_dim": 0},
        ),
        AblationSpec(
            name="no_graph_metadata_conditioning",
            label="No graph metadata conditioning",
            category="conditioning",
            question="Do graph-level family, state, and substrate embeddings add signal beyond residue region conditioning?",
            overrides={
                **FINAL_SINGLE_OVERRIDES,
                "model.family_embedding_dim": 0,
                "model.state_embedding_dim": 0,
                "model.substrate_embedding_dim": 0,
            },
        ),
        AblationSpec(
            name="no_tbdt_conditioning",
            label="No TBDT conditioning",
            category="conditioning",
            question="Does any learned TBDT metadata conditioning help relative to raw node and edge graph features?",
            overrides={
                **FINAL_SINGLE_OVERRIDES,
                "model.use_tbdt_conditioning": False,
                "model.in_channels": 144,
                "model.condition_embedding_dim": 0,
                "model.region_embedding_dim": 0,
                "model.family_embedding_dim": 0,
                "model.state_embedding_dim": 0,
                "model.substrate_embedding_dim": 0,
            },
        ),
        AblationSpec(
            name="no_scaffold_anchor",
            label="No scaffold anchor",
            category="scaffold_prior",
            question="Does the high-pLDDT barrel-core zero-displacement anchor preserve the scaffold without hurting functional regions?",
            overrides={**FINAL_SINGLE_OVERRIDES, "model.scaffold_anchor_weight": 0.0},
        ),
        AblationSpec(
            name="no_region_loss_weights",
            label="No region loss weights",
            category="loss_weighting",
            question="Do region-specific supervised loss weights matter when the scaffold anchor is retained?",
            overrides={
                **FINAL_SINGLE_OVERRIDES,
                "model.barrel_core_loss_weight": 1.0,
                "model.eval_region_loss_weight": 1.0,
                "model.plug_loss_weight": 1.0,
                "model.extracellular_loop_loss_weight": 1.0,
                "model.tonb_box_loss_weight": 1.0,
                "model.substrate_contact_loss_weight": 1.0,
            },
        ),
        AblationSpec(
            name="no_scaffold_prior",
            label="No scaffold prior",
            category="loss_weighting",
            question="What happens when both region-weighted supervision and the explicit scaffold anchor are removed?",
            overrides={
                **FINAL_SINGLE_OVERRIDES,
                "model.barrel_core_loss_weight": 1.0,
                "model.eval_region_loss_weight": 1.0,
                "model.plug_loss_weight": 1.0,
                "model.extracellular_loop_loss_weight": 1.0,
                "model.tonb_box_loss_weight": 1.0,
                "model.substrate_contact_loss_weight": 1.0,
                "model.scaffold_anchor_weight": 0.0,
            },
        ),
        AblationSpec(
            name="no_plddt_node_feature",
            label="No node pLDDT feature",
            category="confidence_features",
            question="Does the normalized AFDB pLDDT node scalar provide useful confidence information?",
            overrides={**FINAL_SINGLE_OVERRIDES, "model.zero_node_scalar_feature_indices": [128]},
        ),
        AblationSpec(
            name="no_pae_edge_features",
            label="No PAE edge features",
            category="confidence_features",
            question="Do AFDB PAE-derived edge uncertainty scalars improve state-displacement prediction?",
            overrides={**FINAL_SINGLE_OVERRIDES, "model.zero_edge_scalar_feature_indices": [16, 17]},
        ),
        AblationSpec(
            name="no_afdb_confidence_features",
            label="No AFDB confidence features",
            category="confidence_features",
            question="Does the model need explicit AFDB pLDDT and PAE confidence features when geometry and sequence embeddings remain?",
            overrides={
                **FINAL_SINGLE_OVERRIDES,
                "model.zero_node_scalar_feature_indices": [128],
                "model.zero_edge_scalar_feature_indices": [16, 17],
            },
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/current_neural_ablation")
    parser.add_argument("--study-prefix", default="tbdt_current_neural_ablation")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--seed", action="append", type=int, default=None, help="Training seed. May repeat.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default=",".join(spec.name for spec in _specs()))
    parser.add_argument("--eval-bootstrap-iter", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-zero-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _parse_variants(text: str) -> list[str]:
    names = [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
    known = {spec.name for spec in _specs()}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown ablation variant(s): {unknown}. Known variants: {sorted(known)}")
    return names


def _stringify_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        text = "[" + ",".join(str(item) for item in value) + "]"
    elif value is None:
        text = "null"
    else:
        text = str(value)
    return f"{key}={text}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _study_name(prefix: str, variant: str, seed: int) -> str:
    return f"{prefix}/{variant}/seed_{int(seed)}"


def _run_train(spec: AblationSpec, seed: int, args: argparse.Namespace, log_path: Path) -> None:
    study_name = _study_name(args.study_prefix, spec.name, seed)
    cmd = [
        sys.executable,
        "train.py",
        "data=tbdt_state",
        "model=gvp_tbdt_module",
        f"trainer.max_epochs={int(args.max_epochs)}",
        f"data.data_dir={args.data_dir}",
        "data.batch_size=1",
        "data.num_workers=0",
        f"data.split_seed={int(args.split_seed)}",
        f"data.split_source={args.split_source}",
        f"seed={int(seed)}",
        f"study_name={study_name}",
        f"logger.save_dir=logs/{study_name}",
        "+run_post_train_tests=false",
        "+trainer.deterministic=true",
        "+trainer.benchmark=false",
        *[_stringify_override(key, value) for key, value in spec.overrides.items()],
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(int(seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _has_checkpoint(study_name: str) -> bool:
    root = Path("checkpoints") / study_name
    return root.exists() and any(root.glob(f"*/{SELECTOR}-*.ckpt"))


def _latest_run_dir(study_name: str) -> Path:
    root = Path("checkpoints") / study_name
    if not root.exists():
        raise FileNotFoundError(f"No checkpoint directory found for {study_name}: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directories under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob(f"{SELECTOR}-*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No {SELECTOR} checkpoint found in {run_dir}")
    return matches[-1]


def _selected_epoch_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"-(\d+)-[0-9.]+\.ckpt$", path.name)
    return int(match.group(1)) if match else None


def _split_files(data_dir: str, split: str, args: argparse.Namespace) -> list[str]:
    splits = build_split_file_lists(
        root=data_dir,
        split_ranges=DEFAULT_SPLIT_RANGES,
        split_seed=int(args.split_seed),
        split_source=args.split_source,
    )
    return splits.get(split, [])


def _predict_and_eval(
    spec: AblationSpec,
    seed: int,
    ckpt: Path,
    split: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prefix = f"{spec.name}_seed{int(seed)}_{split}"
    pred_dir = out_dir / "predictions" / prefix
    metrics_json = out_dir / "metrics" / f"{prefix}.json"
    metrics_csv = out_dir / "metrics" / f"{prefix}.csv"
    paired_delta_csv = out_dir / "paired_delta" / f"{prefix}.csv"
    tonb_metrics_csv = out_dir / "tonb_state" / f"{prefix}.csv"
    if args.force or not pred_dir.exists():
        predict_graphs(
            argparse.Namespace(
                ckpt=str(ckpt),
                data_dir=args.data_dir,
                output_dir=str(pred_dir),
                split=split,
                split_source=args.split_source,
                split_seed=int(args.split_seed),
                batch_size=1,
                num_workers=0,
                device=args.device,
                report_path=str(out_dir / "prediction_reports" / f"{prefix}.json"),
            )
        )
    return evaluate_regions(
        argparse.Namespace(
            inputs=_split_files(args.data_dir, split, args),
            predictions=str(pred_dir),
            output_json=str(metrics_json),
            output_csv=str(metrics_csv),
            region_json=None,
            include_all_region=True,
            direction_threshold=1.0,
            add_derived_regions=True,
            plug_apical_fraction=0.35,
            plug_extension_residues=12,
            bootstrap_iter=int(args.eval_bootstrap_iter),
            bootstrap_seed=42,
            paired_delta_csv=str(paired_delta_csv),
            tonb_metrics_csv=str(tonb_metrics_csv),
            tonb_exposure_threshold=1.0,
        )
    )


def _evaluate_zero(split: str, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics_json = out_dir / "metrics" / f"raw_afdb_zero_{split}.json"
    metrics_csv = out_dir / "metrics" / f"raw_afdb_zero_{split}.csv"
    paired_delta_csv = out_dir / "paired_delta" / f"raw_afdb_zero_{split}.csv"
    tonb_metrics_csv = out_dir / "tonb_state" / f"raw_afdb_zero_{split}.csv"
    return evaluate_regions(
        argparse.Namespace(
            inputs=_split_files(args.data_dir, split, args),
            predictions=None,
            output_json=str(metrics_json),
            output_csv=str(metrics_csv),
            region_json=None,
            include_all_region=True,
            direction_threshold=1.0,
            add_derived_regions=True,
            plug_apical_fraction=0.35,
            plug_extension_residues=12,
            bootstrap_iter=int(args.eval_bootstrap_iter),
            bootstrap_seed=42,
            paired_delta_csv=str(paired_delta_csv),
            tonb_metrics_csv=str(tonb_metrics_csv),
            tonb_exposure_threshold=1.0,
        )
    )


def _region_metric(report: dict[str, Any], region: str, metric: str) -> float:
    try:
        return float(report["aggregate_by_region"][region][metric])
    except Exception:
        return float("nan")


def _paired_metric(report: dict[str, Any], region: str, metric: str) -> float:
    try:
        return float(report["paired_delta_by_region"][region][metric])
    except Exception:
        return float("nan")


def _summary_row(
    *,
    spec: AblationSpec | None,
    seed: int | None,
    split: str,
    report: dict[str, Any],
    checkpoint: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": spec.name if spec is not None else "raw_afdb_zero",
        "label": spec.label if spec is not None else "Raw AFDB / zero displacement",
        "category": spec.category if spec is not None else "baseline",
        "question": spec.question if spec is not None else "Natural zero-displacement baseline in the aligned AFDB frame.",
        "seed": seed if seed is not None else "",
        "split": split,
        "selector": SELECTOR if checkpoint is not None else "",
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "selected_epoch": _selected_epoch_from_checkpoint(checkpoint) if checkpoint is not None else "",
        "max_epochs": int(args.max_epochs) if checkpoint is not None else "",
        "split_source": args.split_source,
        "split_seed": int(args.split_seed),
        "data_dir": args.data_dir,
    }
    if spec is not None:
        for key, value in spec.overrides.items():
            row[key] = value
    for region in REGIONS:
        for metric in (
            "n_residues",
            "sample_count",
            "zero_error_rms",
            "prediction_error_rms",
            "target_displacement_rms",
            "predicted_displacement_mean",
            "mse_improvement_vs_zero_fraction",
            "better_than_zero_rate",
            "sample_improvement_rate",
            "sample_improvement_median",
            "direction_cosine_mean",
            "magnitude_mae",
        ):
            row[f"{region}_{metric}"] = _region_metric(report, region, metric)
        for metric in (
            "n_targets",
            "n_improved",
            "n_worsened",
            "n_tied",
            "improved_fraction",
            "median_delta_rmsd_method_minus_raw",
            "mean_delta_rmsd_method_minus_raw",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "median_delta_ci95_low",
            "median_delta_ci95_high",
            "wilcoxon_n_nonzero",
            "wilcoxon_p_less_method_lt_raw",
            "wilcoxon_p_two_sided",
            "signed_rank_biserial_effect_method_lt_raw",
        ):
            row[f"{region}_paired_{metric}"] = _paired_metric(report, region, metric)
    tonb_summary = report.get("tonb_state_summary") or {}
    for key, value in tonb_summary.items():
        row[f"tonb_state_{key}"] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_seed_aggregate(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        key
        for row in rows
        if row.get("split") == "test"
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"seed", "split_seed"}
    ]
    numeric_keys = list(dict.fromkeys(numeric_keys))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "test" or row.get("variant") == "raw_afdb_zero":
            continue
        grouped.setdefault((str(row["variant"]), str(row["label"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (variant, label), group_rows in sorted(grouped.items()):
        out: dict[str, Any] = {"variant": variant, "label": label, "split": "test", "n_seeds": len(group_rows)}
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if key in row and math.isfinite(float(row[key]))]
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                sd = math.sqrt(variance)
            else:
                sd = float("nan")
            out[f"{key}_mean"] = mean
            out[f"{key}_sd"] = sd
        aggregate_rows.append(out)
    _write_csv(path, aggregate_rows)
    return aggregate_rows


def main() -> None:
    args = parse_args()
    seeds = args.seed or [42]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(_parse_variants(args.variants))
    specs = [spec for spec in _specs() if spec.name in wanted]

    (out_dir / "ablation_protocol.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(out_dir),
                    "study_prefix": args.study_prefix,
                    "data_dir": args.data_dir,
                    "split_source": args.split_source,
                    "split_seed": int(args.split_seed),
                    "seeds": seeds,
                    "max_epochs": int(args.max_epochs),
                    "selector": SELECTOR,
                    "feature_indices": {
                        "node_128": "normalized AFDB pLDDT in 144-dimensional node scalar input",
                        "edge_16": "normalized PAE edge scalar after 16 distance RBF features",
                        "edge_17": "PAE confidence edge scalar after normalized PAE",
                    },
                    "variants": [
                        {
                            "name": spec.name,
                            "label": spec.label,
                            "category": spec.category,
                            "question": spec.question,
                            "overrides": spec.overrides,
                        }
                        for spec in specs
                    ],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    if args.include_zero_baseline:
        for split in ("val", "test"):
            report = _evaluate_zero(split, out_dir, args)
            rows.append(_summary_row(spec=None, seed=None, split=split, report=report, checkpoint=None, args=args))

    for spec in specs:
        for seed in seeds:
            study_name = _study_name(args.study_prefix, spec.name, seed)
            if not args.skip_train and (args.force or not _has_checkpoint(study_name)):
                _run_train(
                    spec,
                    seed,
                    args,
                    log_path=out_dir / "train_logs" / spec.name / f"seed_{int(seed)}.log",
                )
            run_dir = _latest_run_dir(study_name)
            ckpt = _checkpoint(run_dir)
            for split in ("val", "test"):
                report = _predict_and_eval(spec, seed, ckpt, split, out_dir, args)
                rows.append(_summary_row(spec=spec, seed=seed, split=split, report=report, checkpoint=ckpt, args=args))
            _write_csv(out_dir / "ablation_summary.csv", rows)
            (out_dir / "ablation_summary.json").write_text(
                json.dumps(_json_safe(rows), indent=2),
                encoding="utf-8",
            )
            aggregate_rows = _write_seed_aggregate(out_dir / "ablation_seed_aggregate.csv", rows)
            (out_dir / "ablation_seed_aggregate.json").write_text(
                json.dumps(_json_safe(aggregate_rows), indent=2),
                encoding="utf-8",
            )

    report = {
        "out_dir": str(out_dir),
        "summary_csv": str(out_dir / "ablation_summary.csv"),
        "summary_json": str(out_dir / "ablation_summary.json"),
        "seed_aggregate_csv": str(out_dir / "ablation_seed_aggregate.csv"),
        "seed_aggregate_json": str(out_dir / "ablation_seed_aggregate.json"),
        "n_rows": len(rows),
        "variants": [spec.name for spec in specs],
        "seeds": seeds,
    }
    (out_dir / "ablation_report.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
