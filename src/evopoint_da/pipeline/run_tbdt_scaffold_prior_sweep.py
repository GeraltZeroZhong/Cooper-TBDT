"""Run scaffold-prior ablations and tuning experiments for TBDT.

The sweep trains variants with validation-selected checkpoints, exports
val/test predictions, evaluates region-vector metrics, and writes a compact
summary for mechanistic reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    description: str
    overrides: dict[str, Any]


def _variants() -> list[Variant]:
    neutral = {
        "model.barrel_core_loss_weight": 1.0,
        "model.eval_region_loss_weight": 1.0,
        "model.plug_loss_weight": 1.0,
        "model.extracellular_loop_loss_weight": 1.0,
        "model.tonb_box_loss_weight": 1.0,
        "model.substrate_contact_loss_weight": 1.0,
        "model.scaffold_anchor_weight": 0.0,
        "model.scaffold_anchor_plddt_min": 80.0,
    }
    current_weights = {
        "model.barrel_core_loss_weight": 0.1,
        "model.eval_region_loss_weight": 1.5,
        "model.plug_loss_weight": 2.0,
        "model.extracellular_loop_loss_weight": 1.0,
        "model.tonb_box_loss_weight": 3.0,
        "model.substrate_contact_loss_weight": 3.0,
    }
    return [
        Variant("sp00_no_prior", "mechanism", "No region weights and no scaffold anchor.", neutral),
        Variant(
            "sp01_anchor_only_w01_t80",
            "mechanism",
            "High-pLDDT barrel anchor only; no region loss reweighting.",
            {**neutral, "model.scaffold_anchor_weight": 0.1},
        ),
        Variant(
            "sp02_core_downweight_only",
            "mechanism",
            "Only barrel-core supervised loss down-weighting.",
            {**neutral, "model.barrel_core_loss_weight": 0.1},
        ),
        Variant(
            "sp03_region_weights_only",
            "mechanism",
            "Functional region weights without explicit scaffold anchor.",
            {**current_weights, "model.scaffold_anchor_weight": 0.0, "model.scaffold_anchor_plddt_min": 80.0},
        ),
        Variant(
            "sp04_core_downweight_anchor",
            "mechanism",
            "Barrel-core down-weighting plus high-pLDDT scaffold anchor.",
            {**neutral, "model.barrel_core_loss_weight": 0.1, "model.scaffold_anchor_weight": 0.1},
        ),
        Variant(
            "sp05_region_anchor_w005_t80",
            "anchor_strength",
            "Current region weights with weak scaffold anchor.",
            {**current_weights, "model.scaffold_anchor_weight": 0.05, "model.scaffold_anchor_plddt_min": 80.0},
        ),
        Variant(
            "sp06_region_anchor_w01_t80",
            "anchor_strength",
            "Current region weights and current scaffold anchor.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.scaffold_anchor_plddt_min": 80.0},
        ),
        Variant(
            "sp07_region_anchor_w02_t80",
            "anchor_strength",
            "Current region weights with stronger scaffold anchor.",
            {**current_weights, "model.scaffold_anchor_weight": 0.2, "model.scaffold_anchor_plddt_min": 80.0},
        ),
        Variant(
            "sp08_region_anchor_w01_t70",
            "anchor_threshold",
            "Current region weights; anchor includes pLDDT >= 70 barrel residues.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.scaffold_anchor_plddt_min": 70.0},
        ),
        Variant(
            "sp09_region_anchor_w01_t90",
            "anchor_threshold",
            "Current region weights; anchor only pLDDT >= 90 barrel residues.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.scaffold_anchor_plddt_min": 90.0},
        ),
        Variant(
            "sp10_moderate_region_anchor",
            "region_weight",
            "Moderate functional weights with scaffold anchor.",
            {
                "model.barrel_core_loss_weight": 0.2,
                "model.eval_region_loss_weight": 1.2,
                "model.plug_loss_weight": 1.5,
                "model.extracellular_loop_loss_weight": 1.0,
                "model.tonb_box_loss_weight": 2.0,
                "model.substrate_contact_loss_weight": 2.0,
                "model.scaffold_anchor_weight": 0.1,
                "model.scaffold_anchor_plddt_min": 80.0,
            },
        ),
        Variant(
            "sp11_tonb_strong_anchor",
            "region_weight",
            "Stronger TonB/substrate-contact weighting with scaffold anchor.",
            {
                **current_weights,
                "model.tonb_box_loss_weight": 4.0,
                "model.substrate_contact_loss_weight": 4.0,
                "model.scaffold_anchor_weight": 0.1,
                "model.scaffold_anchor_plddt_min": 80.0,
            },
        ),
        Variant(
            "sp12_low_lr_region_anchor",
            "optimizer",
            "Current prior with lower learning rate.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.lr": "2e-4"},
        ),
        Variant(
            "sp13_outscale15_region_anchor",
            "head_scale",
            "Current prior with lower output scale.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.output_scale": 1.5},
        ),
        Variant(
            "sp14_dropout10_region_anchor",
            "regularization",
            "Current prior with stronger GVP dropout.",
            {**current_weights, "model.scaffold_anchor_weight": 0.1, "model.gvp_dropout": 0.1},
        ),
        Variant(
            "sp15_eval_only_anchor",
            "region_component",
            "Broad evaluation-region weighting with core down-weighting and scaffold anchor.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.eval_region_loss_weight": 1.5,
                "model.scaffold_anchor_weight": 0.1,
            },
        ),
        Variant(
            "sp16_plug_only_anchor",
            "region_component",
            "Plug-domain weighting with core down-weighting and scaffold anchor.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.plug_loss_weight": 2.0,
                "model.scaffold_anchor_weight": 0.1,
            },
        ),
        Variant(
            "sp17_tonb_only_anchor",
            "region_component",
            "TonB-box weighting with core down-weighting and scaffold anchor.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.tonb_box_loss_weight": 3.0,
                "model.scaffold_anchor_weight": 0.1,
            },
        ),
        Variant(
            "sp18_substrate_only_anchor",
            "region_component",
            "Substrate-contact weighting with core down-weighting and scaffold anchor.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.substrate_contact_loss_weight": 3.0,
                "model.scaffold_anchor_weight": 0.1,
            },
        ),
        Variant(
            "sp19_specific_regions_anchor",
            "region_component",
            "Plug, TonB-box, and substrate-contact weights without broad eval-region upweighting.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.plug_loss_weight": 2.0,
                "model.tonb_box_loss_weight": 3.0,
                "model.substrate_contact_loss_weight": 3.0,
                "model.scaffold_anchor_weight": 0.1,
            },
        ),
        Variant(
            "sp20_full_region_no_core_downweight",
            "region_component",
            "Current region weights and scaffold anchor but no barrel-core supervised down-weighting.",
            {**current_weights, "model.barrel_core_loss_weight": 1.0, "model.scaffold_anchor_weight": 0.1},
        ),
        Variant(
            "sp21_core005_anchor",
            "core_weight",
            "Stronger barrel-core supervised down-weighting plus scaffold anchor.",
            {**neutral, "model.barrel_core_loss_weight": 0.05, "model.scaffold_anchor_weight": 0.1},
        ),
        Variant(
            "sp22_core02_anchor",
            "core_weight",
            "Milder barrel-core supervised down-weighting plus scaffold anchor.",
            {**neutral, "model.barrel_core_loss_weight": 0.2, "model.scaffold_anchor_weight": 0.1},
        ),
        Variant(
            "sp23_core_anchor_w005",
            "core_anchor_strength",
            "Core down-weighting with weak scaffold anchor and no functional region weights.",
            {**neutral, "model.barrel_core_loss_weight": 0.1, "model.scaffold_anchor_weight": 0.05},
        ),
        Variant(
            "sp24_core_anchor_w02",
            "core_anchor_strength",
            "Core down-weighting with stronger scaffold anchor and no functional region weights.",
            {**neutral, "model.barrel_core_loss_weight": 0.1, "model.scaffold_anchor_weight": 0.2},
        ),
        Variant(
            "sp25_core_anchor_lr2e4",
            "optimizer",
            "Core down-weighting and scaffold anchor with lower learning rate.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.scaffold_anchor_weight": 0.1,
                "model.lr": "2e-4",
            },
        ),
        Variant(
            "sp26_core_anchor_lr1e4",
            "optimizer",
            "Core down-weighting and scaffold anchor with conservative learning rate.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.scaffold_anchor_weight": 0.1,
                "model.lr": "1e-4",
            },
        ),
        Variant(
            "sp27_core_anchor_outscale15",
            "head_scale",
            "Core down-weighting and scaffold anchor with lower output scale.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.scaffold_anchor_weight": 0.1,
                "model.output_scale": 1.5,
            },
        ),
        Variant(
            "sp28_core_anchor_dropout10",
            "regularization",
            "Core down-weighting and scaffold anchor with stronger GVP dropout.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.scaffold_anchor_weight": 0.1,
                "model.gvp_dropout": 0.1,
            },
        ),
        Variant(
            "sp29_region_anchor_w005_lr2e4",
            "winner_focus",
            "Current region weights with weak scaffold anchor and lower learning rate.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.lr": "2e-4",
            },
        ),
        Variant(
            "sp30_region_anchor_w005_lr1e4",
            "winner_focus",
            "Current region weights with weak scaffold anchor and conservative learning rate.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.lr": "1e-4",
            },
        ),
        Variant(
            "sp31_region_anchor_w005_dropout10",
            "winner_focus",
            "Current region weights with weak scaffold anchor and stronger GVP dropout.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.gvp_dropout": 0.1,
            },
        ),
        Variant(
            "sp32_region_anchor_w005_outscale15",
            "winner_focus",
            "Current region weights with weak scaffold anchor and lower output scale.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.output_scale": 1.5,
            },
        ),
        Variant(
            "sp33_region_anchor_w005_lr2e4_dropout10",
            "winner_focus",
            "Current region weights with weak scaffold anchor, lower learning rate, and stronger dropout.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.lr": "2e-4",
                "model.gvp_dropout": 0.1,
            },
        ),
        Variant(
            "sp34_region_anchor_w005_lr2e4_outscale15",
            "winner_focus",
            "Current region weights with weak scaffold anchor, lower learning rate, and lower output scale.",
            {
                **current_weights,
                "model.scaffold_anchor_weight": 0.05,
                "model.scaffold_anchor_plddt_min": 80.0,
                "model.lr": "2e-4",
                "model.output_scale": 1.5,
            },
        ),
        Variant(
            "sp35_tonb_only_anchor_lr1e4",
            "winner_focus",
            "TonB-box weighting with core down-weighting, scaffold anchor, and conservative learning rate.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.tonb_box_loss_weight": 3.0,
                "model.scaffold_anchor_weight": 0.1,
                "model.lr": "1e-4",
            },
        ),
        Variant(
            "sp36_tonb_only_anchor_w005",
            "winner_focus",
            "TonB-box weighting with core down-weighting and weak scaffold anchor.",
            {
                **neutral,
                "model.barrel_core_loss_weight": 0.1,
                "model.tonb_box_loss_weight": 3.0,
                "model.scaffold_anchor_weight": 0.05,
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaffold-prior ablation and tuning sweep.")
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/scaffold_prior_sweep")
    parser.add_argument("--study-prefix", default="tbdt_scaffold_prior_sweep")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--only", action="append", default=[], help="Variant name to run. May be repeated.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoints when possible.")
    parser.add_argument("--force", action="store_true", help="Rerun training/prediction even if outputs exist.")
    parser.add_argument(
        "--checkpoint-selector",
        action="append",
        default=[],
        choices=["best-disp1to5", "best-flex", "best-selection", "best-disp1to2"],
    )
    return parser.parse_args()


def _stringify_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"{key}={text}"


def _run_train(variant: Variant, args: argparse.Namespace, log_path: Path) -> None:
    study_name = f"{args.study_prefix}/{variant.name}"
    cmd = [
        sys.executable,
        "train.py",
        "data=tbdt_state",
        "model=gvp_tbdt_module",
        f"trainer.max_epochs={int(args.max_epochs)}",
        "data.batch_size=1",
        f"seed={int(args.seed)}",
        f"study_name={study_name}",
        f"logger.save_dir=logs/{args.study_prefix}/{variant.name}",
        "+run_post_train_tests=false",
        *[_stringify_override(key, value) for key, value in variant.overrides.items()],
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _latest_run_dir(study_prefix: str, variant_name: str) -> Path:
    root = Path("checkpoints") / study_prefix / variant_name
    if not root.exists():
        raise FileNotFoundError(f"No checkpoint directory found for {study_prefix}/{variant_name}: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directories under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint(run_dir: Path, selector: str) -> Path:
    matches = sorted(run_dir.glob(f"{selector}-*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No {selector} checkpoint found in {run_dir}")
    return matches[-1]


def _split_files(args: argparse.Namespace, split: str) -> list[str]:
    splits = build_split_file_lists(
        root=args.data_dir,
        split_ranges=DEFAULT_SPLIT_RANGES,
        split_seed=int(args.seed),
        split_source=args.split_source,
    )
    return splits.get(split, [])


def _predict_and_eval(
    *,
    variant: Variant,
    selector: str,
    ckpt: Path,
    split: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    pred_dir = out_dir / "predictions" / f"{variant.name}_{selector}_{split}"
    metrics_json = out_dir / "metrics" / f"{variant.name}_{selector}_{split}.json"
    metrics_csv = out_dir / "metrics" / f"{variant.name}_{selector}_{split}.csv"
    if not pred_dir.exists() or args.force:
        predict_graphs(
            argparse.Namespace(
                ckpt=str(ckpt),
                data_dir=args.data_dir,
                output_dir=str(pred_dir),
                split=split,
                split_source=args.split_source,
                split_seed=int(args.seed),
                batch_size=1,
                num_workers=0,
                device=args.device,
                report_path=str(out_dir / "prediction_reports" / f"{variant.name}_{selector}_{split}.json"),
            )
        )
    sample_files = _split_files(args, split)
    report = evaluate_regions(
        argparse.Namespace(
            inputs=sample_files,
            predictions=str(pred_dir),
            output_json=str(metrics_json),
            output_csv=str(metrics_csv),
            region_json=None,
            include_all_region=True,
            direction_threshold=1.0,
        )
    )
    return report


def _region_metric(report: dict[str, Any], region: str, key: str, default: float = float("nan")) -> float:
    try:
        value = report["aggregate_by_region"][region][key]
        return float(value)
    except Exception:
        return default


def _score(report: dict[str, Any]) -> float:
    functional = np_mean(
        [
            _region_metric(report, "eval", "mse_improvement_vs_zero_fraction"),
            _region_metric(report, "plug", "mse_improvement_vs_zero_fraction"),
            _region_metric(report, "tonb_box", "mse_improvement_vs_zero_fraction"),
        ]
    )
    core_pred = _region_metric(report, "barrel_core", "predicted_displacement_mean", 0.0)
    core_penalty = max(0.0, core_pred - 0.05)
    return functional - 0.25 * core_penalty


def np_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _summary_row(
    variant: Variant,
    selector: str,
    split: str,
    report: dict[str, Any],
    ckpt: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant.name,
        "group": variant.group,
        "description": variant.description,
        "selector": selector,
        "split": split,
        "checkpoint": str(ckpt),
        "score": _score(report),
        "max_epochs": int(args.max_epochs),
    }
    for key, value in variant.overrides.items():
        row[key] = value
    for region in REGIONS:
        for metric in [
            "prediction_error_rms",
            "zero_error_rms",
            "mse_improvement_vs_zero_fraction",
            "predicted_displacement_mean",
            "sample_improvement_rate",
            "direction_cosine_mean",
            "magnitude_mae",
        ]:
            row[f"{region}_{metric}"] = _region_metric(report, region, metric)
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selectors = args.checkpoint_selector or ["best-disp1to5", "best-flex"]
    variants = _variants()
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted.difference({variant.name for variant in variants})
        if missing:
            raise ValueError(f"Unknown variant(s): {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    variant_specs = [
        {
            "name": variant.name,
            "group": variant.group,
            "description": variant.description,
            "overrides": variant.overrides,
        }
        for variant in variants
    ]
    (out_dir / "variant_specs.json").write_text(json.dumps(variant_specs, indent=2), encoding="utf-8")

    for idx, variant in enumerate(variants, start=1):
        print(f"[{idx}/{len(variants)}] {variant.name}: {variant.description}", flush=True)
        log_path = out_dir / "train_logs" / f"{variant.name}.log"
        if not args.skip_train:
            if args.force or not (Path("checkpoints") / args.study_prefix / variant.name).exists():
                _run_train(variant, args, log_path)
            else:
                print(f"  reuse existing checkpoint directory for {variant.name}", flush=True)
        run_dir = _latest_run_dir(args.study_prefix, variant.name)
        for selector in selectors:
            ckpt = _checkpoint(run_dir, selector)
            for split in ("val", "test"):
                report = _predict_and_eval(
                    variant=variant,
                    selector=selector,
                    ckpt=ckpt,
                    split=split,
                    args=args,
                    out_dir=out_dir,
                )
                rows.append(_summary_row(variant, selector, split, report, ckpt, args))
        _write_csv(out_dir / "scaffold_prior_sweep_summary.csv", rows)
        (out_dir / "scaffold_prior_sweep_summary.json").write_text(
            json.dumps(_json_safe(rows), indent=2),
            encoding="utf-8",
        )

    val_rows = [row for row in rows if row["split"] == "val"]
    best_val = sorted(val_rows, key=lambda row: float(row["score"]), reverse=True)[:10]
    report = {
        "out_dir": str(out_dir),
        "study_prefix": args.study_prefix,
        "max_epochs": int(args.max_epochs),
        "selectors": selectors,
        "n_rows": len(rows),
        "best_val": best_val,
    }
    (out_dir / "scaffold_prior_sweep_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
