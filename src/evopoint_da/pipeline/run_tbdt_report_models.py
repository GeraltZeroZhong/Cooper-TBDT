"""Run fixed report-facing TBDT neural models after graph-feature rebuilds."""

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
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    study_name: str
    data_dir: str
    max_epochs: int
    seed: int
    selector: str
    overrides: dict[str, Any]
    init_from: str | None = None


REPORT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="gold_balanced",
        label="Gold-only balanced ablation",
        study_name="tbdt_report_gold_balanced_fixed_pae",
        data_dir="data/processed_tbdt_gold_graphs",
        max_epochs=40,
        seed=42,
        selector="best-selection",
        overrides={
            "model.lr": "3e-4",
            "model.barrel_core_loss_weight": 0.1,
            "model.eval_region_loss_weight": 1.5,
            "model.plug_loss_weight": 2.0,
            "model.extracellular_loop_loss_weight": 1.0,
            "model.tonb_box_loss_weight": 3.0,
            "model.substrate_contact_loss_weight": 3.0,
            "model.scaffold_anchor_weight": 0.0,
            "model.coord_init_gain": 0.01,
            "model.output_scale": 2.0,
            "model.gvp_dropout": 0.05,
        },
    ),
    ModelSpec(
        name="gold_plug_specialist",
        label="Gold-only plug/eval specialist",
        study_name="tbdt_report_gold_plug_specialist_fixed_pae",
        data_dir="data/processed_tbdt_gold_graphs",
        max_epochs=40,
        seed=42,
        selector="best-selection",
        overrides={
            "model.lr": "1e-4",
            "model.weight_decay": "1e-5",
            "model.barrel_core_loss_weight": 0.05,
            "model.eval_region_loss_weight": 3.0,
            "model.plug_loss_weight": 5.0,
            "model.extracellular_loop_loss_weight": 1.0,
            "model.tonb_box_loss_weight": 1.0,
            "model.substrate_contact_loss_weight": 1.0,
            "model.scaffold_anchor_weight": 0.0,
            "model.coord_init_gain": 0.005,
            "model.output_scale": 1.5,
            "model.gvp_dropout": 0.1,
        },
    ),
    ModelSpec(
        name="gold_tonb_specialist",
        label="Gold-only TonB specialist",
        study_name="tbdt_report_gold_tonb_specialist_fixed_pae",
        data_dir="data/processed_tbdt_gold_graphs",
        max_epochs=40,
        seed=42,
        selector="best-flex",
        overrides={
            "model.lr": "2e-4",
            "model.barrel_core_loss_weight": 0.05,
            "model.eval_region_loss_weight": 1.0,
            "model.plug_loss_weight": 1.0,
            "model.extracellular_loop_loss_weight": 1.0,
            "model.tonb_box_loss_weight": 8.0,
            "model.substrate_contact_loss_weight": 4.0,
            "model.scaffold_anchor_weight": 0.0,
            "model.coord_init_gain": 0.005,
            "model.output_scale": 2.0,
            "model.gvp_dropout": 0.05,
        },
    ),
    ModelSpec(
        name="silver_pretrain",
        label="Silver clean geometric pretrain",
        study_name="tbdt_report_silver_pretrain_fixed_pae",
        data_dir="data/processed_tbdt_silver_graphs_clean",
        max_epochs=25,
        seed=42,
        selector="best-selection",
        overrides=FINAL_SINGLE_OVERRIDES,
    ),
    ModelSpec(
        name="silver_pretrain_gold_finetune",
        label="Silver pretrain then Gold fine-tune",
        study_name="tbdt_report_silver_pretrain_gold_finetune_fixed_pae",
        data_dir="data/processed_tbdt_gold_graphs",
        max_epochs=40,
        seed=42,
        selector="best-selection",
        overrides=FINAL_SINGLE_OVERRIDES,
        init_from="silver_pretrain",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/report_models")
    parser.add_argument("--models", default=",".join(spec.name for spec in REPORT_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-bootstrap-iter", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--primary-base-prefix",
        default="artifacts/tbdt_v1/seed_stability_best_selection/predictions/seed_404_best-selection",
    )
    return parser.parse_args()


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


def _parse_model_names(text: str) -> list[str]:
    names = [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
    known = {spec.name for spec in REPORT_MODELS}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown report model(s): {unknown}. Known models: {sorted(known)}")
    return names


def _stringify_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"{key}={text}"


def _latest_run_dir(study_name: str) -> Path:
    root = Path("checkpoints") / study_name
    if not root.exists():
        raise FileNotFoundError(f"No checkpoint directory found for {study_name}: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directories under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint(run_dir: Path, selector: str) -> Path:
    matches = sorted(run_dir.glob(f"{selector}-*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No {selector} checkpoint found in {run_dir}")
    return matches[-1]


def _has_checkpoint(study_name: str, selector: str) -> bool:
    root = Path("checkpoints") / study_name
    return root.exists() and any(root.glob(f"*/{selector}-*.ckpt"))


def _selected_epoch_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"-(\d+)-[0-9.]+\.ckpt$", path.name)
    return int(match.group(1)) if match else None


def _run_train(spec: ModelSpec, *, init_checkpoint: Path | None, log_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "train.py",
        "data=tbdt_state",
        "model=gvp_tbdt_module",
        f"trainer.max_epochs={int(spec.max_epochs)}",
        f"data.data_dir={spec.data_dir}",
        "data.batch_size=1",
        "data.num_workers=0",
        f"data.split_seed={int(args.split_seed)}",
        f"data.split_source={args.split_source}",
        f"seed={int(spec.seed)}",
        f"study_name={spec.study_name}",
        f"logger.save_dir=logs/{spec.study_name}",
        "+run_post_train_tests=false",
        "+trainer.deterministic=true",
        "+trainer.benchmark=false",
        *[_stringify_override(key, value) for key, value in spec.overrides.items()],
    ]
    if init_checkpoint is not None:
        cmd.append(f"+init_checkpoint={init_checkpoint}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(int(spec.seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _split_files(data_dir: str, split: str, args: argparse.Namespace) -> list[str]:
    splits = build_split_file_lists(
        root=data_dir,
        split_ranges=DEFAULT_SPLIT_RANGES,
        split_seed=int(args.split_seed),
        split_source=args.split_source,
    )
    return splits.get(split, [])


def _predict_and_eval(spec: ModelSpec, ckpt: Path, split: str, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    pred_dir = out_dir / "predictions" / f"{spec.name}_{split}"
    metrics_json = out_dir / "metrics" / f"{spec.name}_{split}.json"
    metrics_csv = out_dir / "metrics" / f"{spec.name}_{split}.csv"
    if args.force or not pred_dir.exists():
        predict_graphs(
            argparse.Namespace(
                ckpt=str(ckpt),
                data_dir=spec.data_dir,
                output_dir=str(pred_dir),
                split=split,
                split_source=args.split_source,
                split_seed=int(args.split_seed),
                batch_size=1,
                num_workers=0,
                device=args.device,
                report_path=str(out_dir / "prediction_reports" / f"{spec.name}_{split}.json"),
            )
        )
    return evaluate_regions(
        argparse.Namespace(
            inputs=_split_files(spec.data_dir, split, args),
            predictions=str(pred_dir),
            output_json=str(metrics_json),
            output_csv=str(metrics_csv),
            region_json=None,
            include_all_region=True,
            direction_threshold=1.0,
            bootstrap_iter=int(args.eval_bootstrap_iter),
            bootstrap_seed=42,
        )
    )


def _region_metric(report: dict[str, Any], region: str, metric: str) -> float:
    try:
        return float(report["aggregate_by_region"][region][metric])
    except Exception:
        return float("nan")


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


def _run_blend(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from evopoint_da.pipeline import blend_tbdt_predictions

    primary_prefix = Path(args.primary_base_prefix)
    base_test = Path(str(primary_prefix) + "_test")
    plug_test = out_dir / "predictions" / "gold_plug_specialist_test"
    tonb_test = out_dir / "predictions" / "gold_tonb_specialist_test"
    plug_val = out_dir / "predictions" / "gold_plug_specialist_val"
    blend_dir = out_dir / "predictions" / "validation_calibrated_region_blend_test"
    report_path = out_dir / "prediction_reports" / "validation_calibrated_region_blend_test.json"

    for required in (base_test, plug_test, tonb_test, plug_val):
        if not required.exists():
            raise FileNotFoundError(f"Missing blend input: {required}")

    blend_tbdt_predictions.main_with_args(
        argparse.Namespace(
            data_dir="data/processed_tbdt_gold_graphs",
            split="test",
            split_source=args.split_source,
            split_seed=int(args.split_seed),
            base_predictions=str(base_test),
            base_scale=1.0,
            region_source=[f"plug={plug_test}", f"tonb_box={tonb_test}"],
            region_scale=[],
            auto_scale_region=["plug"],
            calibration_split="val",
            calibration_region_source=[f"plug={plug_val}"],
            max_scale=12.0,
            min_calibration_residues=100,
            priority="tonb_box,plug,substrate_contact,extracellular_loop,eval,barrel_core",
            output_dir=str(blend_dir),
            report_path=str(report_path),
        )
    )
    metrics_json = out_dir / "metrics" / "validation_calibrated_region_blend_test.json"
    metrics_csv = out_dir / "metrics" / "validation_calibrated_region_blend_test.csv"
    report = evaluate_regions(
        argparse.Namespace(
            inputs=_split_files("data/processed_tbdt_gold_graphs", "test", args),
            predictions=str(blend_dir),
            output_json=str(metrics_json),
            output_csv=str(metrics_csv),
            region_json=None,
            include_all_region=True,
            direction_threshold=1.0,
            bootstrap_iter=int(args.eval_bootstrap_iter),
            bootstrap_seed=42,
        )
    )
    return {"prediction_report": str(report_path), "metrics_json": str(metrics_json), "metrics": report}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_names = set(_parse_model_names(args.models))
    specs = [spec for spec in REPORT_MODELS if spec.name in selected_names]

    checkpoints: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        init_checkpoint = checkpoints.get(spec.init_from or "") if spec.init_from else None
        if not args.skip_train and (args.force or not _has_checkpoint(spec.study_name, spec.selector)):
            _run_train(spec, init_checkpoint=init_checkpoint, log_path=out_dir / "train_logs" / f"{spec.name}.log", args=args)
        run_dir = _latest_run_dir(spec.study_name)
        ckpt = _checkpoint(run_dir, spec.selector)
        checkpoints[spec.name] = ckpt
        for split in ("val", "test"):
            report = _predict_and_eval(spec, ckpt, split, out_dir, args)
            row = {
                "model": spec.name,
                "label": spec.label,
                "split": split,
                "selector": spec.selector,
                "checkpoint": str(ckpt),
                "selected_epoch": _selected_epoch_from_checkpoint(ckpt),
                "data_dir": spec.data_dir,
            }
            for region in ("eval", "plug", "tonb_box", "barrel_core", "all"):
                row[f"{region}_prediction_error_rms"] = _region_metric(report, region, "prediction_error_rms")
                row[f"{region}_predicted_displacement_mean"] = _region_metric(
                    report,
                    region,
                    "predicted_displacement_mean",
                )
                row[f"{region}_mse_improvement_vs_zero_fraction"] = _region_metric(
                    report,
                    region,
                    "mse_improvement_vs_zero_fraction",
                )
            rows.append(row)
        _write_csv(out_dir / "report_model_summary.csv", rows)
        (out_dir / "report_model_summary.json").write_text(
            json.dumps(_json_safe(rows), indent=2),
            encoding="utf-8",
        )

    blend_report = None
    if args.run_blend:
        blend_report = _run_blend(out_dir, args)

    report = {
        "out_dir": str(out_dir),
        "models": [spec.name for spec in specs],
        "summary_csv": str(out_dir / "report_model_summary.csv"),
        "summary_json": str(out_dir / "report_model_summary.json"),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "blend": blend_report,
    }
    (out_dir / "report_models_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
