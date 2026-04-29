"""Run fixed-seed stability checks for the final single scaffold-prior model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
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
DEFAULT_SEEDS = (42, 101, 202, 303, 404)
REGIONS = ("eval", "plug", "tonb_box", "barrel_core", "all")
METRICS = (
    "prediction_error_rms",
    "zero_error_rms",
    "mse_improvement_vs_zero_fraction",
    "predicted_displacement_mean",
    "sample_improvement_rate",
    "direction_cosine_mean",
    "magnitude_mae",
)

# Validation-selected final single-model configuration. This is intentionally a
# single fixed specification, not a sweep.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/tbdt_v1/seed_stability")
    parser.add_argument("--study-prefix", default="tbdt_single_scaffold_prior_seed_stability")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--checkpoint-selector", default="best-disp1to5")
    parser.add_argument("--eval-bootstrap-iter", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing seed checkpoints.")
    parser.add_argument("--force", action="store_true", help="Rerun training and prediction even when outputs exist.")
    parser.add_argument(
        "--no-deterministic-trainer",
        action="store_true",
        help="Do not pass deterministic Trainer settings to train.py.",
    )
    return parser.parse_args()


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for item in re.split(r"[,\s]+", str(text).strip()):
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique: {seeds}")
    return seeds


def _stringify_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"{key}={text}"


def _seed_name(seed: int) -> str:
    return f"seed_{int(seed)}"


def _run_train(seed: int, args: argparse.Namespace, log_path: Path) -> None:
    seed_name = _seed_name(seed)
    study_name = f"{args.study_prefix}/{seed_name}"
    cmd = [
        sys.executable,
        "train.py",
        "data=tbdt_state",
        "model=gvp_tbdt_module",
        f"trainer.max_epochs={int(args.max_epochs)}",
        "data.batch_size=1",
        "data.num_workers=0",
        f"data.split_seed={int(args.split_seed)}",
        f"data.split_source={args.split_source}",
        f"seed={int(seed)}",
        f"study_name={study_name}",
        f"logger.save_dir=logs/{args.study_prefix}/{seed_name}",
        "+run_post_train_tests=false",
        *[_stringify_override(key, value) for key, value in FINAL_SINGLE_OVERRIDES.items()],
    ]
    if not args.no_deterministic_trainer:
        cmd.extend(["+trainer.deterministic=true", "+trainer.benchmark=false"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(int(seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _latest_run_dir(study_prefix: str, seed: int) -> Path:
    root = Path("checkpoints") / study_prefix / _seed_name(seed)
    if not root.exists():
        raise FileNotFoundError(f"No checkpoint directory found for seed {seed}: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint run directories under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint(run_dir: Path, selector: str) -> Path:
    matches = sorted(run_dir.glob(f"{selector}-*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No {selector} checkpoint found in {run_dir}")
    return matches[-1]


def _has_checkpoint(study_prefix: str, seed: int, selector: str) -> bool:
    root = Path("checkpoints") / study_prefix / _seed_name(seed)
    return root.exists() and any(root.glob(f"*/{selector}-*.ckpt"))


def _split_files(args: argparse.Namespace, split: str) -> list[str]:
    splits = build_split_file_lists(
        root=args.data_dir,
        split_ranges=DEFAULT_SPLIT_RANGES,
        split_seed=int(args.split_seed),
        split_source=args.split_source,
    )
    return splits.get(split, [])


def _predict_and_eval(
    *,
    seed: int,
    ckpt: Path,
    split: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    seed_name = _seed_name(seed)
    selector = str(args.checkpoint_selector)
    pred_dir = out_dir / "predictions" / f"{seed_name}_{selector}_{split}"
    metrics_json = out_dir / "metrics" / f"{seed_name}_{selector}_{split}.json"
    metrics_csv = out_dir / "metrics" / f"{seed_name}_{selector}_{split}.csv"
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
                report_path=str(out_dir / "prediction_reports" / f"{seed_name}_{selector}_{split}.json"),
            )
        )

    return evaluate_regions(
        argparse.Namespace(
            inputs=_split_files(args, split),
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


def _region_metric(report: dict[str, Any], region: str, key: str, default: float = float("nan")) -> float:
    try:
        value = report["aggregate_by_region"][region][key]
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _score(report: dict[str, Any]) -> float:
    functional = _mean(
        [
            _region_metric(report, "eval", "mse_improvement_vs_zero_fraction"),
            _region_metric(report, "plug", "mse_improvement_vs_zero_fraction"),
            _region_metric(report, "tonb_box", "mse_improvement_vs_zero_fraction"),
        ]
    )
    core_pred = _region_metric(report, "barrel_core", "predicted_displacement_mean", 0.0)
    return functional - 0.25 * max(0.0, core_pred - 0.05)


def _selected_epoch_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"-(\d+)-[0-9.]+\.ckpt$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _summary_row(seed: int, split: str, report: dict[str, Any], ckpt: Path, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": "single_scaffold_prior",
        "seed": int(seed),
        "split_seed": int(args.split_seed),
        "split_source": str(args.split_source),
        "selector": str(args.checkpoint_selector),
        "split": split,
        "checkpoint": str(ckpt),
        "selected_epoch": _selected_epoch_from_checkpoint(ckpt),
        "score": _score(report),
        "max_epochs": int(args.max_epochs),
        "deterministic_trainer": not bool(args.no_deterministic_trainer),
    }
    for key, value in FINAL_SINGLE_OVERRIDES.items():
        row[key] = value
    for region in REGIONS:
        for metric in METRICS:
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
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        for region in REGIONS:
            for metric in METRICS:
                key = f"{region}_{metric}"
                values = [float(row[key]) for row in split_rows if key in row and math.isfinite(float(row[key]))]
                if not values:
                    continue
                aggregate_rows.append(
                    {
                        "split": split,
                        "region": region,
                        "metric": metric,
                        "n_seeds": len(values),
                        "mean": _mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                    }
                )
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        values = [float(row["score"]) for row in split_rows if math.isfinite(float(row["score"]))]
        if values:
            aggregate_rows.append(
                {
                    "split": split,
                    "region": "selection",
                    "metric": "score",
                    "n_seeds": len(values),
                    "mean": _mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    return aggregate_rows


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _write_markdown(path: Path, rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    lines = [
        "# Single Scaffold-Prior Seed Stability",
        "",
        "Fixed model: final single scaffold-prior configuration; checkpoint selector: "
        f"`{args.checkpoint_selector}`; split source: `{args.split_source}`; split seed: `{args.split_seed}`.",
        "",
        "| seed | selected epoch | eval RMSD A | plug RMSD A | TonB RMSD A | barrel-core pred mean A | score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(test_rows, key=lambda item: int(item["seed"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    "" if row["selected_epoch"] is None else str(row["selected_epoch"]),
                    _format_float(row.get("eval_prediction_error_rms")),
                    _format_float(row.get("plug_prediction_error_rms")),
                    _format_float(row.get("tonb_box_prediction_error_rms")),
                    _format_float(row.get("barrel_core_predicted_displacement_mean")),
                    _format_float(row.get("score"), digits=4),
                ]
            )
            + " |"
        )

    lookup = {
        (row["split"], row["region"], row["metric"]): row
        for row in aggregate_rows
        if row.get("split") == "test"
    }
    lines.extend(
        [
            "",
            "## Test Aggregate",
            "",
            "| endpoint | mean | std | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for region, metric, label in [
        ("eval", "prediction_error_rms", "eval RMSD A"),
        ("plug", "prediction_error_rms", "plug RMSD A"),
        ("tonb_box", "prediction_error_rms", "TonB RMSD A"),
        ("barrel_core", "predicted_displacement_mean", "barrel-core predicted mean A"),
        ("selection", "score", "validation-style score"),
    ]:
        item = lookup.get(("test", region, metric))
        if not item:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_float(item.get("mean"), digits=4 if metric == "score" else 3),
                    _format_float(item.get("std"), digits=4 if metric == "score" else 3),
                    _format_float(item.get("min"), digits=4 if metric == "score" else 3),
                    _format_float(item.get("max"), digits=4 if metric == "score" else 3),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    settings = {
        "model": "single_scaffold_prior",
        "seeds": seeds,
        "study_prefix": str(args.study_prefix),
        "data_dir": str(args.data_dir),
        "max_epochs": int(args.max_epochs),
        "split_source": str(args.split_source),
        "split_seed": int(args.split_seed),
        "checkpoint_selector": str(args.checkpoint_selector),
        "eval_bootstrap_iter": int(args.eval_bootstrap_iter),
        "deterministic_trainer": not bool(args.no_deterministic_trainer),
        "overrides": FINAL_SINGLE_OVERRIDES,
    }
    (out_dir / "seed_stability_settings.json").write_text(
        json.dumps(_json_safe(settings), indent=2),
        encoding="utf-8",
    )

    for idx, seed in enumerate(seeds, start=1):
        print(f"[{idx}/{len(seeds)}] seed={seed}", flush=True)
        log_path = out_dir / "train_logs" / f"{_seed_name(seed)}.log"
        if not args.skip_train and (args.force or not _has_checkpoint(args.study_prefix, seed, args.checkpoint_selector)):
            _run_train(seed, args, log_path)
        else:
            print(f"  reuse existing checkpoint for seed={seed}", flush=True)

        run_dir = _latest_run_dir(args.study_prefix, seed)
        ckpt = _checkpoint(run_dir, args.checkpoint_selector)
        for split in ("val", "test"):
            report = _predict_and_eval(seed=seed, ckpt=ckpt, split=split, args=args, out_dir=out_dir)
            rows.append(_summary_row(seed, split, report, ckpt, args))

        aggregate_rows = _aggregate(rows)
        _write_csv(out_dir / "seed_stability_summary.csv", rows)
        _write_csv(out_dir / "seed_stability_aggregate.csv", aggregate_rows)
        (out_dir / "seed_stability_summary.json").write_text(
            json.dumps(_json_safe(rows), indent=2),
            encoding="utf-8",
        )
        _write_markdown(out_dir / "seed_stability_report.md", rows, aggregate_rows, args)

    aggregate_rows = _aggregate(rows)
    report = {
        "settings": settings,
        "n_rows": len(rows),
        "summary_csv": str(out_dir / "seed_stability_summary.csv"),
        "aggregate_csv": str(out_dir / "seed_stability_aggregate.csv"),
        "report_md": str(out_dir / "seed_stability_report.md"),
        "aggregate": aggregate_rows,
    }
    (out_dir / "seed_stability_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
