from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import assert_columns, read_table, write_csv
from .metrics import (
    build_report,
    compute_first_hit_stats,
    per_target_pose_metrics,
    select_top1_and_rank,
    summarize_delta,
    summarize_top1,
    summarize_topn_success,
    summarize_topn_success_and_valid,
)
from .reporting import flatten_report_metrics, to_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantify docking success rate and delta Vina score.")
    p.add_argument("--pose-file", type=Path, default=None)
    p.add_argument("--score-file", type=Path, default=None)
    p.add_argument("--target-col", default="target_id")
    p.add_argument("--rmsd-col", default="rmsd")
    p.add_argument("--pose-valid-col", default=None)
    p.add_argument("--rank-col", default="rank")
    p.add_argument("--pose-score-col", default=None)
    p.add_argument("--score-direction", choices=["lower_better", "higher_better"], default="lower_better")
    p.add_argument("--holoshift-score-col", default="score_holoshift")
    p.add_argument("--af2-score-col", default="score_af2")
    p.add_argument("--rmsd-threshold", type=float, default=2.0)
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topn-levels", default="1,2,3,5,10")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-markdown", type=Path, default=None)
    p.add_argument("--out-top1-csv", type=Path, default=None)
    p.add_argument("--out-metrics-csv", type=Path, default=None)
    p.add_argument("--out-target-csv", type=Path, default=None)
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Optional JSON output from pipeline/eval_run.py to merge into a unified report.",
    )
    p.add_argument("--run-id", default=None, help="Optional experiment run identifier for traceability.")
    p.add_argument("--ckpt-id", default=None, help="Optional checkpoint identifier/path for traceability.")
    p.add_argument("--split-seed", type=int, default=None, help="Optional split seed used in experiment.")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_calibration_section(payload: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "alpha",
        "target_coverage",
        "num_calibration_nodes",
        "qhat",
        "score_quantiles",
        "estimated_empirical_coverage",
        "calibration_quality",
        "tail_risk",
        "bootstrap_ci95",
        "stratified_by_plddt",
        "stratified_by_displacement",
        "runtime_cost",
        "multi_seed",
        "drift_check",
    ]
    return {k: payload[k] for k in keep_keys if k in payload}


def main() -> None:
    args = parse_args()
    if args.pose_file is None and args.score_file is None:
        raise ValueError("Please provide at least one of --pose-file or --score-file")

    topn_levels = [int(x.strip()) for x in str(args.topn_levels).split(",") if x.strip()]
    if any(x <= 0 for x in topn_levels):
        raise ValueError("--topn-levels must contain positive integers.")

    top1_rows: list[dict[str, str]] = []
    target_metrics: list[dict[str, object]] = []
    top1_summary = None
    delta_summary = None
    topn_summary = None
    first_hit_summary = None
    topn_valid_summary = None
    rmsd_values: list[float] = []
    hs_values: list[float] = []
    af2_values: list[float] = []
    delta_values: list[float] = []

    if args.pose_file is not None:
        pose_rows = read_table(args.pose_file)
        assert_columns(pose_rows, [args.target_col, args.rmsd_col], "pose-file")
        top1_rows, ranked_by_target = select_top1_and_rank(
            rows=pose_rows,
            target_col=args.target_col,
            rank_col=args.rank_col,
            pose_score_col=args.pose_score_col,
            score_direction=args.score_direction,
        )
        top1_summary, rmsd_values = summarize_top1(
            top1_rows=top1_rows,
            rmsd_col=args.rmsd_col,
            threshold=args.rmsd_threshold,
            n_iter=args.bootstrap_iter,
            seed=args.seed,
        )
        topn_summary = summarize_topn_success(
            ranked_by_target=ranked_by_target,
            rmsd_col=args.rmsd_col,
            threshold=args.rmsd_threshold,
            topn_levels=topn_levels,
        )
        first_hit_summary = compute_first_hit_stats(
            ranked_by_target=ranked_by_target,
            rmsd_col=args.rmsd_col,
            threshold=args.rmsd_threshold,
            topn_levels=topn_levels,
        )
        target_metrics = per_target_pose_metrics(
            ranked_by_target=ranked_by_target,
            target_col=args.target_col,
            rmsd_col=args.rmsd_col,
            topn_levels=topn_levels,
            threshold=args.rmsd_threshold,
            pose_score_col=args.pose_score_col,
            pose_valid_col=args.pose_valid_col,
        )
        if args.pose_valid_col is not None:
            topn_valid_summary = summarize_topn_success_and_valid(
                ranked_by_target=ranked_by_target,
                rmsd_col=args.rmsd_col,
                valid_col=args.pose_valid_col,
                threshold=args.rmsd_threshold,
                topn_levels=topn_levels,
            )

    if args.score_file is not None:
        score_rows = read_table(args.score_file)
        assert_columns(score_rows, [args.target_col, args.holoshift_score_col, args.af2_score_col], "score-file")
        delta_summary, hs_values, af2_values, delta_values = summarize_delta(
            rows=score_rows,
            hs_col=args.holoshift_score_col,
            af2_col=args.af2_score_col,
        )

    report = build_report(
        top1_summary=top1_summary,
        delta_summary=delta_summary,
        rmsd_values=rmsd_values,
        topn_summary=topn_summary,
        first_hit_summary=first_hit_summary,
        topn_valid_summary=topn_valid_summary,
        hs_values=hs_values,
        af2_values=af2_values,
        delta_values=delta_values,
        meta={
            "rmsd_threshold": args.rmsd_threshold,
            "bootstrap_iter": args.bootstrap_iter,
            "seed": args.seed,
            "score_direction": args.score_direction,
            "formula": "delta_score = score_holoshift - score_af2",
            "interpretation": "lower score is better; delta < 0 means HoloShift improved over AF2",
            "topn_levels": topn_levels,
        },
        target_metrics=target_metrics,
    )
    traceability_meta: dict[str, Any] = {
        "run_id": args.run_id,
        "ckpt_id": args.ckpt_id,
        "split_seed": args.split_seed,
    }
    report["traceability_meta"] = {k: v for k, v in traceability_meta.items() if v is not None}

    if args.calibration_json is not None:
        calibration_payload = _load_json(args.calibration_json)
        report["calibration_section"] = _extract_calibration_section(calibration_payload)

        cross_links: dict[str, Any] = {}
        if "top1_success" in report and "calibration_section" in report:
            top1_success = report["top1_success"].get("success_rate")
            tail_risk_2a = report["calibration_section"].get("tail_risk", {}).get("error_gt_2.0A_rate")
            qhat = report["calibration_section"].get("qhat")
            if top1_success is not None:
                cross_links["top1_success_rate"] = top1_success
            if tail_risk_2a is not None:
                cross_links["calib_error_gt_2A_rate"] = tail_risk_2a
            if qhat is not None:
                cross_links["calib_qhat"] = qhat
            if top1_success is not None and tail_risk_2a is not None:
                cross_links["risk_success_gap"] = float(top1_success) - float(tail_risk_2a)

        if cross_links:
            report["cross_links"] = cross_links

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_markdown is not None:
        args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.write_text(to_markdown(report), encoding="utf-8")
    if args.out_top1_csv is not None:
        write_csv(args.out_top1_csv, top1_rows)
    if args.out_target_csv is not None:
        write_csv(args.out_target_csv, target_metrics)
    if args.out_metrics_csv is not None:
        write_csv(args.out_metrics_csv, flatten_report_metrics(report))


if __name__ == "__main__":
    main()
