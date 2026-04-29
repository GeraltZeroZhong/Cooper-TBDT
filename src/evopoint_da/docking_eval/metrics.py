from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .stats import (
    bootstrap_ci_indicator,
    is_truthy,
    mean,
    pearson,
    percentile,
    safe_float,
    spearman,
    std_sample,
    wilson_ci,
)


@dataclass
class Top1Summary:
    n_targets: int
    n_success: int
    success_rate: float
    success_rate_percent: float
    rmsd_threshold: float
    ci95_wilson_low: float
    ci95_wilson_high: float
    ci95_bootstrap_low: float
    ci95_bootstrap_high: float
    rmsd_mean: float
    rmsd_median: float
    rmsd_std: float
    rmsd_min: float
    rmsd_p25: float
    rmsd_p75: float
    rmsd_p90: float
    rmsd_p95: float
    rmsd_max: float


@dataclass
class DeltaScoreSummary:
    n_targets: int
    n_improved: int
    improvement_rate: float
    improvement_rate_percent: float
    mean_delta: float
    median_delta: float
    std_delta: float
    min_delta: float
    p25_delta: float
    p75_delta: float
    p90_delta: float
    p95_delta: float
    max_delta: float
    mean_cooper_tbdt: float
    mean_af2: float
    paired_t_stat: float


def select_top1_and_rank(
    rows: list[dict[str, str]],
    target_col: str,
    rank_col: str,
    pose_score_col: str | None,
    score_direction: str,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(target_col, ""), []).append(row)

    use_rank = all(rank_col in row for row in rows)
    if use_rank:
        def key_fn(r: dict[str, str]) -> tuple[float, str]:
            v = safe_float(r.get(rank_col))
            return (v if v is not None else float("inf"), str(r))
    elif pose_score_col is not None and all(pose_score_col in row for row in rows):
        reverse = score_direction == "higher_better"

        def key_fn(r: dict[str, str]) -> tuple[float, str]:
            v = safe_float(r.get(pose_score_col))
            if v is None:
                return (float("-inf") if reverse else float("inf"), str(r))
            return (v, str(r))
    else:
        raise ValueError("Need rank column or pose score column to identify Top-1.")

    top1: list[dict[str, str]] = []
    ranked_by_target: dict[str, list[dict[str, str]]] = {}
    for _, items in grouped.items():
        if use_rank:
            ranked = sorted(items, key=key_fn)
        elif score_direction == "higher_better":
            ranked = sorted(items, key=key_fn, reverse=True)
        else:
            ranked = sorted(items, key=key_fn)

        best = ranked[0]
        tid = best.get(target_col, "")
        enriched = []
        for idx, raw in enumerate(ranked, start=1):
            item = dict(raw)
            item["_ranked_position"] = str(idx)
            enriched.append(item)
        ranked_by_target[tid] = enriched

        row = dict(best)
        row["_top1_selection_method"] = "rank" if use_rank else "pose_score"
        top1.append(row)

    return top1, ranked_by_target


def summarize_top1(top1_rows: list[dict[str, str]], rmsd_col: str, threshold: float, n_iter: int, seed: int) -> tuple[Top1Summary, list[float]]:
    rmsd_values = [v for v in (safe_float(r.get(rmsd_col)) for r in top1_rows) if v is not None]
    n = len(rmsd_values)
    flags = [1 if x < threshold else 0 for x in rmsd_values]
    k = sum(flags)
    sr = k / n if n else float("nan")
    wl, wh = wilson_ci(k, n)
    bl, bh = bootstrap_ci_indicator(flags, n_iter=n_iter, seed=seed)
    return Top1Summary(
        n_targets=n,
        n_success=k,
        success_rate=sr,
        success_rate_percent=sr * 100 if n else float("nan"),
        rmsd_threshold=threshold,
        ci95_wilson_low=wl,
        ci95_wilson_high=wh,
        ci95_bootstrap_low=bl,
        ci95_bootstrap_high=bh,
        rmsd_mean=mean(rmsd_values),
        rmsd_median=percentile(rmsd_values, 0.5),
        rmsd_std=std_sample(rmsd_values),
        rmsd_min=min(rmsd_values) if rmsd_values else float("nan"),
        rmsd_p25=percentile(rmsd_values, 0.25),
        rmsd_p75=percentile(rmsd_values, 0.75),
        rmsd_p90=percentile(rmsd_values, 0.9),
        rmsd_p95=percentile(rmsd_values, 0.95),
        rmsd_max=max(rmsd_values) if rmsd_values else float("nan"),
    ), rmsd_values


def summarize_delta(
    rows: list[dict[str, str]],
    cooper_tbdt_col: str,
    af2_col: str,
) -> tuple[DeltaScoreSummary, list[float], list[float], list[float]]:
    cooper_tbdt_values: list[float] = []
    af2_values: list[float] = []
    delta_values: list[float] = []
    for row in rows:
        cooper_tbdt = safe_float(row.get(cooper_tbdt_col))
        af2 = safe_float(row.get(af2_col))
        if cooper_tbdt is None or af2 is None:
            continue
        cooper_tbdt_values.append(cooper_tbdt)
        af2_values.append(af2)
        delta_values.append(cooper_tbdt - af2)

    n = len(delta_values)
    improved = [1 if d < 0 else 0 for d in delta_values]
    k = sum(improved)
    rate = k / n if n else float("nan")
    if n > 1:
        d_mean = mean(delta_values)
        d_std = std_sample(delta_values)
        t_stat = d_mean / (d_std / math.sqrt(n)) if not math.isnan(d_std) and d_std > 0 else float("nan")
    else:
        t_stat = float("nan")

    return DeltaScoreSummary(
        n_targets=n,
        n_improved=k,
        improvement_rate=rate,
        improvement_rate_percent=rate * 100 if n else float("nan"),
        mean_delta=mean(delta_values),
        median_delta=percentile(delta_values, 0.5),
        std_delta=std_sample(delta_values),
        min_delta=min(delta_values) if delta_values else float("nan"),
        p25_delta=percentile(delta_values, 0.25),
        p75_delta=percentile(delta_values, 0.75),
        p90_delta=percentile(delta_values, 0.9),
        p95_delta=percentile(delta_values, 0.95),
        max_delta=max(delta_values) if delta_values else float("nan"),
        mean_cooper_tbdt=mean(cooper_tbdt_values),
        mean_af2=mean(af2_values),
        paired_t_stat=t_stat,
    ), cooper_tbdt_values, af2_values, delta_values


def summarize_topn_success(
    ranked_by_target: dict[str, list[dict[str, str]]],
    rmsd_col: str,
    threshold: float,
    topn_levels: list[int],
) -> dict[str, float]:
    result: dict[str, float] = {}
    n_targets = len(ranked_by_target)
    for n in topn_levels:
        hit = 0
        for poses in ranked_by_target.values():
            if any((safe_float(p.get(rmsd_col)) is not None and safe_float(p.get(rmsd_col)) < threshold) for p in poses[:n]):
                hit += 1
        result[f"top{n}_success_rate"] = hit / n_targets if n_targets else float("nan")
    return result


def compute_first_hit_stats(
    ranked_by_target: dict[str, list[dict[str, str]]],
    rmsd_col: str,
    threshold: float,
    topn_levels: list[int],
) -> dict[str, Any]:
    first_hits: list[int] = []
    misses = 0
    for poses in ranked_by_target.values():
        hit_rank = None
        for i, p in enumerate(poses, start=1):
            rv = safe_float(p.get(rmsd_col))
            if rv is not None and rv < threshold:
                hit_rank = i
                break
        if hit_rank is None:
            misses += 1
        else:
            first_hits.append(hit_rank)

    n_targets = len(ranked_by_target)
    cdf = {}
    for n in topn_levels:
        cdf[f"first_hit_le_top{n}"] = (
            sum(1 for r in first_hits if r <= n) / n_targets if n_targets else float("nan")
        )

    return {
        "n_targets": n_targets,
        "n_hits": len(first_hits),
        "n_misses": misses,
        "miss_rate": misses / n_targets if n_targets else float("nan"),
        "first_hit_rank_mean": mean([float(x) for x in first_hits]) if first_hits else float("nan"),
        "first_hit_rank_median": percentile([float(x) for x in first_hits], 0.5) if first_hits else float("nan"),
        "first_hit_rank_p75": percentile([float(x) for x in first_hits], 0.75) if first_hits else float("nan"),
        "first_hit_rank_p90": percentile([float(x) for x in first_hits], 0.9) if first_hits else float("nan"),
        "cdf": cdf,
    }


def summarize_topn_success_and_valid(
    ranked_by_target: dict[str, list[dict[str, str]]],
    rmsd_col: str,
    valid_col: str,
    threshold: float,
    topn_levels: list[int],
) -> dict[str, float]:
    result: dict[str, float] = {}
    n_targets = len(ranked_by_target)
    for n in topn_levels:
        hit = 0
        for poses in ranked_by_target.values():
            ok = False
            for pose in poses[:n]:
                rv = safe_float(pose.get(rmsd_col))
                if rv is not None and rv < threshold and is_truthy(pose.get(valid_col)):
                    ok = True
                    break
            if ok:
                hit += 1
        result[f"top{n}_success_and_valid_rate"] = hit / n_targets if n_targets else float("nan")
    return result


def per_target_pose_metrics(
    ranked_by_target: dict[str, list[dict[str, str]]],
    target_col: str,
    rmsd_col: str,
    topn_levels: list[int],
    threshold: float,
    pose_score_col: str | None,
    pose_valid_col: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, poses in ranked_by_target.items():
        rmsd_all = [safe_float(p.get(rmsd_col)) for p in poses]
        rmsd_valid = [x for x in rmsd_all if x is not None]

        first_hit_rank = None
        for i, rv in enumerate(rmsd_all, start=1):
            if rv is not None and rv < threshold:
                first_hit_rank = i
                break

        rec: dict[str, Any] = {
            target_col: tid,
            "n_poses": len(poses),
            "top1_rmsd": rmsd_all[0] if rmsd_all else float("nan"),
            "best_rmsd_any_pose": min(rmsd_valid) if rmsd_valid else float("nan"),
            "median_rmsd_all_poses": percentile(rmsd_valid, 0.5) if rmsd_valid else float("nan"),
            "first_hit_rank": first_hit_rank if first_hit_rank is not None else "NA",
        }
        rec["top1_success"] = 1 if (rmsd_all and rmsd_all[0] is not None and rmsd_all[0] < threshold) else 0

        if pose_valid_col:
            top1_valid = is_truthy(poses[0].get(pose_valid_col)) if poses else False
            rec["top1_valid"] = 1 if top1_valid else 0
            rec["top1_success_and_valid"] = 1 if rec["top1_success"] == 1 and top1_valid else 0

        for n in topn_levels:
            rec[f"top{n}_success"] = 1 if (first_hit_rank is not None and first_hit_rank <= n) else 0
            if pose_valid_col:
                okv = 0
                for pose in poses[:n]:
                    rv = safe_float(pose.get(rmsd_col))
                    vv = is_truthy(pose.get(pose_valid_col))
                    if rv is not None and rv < threshold and vv:
                        okv = 1
                        break
                rec[f"top{n}_success_and_valid"] = okv

        if pose_score_col:
            scores = [safe_float(p.get(pose_score_col)) for p in poses]
            paired = [(s, r) for s, r in zip(scores, rmsd_all) if s is not None and r is not None]
            if len(paired) > 1:
                sc = [x[0] for x in paired]
                rr = [x[1] for x in paired]
                rec["pearson_score_vs_rmsd"] = pearson(sc, rr)
                rec["spearman_score_vs_rmsd"] = spearman(sc, rr)
            else:
                rec["pearson_score_vs_rmsd"] = float("nan")
                rec["spearman_score_vs_rmsd"] = float("nan")

        rows.append(rec)
    return rows


def build_report(
    top1_summary: Top1Summary | None,
    delta_summary: DeltaScoreSummary | None,
    rmsd_values: list[float],
    topn_summary: dict[str, float] | None,
    first_hit_summary: dict[str, Any] | None,
    topn_valid_summary: dict[str, float] | None,
    cooper_tbdt_values: list[float],
    af2_values: list[float],
    delta_values: list[float],
    meta: dict[str, Any],
    target_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"meta": meta}
    if top1_summary is not None:
        report["top1_success"] = asdict(top1_summary)
        report["topn_success"] = topn_summary or {}
        report["first_hit_rank"] = first_hit_summary or {}
        report["top1_success_curve"] = {
            f"rmsd_lt_{th:.1f}A": (
                sum(1 for x in rmsd_values if x < th) / len(rmsd_values) if rmsd_values else float("nan")
            )
            for th in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        }
    if topn_valid_summary is not None:
        report["topn_success_and_valid"] = topn_valid_summary
    if delta_summary is not None:
        report["delta_score"] = asdict(delta_summary)
        if cooper_tbdt_values and af2_values:
            report["delta_score_extra"] = {
                "pearson_corr_cooper_tbdt_vs_af2": pearson(cooper_tbdt_values, af2_values),
                "n_cooper_tbdt_better": sum(1 for c, a in zip(cooper_tbdt_values, af2_values) if c < a),
                "n_af2_better": sum(1 for c, a in zip(cooper_tbdt_values, af2_values) if c > a),
                "n_tied": sum(1 for c, a in zip(cooper_tbdt_values, af2_values) if c == a),
                "delta_negative_rate": sum(1 for d in delta_values if d < 0) / len(delta_values) if delta_values else float("nan"),
                "delta_positive_rate": sum(1 for d in delta_values if d > 0) / len(delta_values) if delta_values else float("nan"),
            }
    report["n_target_metrics_rows"] = len(target_metrics)
    return report
