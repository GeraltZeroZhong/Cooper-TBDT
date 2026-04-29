from __future__ import annotations

from typing import Any


def _flatten(prefix: str, payload: Any, rows: list[dict[str, Any]], section: str) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten(next_prefix, v, rows, section)
        return
    if isinstance(payload, list):
        return
    rows.append({"section": section, "metric": prefix, "value": payload})


def flatten_report_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, payload in report.items():
        _flatten("", payload, rows, section=section if isinstance(payload, dict) else "root")
    return rows


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Docking Power / Pose Prediction Report", "", "## Meta"]
    for k, v in report.get("meta", {}).items():
        lines.append(f"- **{k}**: {v}")

    if "top1_success" in report:
        t = report["top1_success"]
        lines += [
            "",
            "## Top-1 & Top-N Success",
            f"- Top-1 success: {t['success_rate_percent']:.2f}% ({t['n_success']}/{t['n_targets']})",
            f"- Top-1 95% CI (Wilson): [{t['ci95_wilson_low']:.4f}, {t['ci95_wilson_high']:.4f}]",
            f"- Top-1 95% CI (Bootstrap): [{t['ci95_bootstrap_low']:.4f}, {t['ci95_bootstrap_high']:.4f}]",
        ]
        if "topn_success" in report:
            lines.append("- Top-N success rates:")
            for k, v in report["topn_success"].items():
                lines.append(f"  - {k}: {v:.4f}")

    if "first_hit_rank" in report:
        fhr = report["first_hit_rank"]
        lines += [
            "",
            "## First-hit Rank",
            f"- n_hits={fhr.get('n_hits')} / n_targets={fhr.get('n_targets')}, miss_rate={fhr.get('miss_rate'):.4f}",
            f"- mean={fhr.get('first_hit_rank_mean'):.4f}, median={fhr.get('first_hit_rank_median'):.4f}, p75={fhr.get('first_hit_rank_p75'):.4f}, p90={fhr.get('first_hit_rank_p90'):.4f}",
        ]
        cdf = fhr.get("cdf", {})
        if isinstance(cdf, dict) and cdf:
            lines.append("- cumulative hit distribution:")
            for k, v in cdf.items():
                lines.append(f"  - {k}: {v:.4f}")

    if "delta_score" in report:
        d = report["delta_score"]
        lines += [
            "",
            "## Δ Vina Score (Cooper-TBDT - AF2)",
            f"- Improvement rate (delta<0): {d['improvement_rate_percent']:.2f}% ({d['n_improved']}/{d['n_targets']})",
            f"- mean={d['mean_delta']:.4f}, median={d['median_delta']:.4f}, std={d['std_delta']:.4f}",
        ]

    if "traceability_meta" in report and report["traceability_meta"]:
        lines += ["", "## Traceability"]
        for k, v in report["traceability_meta"].items():
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)
