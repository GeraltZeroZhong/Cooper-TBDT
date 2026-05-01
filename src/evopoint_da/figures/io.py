"""Small data-loading helpers for Cooper-TBDT figure builders."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV input not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON input not found: {json_path}")
    payload = json.loads(json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    return payload


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_float(value: Any, *, field: str, source: str) -> float:
    parsed = as_float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Missing or non-finite {field!r} in {source}")
    return parsed


def sample_region_value(
    payload: Mapping[str, Any],
    *,
    sample_id: str,
    region: str,
    metric: str,
    source: str,
) -> float:
    for sample in payload.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        if sample.get("sample_id") != sample_id:
            continue
        regions = sample.get("regions", {})
        if not isinstance(regions, Mapping) or region not in regions:
            raise ValueError(f"Region {region!r} missing for sample {sample_id!r} in {source}")
        value = regions[region].get(metric)
        return finite_float(value, field=metric, source=f"{source}:{sample_id}:{region}")
    raise ValueError(f"Sample {sample_id!r} missing in {source}")


def iter_sample_region_values(
    payload: Mapping[str, Any],
    *,
    region: str,
    metric: str,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        sample_id = str(sample.get("sample_id", ""))
        regions = sample.get("regions", {})
        if not sample_id or not isinstance(regions, Mapping) or region not in regions:
            continue
        region_payload = regions[region]
        if not isinstance(region_payload, Mapping) or metric not in region_payload:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "value": finite_float(
                    region_payload.get(metric),
                    field=metric,
                    source=f"{source}:{sample_id}:{region}",
                ),
            }
        )
    return rows

