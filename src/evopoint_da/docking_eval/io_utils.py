from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=sep))


def assert_columns(rows: list[dict[str, str]], required: list[str], file_label: str) -> None:
    if not rows:
        raise ValueError(f"{file_label} is empty.")
    columns = set(rows[0].keys())
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"{file_label} missing required columns: {missing}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_set: set[str] = set()
    for row in rows:
        field_set.update(row.keys())
    fieldnames = sorted(field_set)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
