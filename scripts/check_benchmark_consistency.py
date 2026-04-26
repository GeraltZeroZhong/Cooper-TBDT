#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check docking benchmark output consistency.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--required-structures", default="")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    summary_path = args.output_dir / "summary.json"
    poses_path = args.output_dir / "poses_all.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    pose_rows = read_csv(poses_path)
    required = [x.strip() for x in args.required_structures.split(",") if x.strip()]

    manifest_cols = set(manifest_rows[0]) if manifest_rows else set()
    summary_structures = [item["label"] for item in summary.get("meta", {}).get("structures", [])]
    pose_structures = sorted({row.get("structure", "") for row in pose_rows if row.get("structure")})
    pose_counts = Counter(row.get("structure", "") for row in pose_rows if row.get("structure"))

    errors: list[str] = []
    warnings: list[str] = []
    for label in required:
        if f"receptor_{label}" not in manifest_cols:
            errors.append(f"manifest missing receptor_{label}")
        if label not in summary_structures:
            errors.append(f"summary missing structure {label}")
        if label not in pose_structures:
            warnings.append(f"poses_all has no rows for structure {label}")

    if summary.get("n_pose_rows") is not None and int(summary.get("n_pose_rows", -1)) != len(pose_rows):
        errors.append(f"summary n_pose_rows={summary.get('n_pose_rows')} but poses_all has {len(pose_rows)} rows")

    report = {
        "status": "ok" if not errors else "failed",
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "n_manifest_rows": len(manifest_rows),
        "manifest_receptor_columns": sorted(col for col in manifest_cols if col.startswith("receptor_")),
        "summary_structures": summary_structures,
        "pose_structures": pose_structures,
        "pose_counts_by_structure": dict(pose_counts),
        "errors": errors,
        "warnings": warnings,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
