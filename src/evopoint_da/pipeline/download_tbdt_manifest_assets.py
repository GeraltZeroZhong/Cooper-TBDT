from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evopoint_da.pipeline.build_tbdt_mixed_manifest import (
    MIXED_FIELDNAMES,
    _download_rcsb_structure,
    _write_csv,
    _write_json,
)
from evopoint_da.pipeline.fetch_tbdt_structures import DEFAULT_AF_VERSION, _download_afdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download assets referenced by an existing TBDT mixed manifest.")
    p.add_argument("--manifest", default="data/tbdt_mixed_manifest.csv")
    p.add_argument("--out-manifest", default=None, help="Defaults to overwriting --manifest.")
    p.add_argument("--tier", action="append", choices=["gold", "silver", "bronze"], help="Tier to download.")
    p.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    p.add_argument("--raw-af2-dir", default="data/raw_af2")
    p.add_argument("--af-version", type=int, default=DEFAULT_AF_VERSION)
    p.add_argument("--download-pae", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sync-tier-manifests", action="store_true")
    p.add_argument("--gold-manifest", default="data/tbdt_gold_manifest.csv")
    p.add_argument("--silver-manifest", default="data/tbdt_silver_manifest.csv")
    p.add_argument("--bronze-manifest", default="data/tbdt_bronze_manifest.csv")
    p.add_argument("--report-path", default="artifacts/tbdt_v1/download_tbdt_manifest_assets_report.json")
    return p.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _download_pdb_task(pdb_id: str, raw_pdb_dir: Path, overwrite: bool) -> dict[str, Any]:
    result = _download_rcsb_structure(pdb_id, raw_pdb_dir, overwrite=overwrite)
    return {"id": pdb_id, "ok": True, **result}


def _download_afdb_task(
    uniprot_id: str,
    raw_af2_dir: Path,
    af_version: int,
    download_pae: bool,
    overwrite: bool,
) -> dict[str, Any]:
    result = _download_afdb(
        uniprot_id,
        raw_af2_dir,
        af_version=af_version,
        download_pae=download_pae,
        overwrite=overwrite,
    )
    return {"id": uniprot_id, "ok": True, **result}


def _run_pool(label: str, fn, items: list[str], workers: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    if not items:
        return results, failures

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_to_id = {pool.submit(fn, item): item for item in items}
        for future in as_completed(future_to_id):
            item = future_to_id[future]
            try:
                results[item] = future.result()
            except Exception as exc:
                failures.append({"kind": label, "id": item, "error": str(exc)})
                results[item] = {"id": item, "ok": False, "error": str(exc)}
    return results, failures


def download_assets(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    fieldnames, rows = _read_csv(manifest_path)
    if not fieldnames:
        fieldnames = list(MIXED_FIELDNAMES)

    tiers = set(args.tier or ["gold", "silver", "bronze"])
    target_rows = [row for row in rows if row.get("evidence_level") in tiers]
    raw_pdb_dir = Path(args.raw_pdb_dir)
    raw_af2_dir = Path(args.raw_af2_dir)

    pdb_ids = sorted({str(row.get("pdb_id") or "").strip().upper() for row in target_rows if row.get("pdb_id")})
    uniprot_ids = sorted({str(row.get("uniprot_id") or "").strip() for row in target_rows if row.get("uniprot_id")})

    pdb_results, pdb_failures = _run_pool(
        "pdb",
        lambda pdb_id: _download_pdb_task(pdb_id, raw_pdb_dir, bool(args.overwrite)),
        pdb_ids,
        int(args.workers),
    )
    afdb_results, afdb_failures = _run_pool(
        "afdb",
        lambda uniprot_id: _download_afdb_task(
            uniprot_id,
            raw_af2_dir,
            int(args.af_version),
            bool(args.download_pae),
            bool(args.overwrite),
        ),
        uniprot_ids,
        int(args.workers),
    )

    status_counts = Counter()
    for row in target_rows:
        pdb_id = str(row.get("pdb_id") or "").strip().upper()
        uniprot_id = str(row.get("uniprot_id") or "").strip()
        status_parts: list[str] = []

        if pdb_id:
            pdb_result = pdb_results.get(pdb_id, {"ok": False})
            if pdb_result.get("ok"):
                row["experimental_pdb"] = str(pdb_result["relative_path"])
                status = f"{pdb_result['format']}:{pdb_result['status']}"
            else:
                status = "structure:failed"
            status_parts.append(status)
            status_counts[status] += 1

        if uniprot_id:
            afdb_result = afdb_results.get(uniprot_id, {"ok": False})
            if afdb_result.get("ok"):
                row["af2_pdb"] = f"raw_af2/AF-{uniprot_id}-F1-model_v{int(args.af_version)}.pdb"
                model_status = afdb_result["model"]["status"]
                status = f"afdb:{model_status}"
            else:
                status = "afdb:failed"
            status_parts.append(status)
            status_counts[status] += 1

        row["download_status"] = ";".join(status_parts) if status_parts else "nothing_to_download"

    out_manifest = Path(args.out_manifest) if args.out_manifest else manifest_path
    _write_csv(out_manifest, rows)

    tier_paths = {
        "gold": Path(args.gold_manifest),
        "silver": Path(args.silver_manifest),
        "bronze": Path(args.bronze_manifest),
    }
    if args.sync_tier_manifests:
        for tier, path in tier_paths.items():
            _write_csv(path, [row for row in rows if row.get("evidence_level") == tier])

    report = {
        "manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "tiers": sorted(tiers),
        "target_rows": len(target_rows),
        "unique_pdb_ids": len(pdb_ids),
        "unique_uniprot_ids": len(uniprot_ids),
        "download_pae": bool(args.download_pae),
        "workers": int(args.workers),
        "status_counts": dict(status_counts),
        "pdb_failures": pdb_failures,
        "afdb_failures": afdb_failures,
        "synced_tier_manifests": bool(args.sync_tier_manifests),
    }
    _write_json(Path(args.report_path), report)
    return report


def main() -> None:
    report = download_assets(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
