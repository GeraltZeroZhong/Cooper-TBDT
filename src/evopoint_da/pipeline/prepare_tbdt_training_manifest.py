from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import requests

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"

PLUG_MARKERS = ("plug", "cork")
BARREL_MARKERS = ("beta-barrel", "b-barrel", "tonb_dep_rec_b-barrel", "barrel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a trainable TBDT manifest with auto region annotations.")
    p.add_argument("--input-manifest", default="data/tbdt_expansion_manifest.csv")
    p.add_argument("--out-manifest", default="data/tbdt_training_manifest.csv")
    p.add_argument("--annotation-dir", default="data/tbdt_region_annotations/auto")
    p.add_argument("--report-path", default="artifacts/tbdt_v1/prepare_training_manifest_report.json")
    p.add_argument("--exclude-uncertain", action="store_true", default=True)
    p.add_argument("--include-uncertain", action="store_false", dest="exclude_uncertain")
    p.add_argument("--require-local-files", action="store_true", default=True)
    p.add_argument("--allow-missing-local-files", action="store_false", dest="require_local_files")
    p.add_argument("--barrel-core-window-count", type=int, default=16)
    p.add_argument("--barrel-core-window-size", type=int, default=10)
    p.add_argument("--tonb-box-length", type=int, default=7)
    return p.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _request_json(url: str) -> Any:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _row_value(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key)
    return str(value).strip() if value is not None and str(value).strip() else default


def _data_relative(path: Path) -> str:
    data_root = Path("data")
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def _resolve_manifest_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else manifest_dir / path


def _split_ranges(ranges: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in sorted(ranges) if int(start) <= int(end)]


def _map_entity_range_to_ref(
    start: int,
    end: int,
    aligned_regions: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    mapped: list[tuple[int, int]] = []
    for region in aligned_regions:
        entity_start = int(region.get("entity_beg_seq_id") or 0)
        ref_start = int(region.get("ref_beg_seq_id") or 0)
        length = int(region.get("length") or 0)
        if entity_start <= 0 or ref_start <= 0 or length <= 0:
            continue
        entity_end = entity_start + length - 1
        lo = max(start, entity_start)
        hi = min(end, entity_end)
        if lo > hi:
            continue
        mapped_start = ref_start + (lo - entity_start)
        mapped_end = ref_start + (hi - entity_start)
        mapped.append((mapped_start, mapped_end))
    return mapped


def _map_feature_positions(feature: dict[str, Any], aligned_regions: list[dict[str, Any]]) -> list[tuple[int, int]]:
    mapped: list[tuple[int, int]] = []
    for position in feature.get("feature_positions") or []:
        if not isinstance(position, dict):
            continue
        start = int(position.get("beg_seq_id") or 0)
        end = int(position.get("end_seq_id") or start)
        if start <= 0 or end <= 0:
            continue
        mapped.extend(_map_entity_range_to_ref(start, end, aligned_regions))
    return mapped


def _feature_ranges(entity: dict[str, Any], markers: tuple[str, ...]) -> list[tuple[int, int]]:
    aligned_regions = []
    for align in entity.get("rcsb_polymer_entity_align") or []:
        if str(align.get("reference_database_name") or "").lower() != "uniprot":
            continue
        aligned_regions.extend(align.get("aligned_regions") or [])

    ranges: list[tuple[int, int]] = []
    for feature in entity.get("rcsb_polymer_entity_feature") or []:
        if not isinstance(feature, dict):
            continue
        text = " ".join([str(feature.get("name") or ""), str(feature.get("feature_id") or "")]).lower()
        if any(marker in text for marker in markers):
            ranges.extend(_map_feature_positions(feature, aligned_regions))
    return ranges


def _modelled_ref_range(entity: dict[str, Any]) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for align in entity.get("rcsb_polymer_entity_align") or []:
        if str(align.get("reference_database_name") or "").lower() != "uniprot":
            continue
        for region in align.get("aligned_regions") or []:
            ref_start = int(region.get("ref_beg_seq_id") or 0)
            length = int(region.get("length") or 0)
            if ref_start > 0 and length > 0:
                spans.append((ref_start, ref_start + length - 1))
    if not spans:
        return None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def _core_windows(
    barrel_ranges: list[tuple[int, int]],
    *,
    count: int,
    size: int,
) -> list[tuple[int, int]]:
    if not barrel_ranges:
        return []
    start = min(lo for lo, _hi in barrel_ranges)
    end = max(hi for _lo, hi in barrel_ranges)
    if start > end:
        return []
    size = max(3, int(size))
    count = max(1, int(count))
    domain_len = end - start + 1
    if domain_len <= size:
        return [(start, end)]
    if count == 1:
        centers = [(start + end) // 2]
    else:
        step = (domain_len - size) / float(count - 1)
        centers = [int(round(start + (size // 2) + i * step)) for i in range(count)]
    windows: list[tuple[int, int]] = []
    half = size // 2
    for center in centers:
        lo = max(start, center - half)
        hi = min(end, lo + size - 1)
        lo = max(start, hi - size + 1)
        windows.append((lo, hi))
    return sorted(set(windows))


def _annotation_has_barrel_core(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
    except Exception:
        return False
    regions = annotation.get("regions", {})
    barrel = regions.get("barrel_core", []) if isinstance(regions, dict) else []
    entries = barrel if isinstance(barrel, list) else [barrel]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("ranges") or entry.get("residues") or entry.get("residue_ids"):
            return True
    return False


def _find_entity_for_row(row: dict[str, str]) -> dict[str, Any]:
    pdb_id = _row_value(row, "pdb_id").upper()
    chain_id = _row_value(row, "pdb_chain")
    uniprot_id = _row_value(row, "uniprot_id")
    entry = _request_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
    entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids") or []
    best: dict[str, Any] | None = None
    for entity_id in entity_ids:
        entity = _request_json(RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id))
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        uniprot_ids = {str(value) for value in identifiers.get("uniprot_ids") or []}
        auth_chains = {str(value) for value in identifiers.get("auth_asym_ids") or []}
        if uniprot_id and uniprot_id not in uniprot_ids:
            continue
        if chain_id and chain_id not in auth_chains:
            continue
        best = entity
        break
    if best is None:
        raise ValueError(f"Could not find matching RCSB polymer entity for {pdb_id} chain={chain_id} uniprot={uniprot_id}")
    return best


def _annotation_path_for(row: dict[str, str], annotation_dir: Path) -> Path:
    target_id = _row_value(row, "target_id")
    if not target_id:
        target_id = f"{_row_value(row, 'family', 'tbdt')}_{_row_value(row, 'uniprot_id').lower()}"
    slug = "".join(c.lower() if c.isalnum() else "_" for c in target_id).strip("_")
    return annotation_dir / f"{slug}.json"


def _existing_annotation_path(row: dict[str, str], manifest_dir: Path, annotation_dir: Path) -> Path | None:
    raw = _row_value(row, "region_annotation_json")
    if raw:
        path = _resolve_manifest_path(raw, manifest_dir)
        if path.exists():
            return path
    target_id = _row_value(row, "target_id")
    if target_id:
        existing = annotation_dir.parent / f"{target_id}.json"
        if existing.exists():
            return existing
    return None


def _build_annotation(
    row: dict[str, str],
    entity: dict[str, Any],
    *,
    barrel_core_window_count: int,
    barrel_core_window_size: int,
    tonb_box_length: int,
) -> dict[str, Any]:
    family = _row_value(row, "family", "tbdt")
    uniprot_id = _row_value(row, "uniprot_id")
    target_id = _row_value(row, "target_id")
    modelled = _modelled_ref_range(entity)
    plug_ranges = _feature_ranges(entity, PLUG_MARKERS)
    barrel_ranges = _feature_ranges(entity, BARREL_MARKERS)
    if not barrel_ranges and modelled is not None:
        model_start, model_end = modelled
        if plug_ranges:
            plug_end = max(end for _start, end in plug_ranges)
            barrel_start = min(model_end, plug_end + 1)
        else:
            barrel_start = model_start + max(0, int((model_end - model_start + 1) * 0.25))
        if barrel_start <= model_end:
            barrel_ranges = [(barrel_start, model_end)]
    core_ranges = _core_windows(
        barrel_ranges,
        count=barrel_core_window_count,
        size=barrel_core_window_size,
    )

    tonb_ranges: list[tuple[int, int]] = []
    if modelled is not None:
        model_start, _model_end = modelled
        tonb_end = model_start + max(1, int(tonb_box_length)) - 1
        if plug_ranges:
            plug_start = min(start for start, _end in plug_ranges)
            tonb_end = min(tonb_end, max(model_start, plug_start - 1))
        tonb_ranges = [(model_start, tonb_end)] if tonb_end >= model_start else []

    return {
        "schema_version": "tbdt_region_annotation_v1",
        "protein": target_id or family,
        "uniprot_id": uniprot_id,
        "organism": "",
        "default_chain": "A",
        "residue_id_namespace": "af2_chain_uniprot_resseq",
        "curation_status": "auto_from_rcsb_polymer_entity_features",
        "notes": [
            "Auto-generated from RCSB polymer entity Pfam/SIFTS metadata.",
            "Barrel core is sampled as evenly spaced windows across the TonB-dependent receptor beta-barrel feature.",
            "TonB box is a heuristic N-terminal modeled segment and must be manually verified before publication-grade evaluation.",
        ],
        "regions": {
            "barrel_core": [{"chain": "A", "ranges": _split_ranges(core_ranges)}],
            "plug": [{"chain": "A", "ranges": _split_ranges(plug_ranges)}],
            "extracellular_loop": [],
            "tonb_box": [{"chain": "A", "ranges": _split_ranges(tonb_ranges)}],
            "substrate_contact": [],
        },
        "eval_regions": ["plug", "extracellular_loop", "tonb_box", "substrate_contact"],
        "loss_weights": {
            "default": 1.0,
            "barrel_core": 0.1,
            "plug": 2.0,
            "extracellular_loop": 2.0,
            "tonb_box": 3.0,
            "substrate_contact": 3.0,
        },
        "references": [
            {"kind": "pdb", "id": _row_value(row, "pdb_id").upper()},
            {"kind": "uniprot", "id": uniprot_id},
        ],
        "auto_metadata": {
            "modelled_ref_range": list(modelled) if modelled else None,
            "plug_ranges": _split_ranges(plug_ranges),
            "barrel_feature_ranges": _split_ranges(barrel_ranges),
            "barrel_core_window_count": len(core_ranges),
        },
    }


def _assign_split(row: dict[str, Any]) -> str:
    split = str(row.get("split") or "").strip()
    if split and split not in {"expansion", "candidate"}:
        return split
    key = f"{row.get('target_id','')}:{row.get('pdb_id','')}:{row.get('state_label','')}"
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _assign_manifest_splits(rows: list[dict[str, Any]]) -> None:
    """Assign stable 70/15/15 splits for rows that came from candidate/expansion pools."""
    candidates = [row for row in rows if str(row.get("split") or "") in {"", "expansion", "candidate"}]
    fixed = [row for row in rows if row not in candidates]
    if not candidates:
        return

    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(str(row.get("pair_id") or "").encode("utf-8")).hexdigest(),
    )
    n = len(ordered)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = n - 2
        n_val = 1
    for idx, row in enumerate(ordered):
        if idx < n_train:
            row["split"] = "train"
        elif idx < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"

    # Preserve explicit seed-style splits if the input already had them.
    for row in fixed:
        row["split"] = str(row.get("split") or "train")


def prepare_training_manifest(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = Path(args.input_manifest)
    fieldnames, rows = _read_csv(input_manifest)
    manifest_dir = input_manifest.parent
    annotation_dir = Path(args.annotation_dir)
    out_manifest = Path(args.out_manifest)

    if "chain" not in fieldnames:
        fieldnames = [*fieldnames, "chain"]
    if "af2_chain" not in fieldnames:
        fieldnames = [*fieldnames, "af2_chain"]
    if "holo_chain" not in fieldnames:
        fieldnames = [*fieldnames, "holo_chain"]

    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    annotation_counts: Counter[str] = Counter()
    seen_pair_ids: set[str] = set()

    for row in rows:
        row = dict(row)
        pair_id = _row_value(row, "pair_id")
        if pair_id in seen_pair_ids:
            skipped.append({"pair_id": pair_id, "reason": "duplicate_pair_id"})
            continue
        seen_pair_ids.add(pair_id)

        if args.exclude_uncertain and _row_value(row, "state_label") == "uncertain":
            skipped.append({"pair_id": pair_id, "reason": "uncertain_state"})
            continue

        if args.require_local_files:
            missing = []
            for key in ("af2_pdb", "experimental_pdb"):
                raw = _row_value(row, key)
                if raw and not _resolve_manifest_path(raw, manifest_dir).exists():
                    missing.append(key)
            if missing:
                skipped.append({"pair_id": pair_id, "reason": f"missing_local_files:{','.join(missing)}"})
                continue

        try:
            annotation_path = _existing_annotation_path(row, manifest_dir, annotation_dir)
            if annotation_path is not None:
                annotation_counts["existing"] += 1
            else:
                annotation_path = _annotation_path_for(row, annotation_dir)
                if not annotation_path.exists():
                    entity = _find_entity_for_row(row)
                    annotation = _build_annotation(
                        row,
                        entity,
                        barrel_core_window_count=args.barrel_core_window_count,
                        barrel_core_window_size=args.barrel_core_window_size,
                        tonb_box_length=args.tonb_box_length,
                    )
                    _write_json(annotation_path, annotation)
                    annotation_counts["generated"] += 1
                elif not _annotation_has_barrel_core(annotation_path):
                    entity = _find_entity_for_row(row)
                    annotation = _build_annotation(
                        row,
                        entity,
                        barrel_core_window_count=args.barrel_core_window_count,
                        barrel_core_window_size=args.barrel_core_window_size,
                        tonb_box_length=args.tonb_box_length,
                    )
                    _write_json(annotation_path, annotation)
                    annotation_counts["regenerated_missing_core"] += 1
                else:
                    annotation_counts["cached"] += 1

            chain = _row_value(row, "pdb_chain")
            row["chain"] = chain
            row["af2_chain"] = "A"
            row["holo_chain"] = chain
            row["region_annotation_json"] = _data_relative(annotation_path)
            notes = _row_value(row, "notes").replace(
                "requires region annotation before supervised training.",
                "auto region annotation generated for supervised smoke training.",
            )
            training_note = "training_manifest_auto_annotation"
            row["notes"] = f"{notes}; {training_note}" if notes else training_note
            prepared.append(row)
        except Exception as exc:
            skipped.append({"pair_id": pair_id, "reason": str(exc)})

    _assign_manifest_splits(prepared)
    _write_csv(out_manifest, fieldnames, prepared)
    report = {
        "input_manifest": str(input_manifest),
        "out_manifest": str(out_manifest),
        "annotation_dir": str(annotation_dir),
        "input_rows": len(rows),
        "prepared_rows": len(prepared),
        "skipped_rows": len(skipped),
        "annotation_counts": dict(annotation_counts),
        "split_counts": dict(Counter(row["split"] for row in prepared)),
        "family_counts": dict(Counter(row.get("family", "unknown") for row in prepared)),
        "state_counts": dict(Counter(row.get("state_label", "unknown") for row in prepared)),
        "substrate_counts": dict(Counter(row.get("substrate_class", "unknown") for row in prepared)),
        "skipped": skipped,
    }
    _write_json(Path(args.report_path), report)
    return report


def main() -> None:
    args = parse_args()
    report = prepare_training_manifest(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
