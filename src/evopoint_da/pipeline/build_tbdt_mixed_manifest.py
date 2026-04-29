from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from evopoint_da.pipeline.fetch_tbdt_structures import (
    DEFAULT_AF_VERSION,
    FAMILY_TERMS,
    RCSB_PDB_URL,
    RCSB_SEARCH_URL,
    _download_afdb,
    _family_from_text,
    _state_label,
    _substrate_class,
)

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

GOLD_ANNOTATION_IDS = (
    "PF00593",
    "PF07715",
    "IPR000531",
    "IPR012910",
    "IPR010105",
    "IPR036942",
)
GOLD_FULL_TEXT_TERMS = (
    '"TonB-dependent receptor"',
    '"TonB-dependent transporter"',
    '"siderophore receptor"',
    '"outer membrane receptor"',
)
BRONZE_UNIPROT_QUERY = "(xref:pfam-PF00593) AND (xref:pfam-PF07715) AND (length:[450 TO 1000])"
TBDT_MARKERS = (
    "tonb-dependent",
    "tonb dependent",
    "tonb_dep_rec",
    "siderophore receptor",
    "outer membrane receptor",
)
BETA_BARREL_MARKERS = (
    "transmembrane proteins: beta-barrel",
    "beta-barrel transmembrane",
    "beta-barrel membrane",
    "outer membrane",
)

MIXED_FIELDNAMES = [
    "target_id",
    "pair_id",
    "family",
    "gene_name",
    "uniprot_id",
    "pdb_id",
    "pdb_chain",
    "af2_pdb",
    "experimental_pdb",
    "state",
    "state_label",
    "substrate",
    "substrate_class",
    "ligand_ccd",
    "ligand_sdf",
    "reference_ligand_sdf",
    "has_tonb",
    "tonb_chain",
    "resolution",
    "method",
    "region_annotation_json",
    "split",
    "notes",
    "task_type",
    "evidence_level",
    "label_weight",
    "source_db",
    "source_query",
    "rcsb_polymer_entity_id",
    "sequence_length",
    "reference_coverage",
    "mutation_count",
    "deletion_count",
    "split_group_id",
    "af_version",
    "download_status",
]

POLYMER_ENTITY_QUERY = """
query($ids:[String!]!) {
  polymer_entities(entity_ids:$ids) {
    rcsb_id
    rcsb_polymer_entity {
      pdbx_description
    }
    entity_poly {
      pdbx_seq_one_letter_code_can
      rcsb_sample_sequence_length
      rcsb_deletion_count
      rcsb_mutation_count
    }
    rcsb_polymer_entity_container_identifiers {
      entry_id
      entity_id
      auth_asym_ids
      uniprot_ids
      reference_sequence_identifiers {
        reference_sequence_coverage
      }
    }
    rcsb_polymer_entity_annotation {
      type
      annotation_id
      name
      provenance_source
    }
  }
}
"""

ENTRY_QUERY = """
query($ids:[String!]!) {
  entries(entry_ids:$ids) {
    rcsb_id
    struct {
      title
    }
    struct_keywords {
      pdbx_keywords
      text
    }
    exptl {
      method
    }
    rcsb_entry_info {
      resolution_combined
      nonpolymer_bound_components
      polymer_entity_count_protein
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a mixed Gold/Silver/Bronze TBDT/βOMP manifest.")
    p.add_argument("--out-manifest", default="data/tbdt_mixed_manifest.csv")
    p.add_argument("--out-gold-manifest", default="data/tbdt_gold_manifest.csv")
    p.add_argument("--out-silver-manifest", default="data/tbdt_silver_manifest.csv")
    p.add_argument("--out-bronze-manifest", default="data/tbdt_bronze_manifest.csv")
    p.add_argument("--report-path", default="artifacts/tbdt_v1/tbdt_mixed_manifest_report.json")
    p.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    p.add_argument("--raw-af2-dir", default="data/raw_af2")
    p.add_argument("--af-version", type=int, default=DEFAULT_AF_VERSION)
    p.add_argument("--resolution-cutoff", type=float, default=3.5)
    p.add_argument("--max-gold", type=int, default=220)
    p.add_argument("--max-silver", type=int, default=320)
    p.add_argument("--max-bronze", type=int, default=600)
    p.add_argument("--gold-search-rows", type=int, default=1000)
    p.add_argument("--silver-search-rows", type=int, default=1600)
    p.add_argument("--min-gold-length", type=int, default=450)
    p.add_argument("--max-gold-length", type=int, default=1000)
    p.add_argument("--min-silver-length", type=int, default=120)
    p.add_argument("--max-silver-length", type=int, default=1400)
    p.add_argument("--min-reference-coverage", type=float, default=0.70)
    p.add_argument("--max-mutations", type=int, default=80)
    p.add_argument("--max-deletions", type=int, default=80)
    p.add_argument("--download-gold", action="store_true")
    p.add_argument("--download-silver", action="store_true")
    p.add_argument("--download-bronze", action="store_true")
    p.add_argument("--download-pae", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MIXED_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MIXED_FIELDNAMES})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _download_url(url: str, path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return {"path": str(path), "url": url, "status": "exists", "bytes": path.stat().st_size}
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    total += len(chunk)
            tmp_path.replace(path)
            return {"path": str(path), "url": url, "status": "downloaded", "bytes": total}
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def _download_rcsb_structure(pdb_id: str, raw_pdb_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    pdb_id = pdb_id.upper()
    pdb_path = raw_pdb_dir / f"{pdb_id}.pdb"
    try:
        result = _download_url(RCSB_PDB_URL.format(pdb_id=pdb_id), pdb_path, overwrite=overwrite)
        return {**result, "format": "pdb", "relative_path": f"raw_pdb/{pdb_id}.pdb"}
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
    cif_path = raw_pdb_dir / f"{pdb_id}.cif"
    result = _download_url(RCSB_CIF_URL.format(pdb_id=pdb_id), cif_path, overwrite=overwrite)
    return {**result, "format": "cif", "relative_path": f"raw_pdb/{pdb_id}.cif"}


def _terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def _full_text(value: str) -> dict[str, Any]:
    return {"type": "terminal", "service": "full_text", "parameters": {"value": value}}


def _group(nodes: list[dict[str, Any]], operator: str = "and") -> dict[str, Any]:
    return {"type": "group", "logical_operator": operator, "nodes": nodes}


def _experimental_filters(resolution_cutoff: float) -> list[dict[str, Any]]:
    return [
        _terminal("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
        _terminal("exptl.method", "in", ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]),
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", float(resolution_cutoff)),
    ]


def _rcsb_search(query: dict[str, Any], *, rows: int) -> tuple[list[str], int]:
    payload = {
        "query": query,
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": int(rows)},
            "results_content_type": ["experimental"],
        },
    }
    response = requests.post(RCSB_SEARCH_URL, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    return [item["identifier"] for item in data.get("result_set", [])], int(data.get("total_count", 0))


def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(RCSB_GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=90)
    response.raise_for_status()
    data = response.json()
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], sort_keys=True))
    return data.get("data") or {}


def _batched(values: list[str], size: int = 100) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _fetch_entities(entity_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for batch in _batched(entity_ids):
        data = _graphql(POLYMER_ENTITY_QUERY, {"ids": batch})
        for entity in data.get("polymer_entities") or []:
            if entity and entity.get("rcsb_id"):
                out[str(entity["rcsb_id"]).upper()] = entity
    return out


def _fetch_entries(entry_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for batch in _batched(sorted(set(entry_ids))):
        data = _graphql(ENTRY_QUERY, {"ids": batch})
        for entry in data.get("entries") or []:
            if entry and entry.get("rcsb_id"):
                out[str(entry["rcsb_id"]).upper()] = entry
    return out


def _annotation_text(entity: dict[str, Any]) -> str:
    chunks: list[str] = []
    for ann in entity.get("rcsb_polymer_entity_annotation") or []:
        if not isinstance(ann, dict):
            continue
        chunks.extend(str(ann.get(key) or "") for key in ("type", "annotation_id", "name", "provenance_source"))
    return " ".join(chunks)


def _entry_text(entry: dict[str, Any]) -> str:
    return json.dumps(
        {
            "struct": entry.get("struct") or {},
            "keywords": entry.get("struct_keywords") or {},
            "ccds": _entry_ccds(entry),
        },
        sort_keys=True,
    )


def _entity_text(entity: dict[str, Any]) -> str:
    return " ".join([
        str(entity.get("rcsb_polymer_entity", {}).get("pdbx_description") or ""),
        _annotation_text(entity),
    ])


def _has_annotation(entity: dict[str, Any], ids: set[str]) -> bool:
    for ann in entity.get("rcsb_polymer_entity_annotation") or []:
        if isinstance(ann, dict) and str(ann.get("annotation_id") or "") in ids:
            return True
    return False


def _is_gold_entity(entity: dict[str, Any]) -> bool:
    text = _entity_text(entity).lower()
    return _has_annotation(entity, set(GOLD_ANNOTATION_IDS)) or any(marker in text for marker in TBDT_MARKERS)


def _is_beta_barrel_membrane_entity(entity: dict[str, Any]) -> bool:
    annotations = entity.get("rcsb_polymer_entity_annotation") or []
    text = _entity_text(entity).lower()
    has_membrane_db = any(
        str(ann.get("type") or "").lower() in {"mpstruc", "opm", "pdbtm", "memprotmd"}
        for ann in annotations
        if isinstance(ann, dict)
    )
    return has_membrane_db and any(marker in text for marker in BETA_BARREL_MARKERS)


def _identifier_parts(entity_id: str) -> tuple[str, str]:
    pdb_id, polymer_entity_id = entity_id.split("_", 1)
    return pdb_id.upper(), polymer_entity_id


def _entity_uniprot(entity: dict[str, Any]) -> str:
    ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids") or []
    return str(ids[0]) if ids else ""


def _entity_chains(entity: dict[str, Any]) -> list[str]:
    ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids") or []
    return [str(value) for value in ids if str(value).strip()]


def _entity_length(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_sample_sequence_length") or 0)


def _entity_coverage(entity: dict[str, Any]) -> float:
    refs = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers") or []
    values = [
        float(ref.get("reference_sequence_coverage"))
        for ref in refs
        if isinstance(ref, dict) and ref.get("reference_sequence_coverage") is not None
    ]
    return max(values) if values else 0.0


def _entity_mutations(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_mutation_count") or 0)


def _entity_deletions(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_deletion_count") or 0)


def _entry_ccds(entry: dict[str, Any]) -> list[str]:
    values = entry.get("rcsb_entry_info", {}).get("nonpolymer_bound_components") or []
    return sorted({str(value).upper() for value in values})


def _entry_resolution(entry: dict[str, Any]) -> float | str:
    values = entry.get("rcsb_entry_info", {}).get("resolution_combined") or []
    return float(min(values)) if values else ""


def _entry_method(entry: dict[str, Any]) -> str:
    exptl = entry.get("exptl") or []
    if exptl and isinstance(exptl[0], dict):
        return str(exptl[0].get("method") or "")
    return ""


def _entry_has_tonb_partner(entries_for_pdb: list[dict[str, Any]], current_entity_id: str) -> bool:
    current = current_entity_id.upper()
    for entity in entries_for_pdb:
        if str(entity.get("rcsb_id") or "").upper() == current:
            continue
        text = _entity_text(entity).lower()
        if "tonb" in text and not _is_gold_entity(entity):
            return True
    return False


def _hash_split(group_id: str) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _clean_family(family: str, *, default: str) -> str:
    family = family.lower().strip()
    return family if family in FAMILY_TERMS or family == "tbdt" else default


def _gene_from_text(text: str) -> str:
    lowered = text.lower()
    for family in FAMILY_TERMS:
        if family in lowered:
            return family
    return ""


def _base_paths(uniprot_id: str, pdb_id: str, af_version: int) -> tuple[str, str]:
    af2 = f"raw_af2/AF-{uniprot_id}-F1-model_v{af_version}.pdb" if uniprot_id else ""
    pdb = f"raw_pdb/{pdb_id}.pdb" if pdb_id else ""
    return af2, pdb


def _row_from_entity(
    *,
    entity_id: str,
    entity: dict[str, Any],
    entry: dict[str, Any],
    entities_by_pdb: dict[str, list[dict[str, Any]]],
    evidence_level: str,
    task_type: str,
    label_weight: float,
    source_query: str,
    af_version: int,
) -> dict[str, Any] | None:
    pdb_id, _polymer_entity_id = _identifier_parts(entity_id)
    uniprot_id = _entity_uniprot(entity)
    chains = _entity_chains(entity)
    if not uniprot_id or not chains:
        return None
    chain = chains[0]
    text = f"{_entry_text(entry)} {_entity_text(entity)}"
    family = _clean_family(_family_from_text(text), default="tbdt" if evidence_level == "gold" else "betaomp")
    gene = _gene_from_text(text)
    ccds = _entry_ccds(entry)
    substrate = _substrate_class(ccds, text)
    has_tonb = _entry_has_tonb_partner(entities_by_pdb.get(pdb_id, []), entity_id)
    state = _state_label(ccds, substrate, has_tonb) if evidence_level == "gold" else "unknown"
    target_id = f"{family}_{uniprot_id.lower()}"
    pair_id = f"{target_id}_{pdb_id.lower()}_{chain.lower()}"
    af2_rel, pdb_rel = _base_paths(uniprot_id, pdb_id, af_version)
    split_group = uniprot_id
    return {
        "target_id": target_id,
        "pair_id": pair_id,
        "family": family,
        "gene_name": gene,
        "uniprot_id": uniprot_id,
        "pdb_id": pdb_id,
        "pdb_chain": chain,
        "af2_pdb": af2_rel,
        "experimental_pdb": pdb_rel,
        "state": state,
        "state_label": state,
        "substrate": substrate,
        "substrate_class": substrate,
        "ligand_ccd": ";".join(ccds),
        "ligand_sdf": "",
        "reference_ligand_sdf": "",
        "has_tonb": str(bool(has_tonb)).lower(),
        "tonb_chain": "",
        "resolution": _entry_resolution(entry),
        "method": _entry_method(entry),
        "region_annotation_json": "",
        "split": _hash_split(split_group),
        "notes": f"{evidence_level} candidate from {source_query}; region annotation required before supervised displacement use.",
        "task_type": task_type,
        "evidence_level": evidence_level,
        "label_weight": float(label_weight),
        "source_db": "RCSB",
        "source_query": source_query,
        "rcsb_polymer_entity_id": entity_id.upper(),
        "sequence_length": _entity_length(entity),
        "reference_coverage": _entity_coverage(entity),
        "mutation_count": _entity_mutations(entity),
        "deletion_count": _entity_deletions(entity),
        "split_group_id": split_group,
        "af_version": int(af_version),
        "download_status": "",
    }


def _quality_pass(
    entity: dict[str, Any],
    *,
    min_length: int,
    max_length: int,
    min_reference_coverage: float,
    max_mutations: int,
    max_deletions: int,
) -> bool:
    length = _entity_length(entity)
    return (
        min_length <= length <= max_length
        and _entity_coverage(entity) >= min_reference_coverage
        and _entity_mutations(entity) <= max_mutations
        and _entity_deletions(entity) <= max_deletions
    )


def _gold_query(args: argparse.Namespace) -> dict[str, Any]:
    annotation_query = _terminal("rcsb_polymer_entity_annotation.annotation_id", "in", list(GOLD_ANNOTATION_IDS))
    text_query = _group([_full_text(term) for term in GOLD_FULL_TEXT_TERMS], "or")
    return _group([_group([annotation_query, text_query], "or"), *_experimental_filters(args.resolution_cutoff)])


def _silver_query(args: argparse.Namespace) -> dict[str, Any]:
    mpstruc_beta = _group([
        _terminal("rcsb_polymer_entity_annotation.type", "exact_match", "mpstruc"),
        _full_text('"TRANSMEMBRANE PROTEINS: BETA-BARREL"'),
    ])
    opm_beta = _group([
        _terminal("rcsb_polymer_entity_annotation.type", "exact_match", "OPM"),
        _full_text('"Beta-barrel transmembrane"'),
    ])
    return _group([_group([mpstruc_beta, opm_beta], "or"), *_experimental_filters(args.resolution_cutoff)])


def _build_rcsb_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_ids, gold_total = _rcsb_search(_gold_query(args), rows=args.gold_search_rows)
    silver_ids, silver_total = _rcsb_search(_silver_query(args), rows=args.silver_search_rows)
    all_ids = list(dict.fromkeys([*gold_ids, *silver_ids]))
    entities = _fetch_entities([identifier.upper() for identifier in all_ids])
    entry_ids = sorted({_identifier_parts(identifier)[0] for identifier in entities})
    entries = _fetch_entries(entry_ids)

    entities_by_pdb: dict[str, list[dict[str, Any]]] = {}
    for entity in entities.values():
        pdb_id, _entity_id = _identifier_parts(str(entity["rcsb_id"]))
        entities_by_pdb.setdefault(pdb_id, []).append(entity)

    rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_uniprot_for_bronze_exclusion: set[str] = set()
    skipped: Counter[str] = Counter()

    for identifier in gold_ids:
        key = identifier.upper()
        entity = entities.get(key)
        if not entity:
            skipped["gold_missing_entity"] += 1
            continue
        if not _is_gold_entity(entity):
            skipped["gold_not_tbdt_after_fetch"] += 1
            continue
        if not _quality_pass(
            entity,
            min_length=args.min_gold_length,
            max_length=args.max_gold_length,
            min_reference_coverage=args.min_reference_coverage,
            max_mutations=args.max_mutations,
            max_deletions=args.max_deletions,
        ):
            skipped["gold_quality_filter"] += 1
            continue
        pdb_id, _ = _identifier_parts(key)
        row = _row_from_entity(
            entity_id=key,
            entity=entity,
            entry=entries.get(pdb_id, {}),
            entities_by_pdb=entities_by_pdb,
            evidence_level="gold",
            task_type="gold_displacement",
            label_weight=1.0,
            source_query="RCSB:TBDT_Pfam_InterPro_text",
            af_version=args.af_version,
        )
        if row is None:
            skipped["gold_missing_uniprot_or_chain"] += 1
            continue
        if row["pair_id"] in seen_pair_ids:
            skipped["gold_duplicate_pair_id"] += 1
            continue
        seen_pair_ids.add(row["pair_id"])
        seen_uniprot_for_bronze_exclusion.add(row["uniprot_id"])
        rows.append(row)
        if sum(1 for r in rows if r["evidence_level"] == "gold") >= args.max_gold:
            break

    gold_entity_ids = {row["rcsb_polymer_entity_id"] for row in rows if row["evidence_level"] == "gold"}
    for identifier in silver_ids:
        key = identifier.upper()
        if key in gold_entity_ids:
            skipped["silver_overlap_gold_entity"] += 1
            continue
        entity = entities.get(key)
        if not entity:
            skipped["silver_missing_entity"] += 1
            continue
        if _is_gold_entity(entity):
            skipped["silver_tbdt_promoted_or_skipped"] += 1
            continue
        if not _is_beta_barrel_membrane_entity(entity):
            skipped["silver_not_beta_barrel_membrane_after_fetch"] += 1
            continue
        if not _quality_pass(
            entity,
            min_length=args.min_silver_length,
            max_length=args.max_silver_length,
            min_reference_coverage=args.min_reference_coverage,
            max_mutations=args.max_mutations,
            max_deletions=args.max_deletions,
        ):
            skipped["silver_quality_filter"] += 1
            continue
        pdb_id, _ = _identifier_parts(key)
        row = _row_from_entity(
            entity_id=key,
            entity=entity,
            entry=entries.get(pdb_id, {}),
            entities_by_pdb=entities_by_pdb,
            evidence_level="silver",
            task_type="silver_betaomp_displacement",
            label_weight=0.35,
            source_query="RCSB:mpstruc_OPM_beta_barrel",
            af_version=args.af_version,
        )
        if row is None:
            skipped["silver_missing_uniprot_or_chain"] += 1
            continue
        if row["pair_id"] in seen_pair_ids:
            skipped["silver_duplicate_pair_id"] += 1
            continue
        seen_pair_ids.add(row["pair_id"])
        seen_uniprot_for_bronze_exclusion.add(row["uniprot_id"])
        rows.append(row)
        if sum(1 for r in rows if r["evidence_level"] == "silver") >= args.max_silver:
            break

    report = {
        "gold_search_total_count": gold_total,
        "gold_search_returned": len(gold_ids),
        "silver_search_total_count": silver_total,
        "silver_search_returned": len(silver_ids),
        "rcsb_entities_fetched": len(entities),
        "rcsb_entries_fetched": len(entries),
        "skipped": dict(skipped),
        "bronze_excluded_uniprots": sorted(seen_uniprot_for_bronze_exclusion),
    }
    return rows, report


def _parse_uniprot_tsv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter="\t")
    return [dict(row) for row in reader]


def _fetch_bronze_uniprot_rows(args: argparse.Namespace, excluded_uniprots: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params: dict[str, Any] = {
        "query": BRONZE_UNIPROT_QUERY,
        "fields": "accession,id,protein_name,gene_names,organism_name,length,xref_alphafolddb,xref_pdb",
        "format": "tsv",
        "size": 500,
    }
    raw_rows: list[dict[str, str]] = []
    next_url: str | None = UNIPROT_SEARCH_URL
    first_total = ""
    page_count = 0
    while next_url and len(raw_rows) < args.max_bronze + len(excluded_uniprots) + 500:
        response = requests.get(next_url, params=params if next_url == UNIPROT_SEARCH_URL else None, timeout=90)
        response.raise_for_status()
        if not first_total:
            first_total = response.headers.get("x-total-results", "")
        raw_rows.extend(_parse_uniprot_tsv(response.text))
        page_count += 1
        link = response.headers.get("Link", "")
        match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        next_url = match.group(1) if match else None
        params = {}
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for raw in raw_rows:
        if len(rows) >= args.max_bronze:
            break
        uniprot_id = str(raw.get("Entry") or "").strip()
        if not uniprot_id:
            skipped["missing_accession"] += 1
            continue
        if uniprot_id in excluded_uniprots:
            skipped["already_in_gold_or_silver"] += 1
            continue
        length = int(raw.get("Length") or 0)
        if not (args.min_gold_length <= length <= args.max_gold_length):
            skipped["length_filter"] += 1
            continue
        text = " ".join(str(raw.get(key) or "") for key in ("Entry Name", "Protein names", "Gene Names", "Organism"))
        family = _clean_family(_family_from_text(text), default="tbdt")
        gene = _gene_from_text(text)
        target_id = f"{family}_{uniprot_id.lower()}"
        pair_id = f"{target_id}_afdb_v{args.af_version}"
        af2_rel, _pdb_rel = _base_paths(uniprot_id, "", args.af_version)
        rows.append(
            {
                "target_id": target_id,
                "pair_id": pair_id,
                "family": family,
                "gene_name": gene,
                "uniprot_id": uniprot_id,
                "pdb_id": "",
                "pdb_chain": "",
                "af2_pdb": af2_rel,
                "experimental_pdb": "",
                "state": "unknown",
                "state_label": "unknown",
                "substrate": "unknown",
                "substrate_class": "unknown",
                "ligand_ccd": "",
                "ligand_sdf": "",
                "reference_ligand_sdf": "",
                "has_tonb": "false",
                "tonb_chain": "",
                "resolution": "",
                "method": "AFDB",
                "region_annotation_json": "",
                "split": _hash_split(uniprot_id),
                "notes": "bronze AFDB-only TBDT homolog from UniProt PF00593/PF07715 query; no experimental displacement label.",
                "task_type": "bronze_pseudo_or_self_supervised",
                "evidence_level": "bronze",
                "label_weight": 0.1,
                "source_db": "UniProt",
                "source_query": "UniProt:PF00593_AND_PF07715",
                "rcsb_polymer_entity_id": "",
                "sequence_length": length,
                "reference_coverage": "",
                "mutation_count": "",
                "deletion_count": "",
                "split_group_id": uniprot_id,
                "af_version": int(args.af_version),
                "download_status": "",
            }
        )
    return rows, {
        "uniprot_query": BRONZE_UNIPROT_QUERY,
        "uniprot_returned_rows": len(raw_rows),
        "uniprot_pages_fetched": page_count,
        "bronze_retained_rows": len(rows),
        "bronze_skipped": dict(skipped),
        "uniprot_total_header": first_total,
    }


def _download_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_pdb_dir = Path(args.raw_pdb_dir)
    raw_af2_dir = Path(args.raw_af2_dir)
    counters = Counter()
    failures: list[dict[str, str]] = []
    for row in rows:
        level = row.get("evidence_level")
        should_download = (
            (level == "gold" and args.download_gold)
            or (level == "silver" and args.download_silver)
            or (level == "bronze" and args.download_bronze)
        )
        if not should_download:
            row["download_status"] = "not_requested"
            continue
        uniprot_id = str(row.get("uniprot_id") or "")
        pdb_id = str(row.get("pdb_id") or "")
        status_parts: list[str] = []
        if pdb_id:
            try:
                pdb_result = _download_rcsb_structure(pdb_id, raw_pdb_dir, overwrite=args.overwrite)
                row["experimental_pdb"] = pdb_result["relative_path"]
                status_parts.append(f"{pdb_result['format']}:{pdb_result['status']}")
                counters[f"{pdb_result['format']}_{pdb_result['status']}"] += 1
            except Exception as exc:
                failures.append({"pair_id": row["pair_id"], "kind": "pdb", "id": pdb_id, "error": str(exc)})
                status_parts.append("structure:failed")
                counters["structure_failed"] += 1
        if uniprot_id:
            try:
                af_result = _download_afdb(
                    uniprot_id,
                    raw_af2_dir,
                    af_version=args.af_version,
                    download_pae=args.download_pae,
                    overwrite=args.overwrite,
                )
                model_status = af_result["model"]["status"]
                status_parts.append(f"afdb:{model_status}")
                counters[f"afdb_{model_status}"] += 1
            except Exception as exc:
                failures.append({"pair_id": row["pair_id"], "kind": "afdb", "id": uniprot_id, "error": str(exc)})
                status_parts.append("afdb:failed")
                counters["afdb_failed"] += 1
        row["download_status"] = ";".join(status_parts) if status_parts else "nothing_to_download"
    return {"download_counts": dict(counters), "download_failures": failures}


def build_mixed_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rcsb_rows, rcsb_report = _build_rcsb_rows(args)
    excluded_uniprots = set(rcsb_report.get("bronze_excluded_uniprots", []))
    bronze_rows, bronze_report = _fetch_bronze_uniprot_rows(args, excluded_uniprots)
    rows = [*rcsb_rows, *bronze_rows]
    download_report = _download_rows(args, rows)
    _write_csv(Path(args.out_manifest), rows)
    tier_outputs = {
        "gold": args.out_gold_manifest,
        "silver": args.out_silver_manifest,
        "bronze": args.out_bronze_manifest,
    }
    for tier, path in tier_outputs.items():
        if path:
            _write_csv(Path(path), [row for row in rows if row["evidence_level"] == tier])

    report = {
        "out_manifest": args.out_manifest,
        "tier_manifests": tier_outputs,
        "af_version": int(args.af_version),
        "resolution_cutoff": float(args.resolution_cutoff),
        "row_count": len(rows),
        "counts_by_evidence": dict(Counter(row["evidence_level"] for row in rows)),
        "counts_by_task_type": dict(Counter(row["task_type"] for row in rows)),
        "counts_by_split": dict(Counter(row["split"] for row in rows)),
        "counts_by_family": dict(Counter(row["family"] for row in rows)),
        "unique_uniprots": len({row["uniprot_id"] for row in rows if row.get("uniprot_id")}),
        "unique_pdb_ids": len({row["pdb_id"] for row in rows if row.get("pdb_id")}),
        "rcsb": rcsb_report,
        "uniprot": bronze_report,
        "downloads": download_report,
    }
    _write_json(Path(args.report_path), report)
    return report


def main() -> None:
    args = parse_args()
    report = build_mixed_manifest(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
