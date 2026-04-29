from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import requests

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
AFDB_MODEL_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.pdb"
AFDB_PAE_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v{version}.json"

DEFAULT_AF_VERSION = 6
DEFAULT_TERMS = (
    "TonB-dependent transporter",
    "tonb dependent transporter",
    "TonB-dependent receptor",
    "fecA",
    "fhuA",
    "btuB",
    "fepA",
    "fyuA",
    "cirA",
    "ferric citrate transporter",
    "ferrichrome receptor",
    "cobalamin transporter",
    "heme receptor",
    "siderophore receptor",
)
FAMILY_TERMS = ("feca", "fhua", "btub", "fepa", "fyua", "cira")
LIGAND_CLASS_BY_CCD = {
    "B12": "cobalamin",
    "CNC": "cobalamin",
    "CYN": "cobalamin",
    "CBO": "cobalamin",
    "COB": "cobalamin",
    "CIT": "ferric_citrate",
    "FLC": "ferric_citrate",
    "HEA": "heme",
    "HEB": "heme",
    "HEC": "heme",
    "HEM": "heme",
    "HEO": "heme",
}
DETERGENT_OR_BUFFER_CCD = {
    "ACT",
    "CL",
    "DMS",
    "EDO",
    "GOL",
    "GP1",
    "HOH",
    "LDA",
    "LIL",
    "LIM",
    "MG",
    "MYR",
    "NA",
    "PEG",
    "PO4",
    "SO4",
}
METAL_CCD = {"CA", "CO", "CU", "FE", "K", "MG", "MN", "NI", "ZN"}
BAD_COMPLEX_RE = re.compile(r"\b(antibody|fab|nanobody|fusion|mbp|lysozyme)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and discover TBDT AFDB/RCSB structures.")
    sub = p.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("download-manifest", help="Download AFDB v6 and RCSB PDB files referenced by a manifest.")
    seed.add_argument("--manifest", default="data/tbdt_state_manifest.csv")
    seed.add_argument("--out-manifest", default="")
    seed.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    seed.add_argument("--raw-af2-dir", default="data/raw_af2")
    seed.add_argument("--af-version", type=int, default=DEFAULT_AF_VERSION)
    seed.add_argument("--download-pae", action="store_true")
    seed.add_argument("--overwrite", action="store_true")
    seed.add_argument("--report-path", default="artifacts/tbdt_v1/download_manifest_report.json")

    discover = sub.add_parser("discover", help="Search RCSB for TBDT-like entries and optionally download inputs.")
    discover.add_argument("--out-manifest", default="data/tbdt_expansion_manifest.csv")
    discover.add_argument("--raw-pdb-dir", default="data/raw_pdb")
    discover.add_argument("--raw-af2-dir", default="data/raw_af2")
    discover.add_argument("--af-version", type=int, default=DEFAULT_AF_VERSION)
    discover.add_argument("--resolution-cutoff", type=float, default=3.5)
    discover.add_argument("--max-search-results", type=int, default=120)
    discover.add_argument("--max-retained", type=int, default=40)
    discover.add_argument("--min-sequence-length", type=int, default=450)
    discover.add_argument("--min-reference-coverage", type=float, default=0.70)
    discover.add_argument("--max-deletions", type=int, default=80)
    discover.add_argument("--max-mutations", type=int, default=80)
    discover.add_argument("--download", action="store_true")
    discover.add_argument("--download-pae", action="store_true")
    discover.add_argument("--require-af2", action="store_true")
    discover.add_argument("--overwrite", action="store_true")
    discover.add_argument("--report-path", default="artifacts/tbdt_v1/rcsb_discovery_report.json")
    discover.add_argument("--term", action="append", dest="terms", help="Additional RCSB full-text search term.")
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


def _request_json(url: str, *, method: str = "get", payload: dict[str, Any] | None = None) -> Any:
    if method == "post":
        response = requests.post(url, json=payload, timeout=60)
    else:
        response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _download_file(url: str, path: Path, *, overwrite: bool = False) -> dict[str, Any]:
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


def _af2_path(raw_af2_dir: Path, uniprot_id: str, af_version: int) -> Path:
    return raw_af2_dir / f"AF-{uniprot_id}-F1-model_v{af_version}.pdb"


def _pae_path(raw_af2_dir: Path, uniprot_id: str) -> Path:
    return raw_af2_dir / f"AF-{uniprot_id}.json"


def _download_afdb(
    uniprot_id: str,
    raw_af2_dir: Path,
    *,
    af_version: int,
    download_pae: bool,
    overwrite: bool,
) -> dict[str, Any]:
    model_url = AFDB_MODEL_URL.format(uniprot=uniprot_id, version=af_version)
    model_path = _af2_path(raw_af2_dir, uniprot_id, af_version)
    result = {"uniprot_id": uniprot_id, "model": None, "pae": None, "ok": False}
    result["model"] = _download_file(model_url, model_path, overwrite=overwrite)
    result["ok"] = True
    if download_pae:
        pae_url = AFDB_PAE_URL.format(uniprot=uniprot_id, version=af_version)
        result["pae"] = _download_file(pae_url, _pae_path(raw_af2_dir, uniprot_id), overwrite=overwrite)
    return result


def _download_pdb(pdb_id: str, raw_pdb_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    pdb_id = pdb_id.upper()
    return _download_file(RCSB_PDB_URL.format(pdb_id=pdb_id), raw_pdb_dir / f"{pdb_id}.pdb", overwrite=overwrite)


def download_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    fieldnames, rows = _read_csv(manifest_path)
    raw_pdb_dir = Path(args.raw_pdb_dir)
    raw_af2_dir = Path(args.raw_af2_dir)

    downloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        pdb_id = str(row.get("pdb_id") or "").strip().upper()
        uniprot_id = str(row.get("uniprot_id") or "").strip()
        if pdb_id:
            row["experimental_pdb"] = f"raw_pdb/{pdb_id}.pdb"
            try:
                downloads.append({"kind": "pdb", "id": pdb_id, **_download_pdb(pdb_id, raw_pdb_dir, overwrite=args.overwrite)})
            except Exception as exc:
                failures.append({"kind": "pdb", "id": pdb_id, "error": str(exc)})
        if uniprot_id:
            row["af2_pdb"] = f"raw_af2/AF-{uniprot_id}-F1-model_v{args.af_version}.pdb"
            try:
                downloads.append(
                    {
                        "kind": "afdb",
                        "id": uniprot_id,
                        **_download_afdb(
                            uniprot_id,
                            raw_af2_dir,
                            af_version=args.af_version,
                            download_pae=args.download_pae,
                            overwrite=args.overwrite,
                        ),
                    }
                )
            except Exception as exc:
                failures.append({"kind": "afdb", "id": uniprot_id, "error": str(exc)})
        updated_rows.append(row)

    out_manifest = Path(args.out_manifest) if args.out_manifest else manifest_path
    _write_csv(out_manifest, fieldnames, updated_rows)
    report = {
        "manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "af_version": int(args.af_version),
        "rows": len(rows),
        "downloads": downloads,
        "failures": failures,
    }
    _write_json(Path(args.report_path), report)
    return report


def _search_query(terms: tuple[str, ...], resolution_cutoff: float, rows: int) -> dict[str, Any]:
    text_nodes = [
        {"type": "terminal", "service": "full_text", "parameters": {"value": term}}
        for term in terms
    ]
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "group", "logical_operator": "or", "nodes": text_nodes},
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "in",
                        "value": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": float(resolution_cutoff),
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": int(rows)},
            "results_content_type": ["experimental"],
        },
    }


def _entry_resolution(entry: dict[str, Any]) -> float | None:
    values = entry.get("rcsb_entry_info", {}).get("resolution_combined") or []
    if not values:
        return None
    return float(min(values))


def _entry_method(entry: dict[str, Any]) -> str:
    exptl = entry.get("exptl") or []
    if exptl and isinstance(exptl[0], dict):
        return str(exptl[0].get("method") or "")
    return str(entry.get("rcsb_entry_info", {}).get("experimental_method") or "")


def _nonpolymer_ccds(entry: dict[str, Any]) -> list[str]:
    info = entry.get("rcsb_entry_info", {})
    ccds = info.get("nonpolymer_bound_components") or []
    return sorted({str(ccd).upper() for ccd in ccds})


def _family_from_text(text: str) -> str:
    lowered = text.lower()
    for family in FAMILY_TERMS:
        if family in lowered:
            return family
    return "tbdt"


def _is_tbdt_entity(entity: dict[str, Any]) -> bool:
    desc = str(entity.get("rcsb_polymer_entity", {}).get("pdbx_description") or "")
    names = entity.get("rcsb_polymer_entity", {}).get("rcsb_polymer_name_combined", {}).get("names") or []
    feature_names = [
        str(feature.get("name") or "")
        for feature in entity.get("rcsb_polymer_entity_feature", []) or []
        if isinstance(feature, dict)
    ]
    text = " ".join([desc, *map(str, names), *feature_names])
    lowered = text.lower()
    if BAD_COMPLEX_RE.search(text):
        return False
    return "tonb-dependent" in lowered or "tonb dependent" in lowered or any(term in lowered for term in FAMILY_TERMS)


def _entity_uniprot(entity: dict[str, Any]) -> str:
    ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids") or []
    return str(ids[0]) if ids else ""


def _entity_chain(entity: dict[str, Any]) -> str:
    ids = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids") or []
    return str(ids[0]) if ids else ""


def _entity_coverage(entity: dict[str, Any]) -> float:
    refs = entity.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers") or []
    coverages = [
        float(ref.get("reference_sequence_coverage"))
        for ref in refs
        if isinstance(ref, dict) and ref.get("reference_sequence_coverage") is not None
    ]
    return max(coverages) if coverages else 0.0


def _entity_length(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_sample_sequence_length") or 0)


def _entity_deletions(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_deletion_count") or 0)


def _entity_mutations(entity: dict[str, Any]) -> int:
    return int(entity.get("entity_poly", {}).get("rcsb_mutation_count") or 0)


def _entry_has_tonb_entity(entities: list[dict[str, Any]]) -> bool:
    for entity in entities:
        desc = str(entity.get("rcsb_polymer_entity", {}).get("pdbx_description") or "")
        lowered = desc.lower()
        if "tonb" in lowered and "dependent receptor" not in lowered and "transporter" not in lowered:
            return True
    return False


def _substrate_class(ccds: list[str], text: str) -> str:
    for ccd in ccds:
        if ccd in LIGAND_CLASS_BY_CCD:
            return LIGAND_CLASS_BY_CCD[ccd]
    lowered = text.lower()
    if not ccds:
        bound_hint = any(marker in lowered for marker in ("bound", "complex", "substrate"))
        if not bound_hint or any(marker in lowered for marker in ("no ligand", "ligand-free", "unliganded", "apo")):
            return "none"
    if "citrate" in lowered or "dicitrate" in lowered:
        return "ferric_citrate"
    if "ferrichrome" in lowered:
        return "ferrichrome"
    if "enterobactin" in lowered:
        return "enterobactin"
    if "cobalamin" in lowered or "vitamin b12" in lowered:
        return "cobalamin"
    if "heme" in lowered:
        return "heme"
    if "siderophore" in lowered:
        return "unknown_siderophore"
    return "none"


def _state_label(ccds: list[str], substrate_class: str, has_tonb: bool) -> str:
    if has_tonb:
        return "tonb_bound"
    relevant = [ccd for ccd in ccds if ccd not in DETERGENT_OR_BUFFER_CCD]
    if not relevant:
        return "substrate_bound" if substrate_class not in {"none", "unknown"} else "apo"
    if substrate_class == "ferric_citrate" and "FE" in ccds:
        return "productive_substrate_bound"
    if substrate_class not in {"none", "unknown"}:
        return "substrate_bound"
    if relevant and all(ccd in METAL_CCD for ccd in relevant):
        return "metal_only"
    return "uncertain"


def _manifest_fieldnames() -> list[str]:
    return [
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
    ]


def discover(args: argparse.Namespace) -> dict[str, Any]:
    terms = tuple(dict.fromkeys((*DEFAULT_TERMS, *(args.terms or []))))
    query = _search_query(terms, args.resolution_cutoff, args.max_search_results)
    search = _request_json(RCSB_SEARCH_URL, method="post", payload=query)
    entry_ids = [item["identifier"].upper() for item in search.get("result_set", [])]

    raw_pdb_dir = Path(args.raw_pdb_dir)
    raw_af2_dir = Path(args.raw_af2_dir)
    rows: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for pdb_id in entry_ids:
        if len(rows) >= args.max_retained:
            break
        try:
            entry = _request_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
            container = entry.get("rcsb_entry_container_identifiers", {})
            entity_ids = container.get("polymer_entity_ids") or []
            protein_entity_count = int(entry.get("rcsb_entry_info", {}).get("polymer_entity_count_protein") or 0)
            if protein_entity_count > 2:
                skipped.append({"pdb_id": pdb_id, "reason": "too_many_protein_entities"})
                continue

            entities = [
                _request_json(RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id))
                for entity_id in entity_ids
            ]
            has_tonb = _entry_has_tonb_entity(entities)
            ccds = _nonpolymer_ccds(entry)
            entry_text = json.dumps(
                {
                    "struct": entry.get("struct", {}),
                    "keywords": entry.get("struct_keywords", {}),
                    "ligands": ccds,
                },
                sort_keys=True,
            )

            for entity in entities:
                if len(rows) >= args.max_retained:
                    break
                if not _is_tbdt_entity(entity):
                    continue
                uniprot_id = _entity_uniprot(entity)
                chain_id = _entity_chain(entity)
                if not uniprot_id or not chain_id:
                    skipped.append({"pdb_id": pdb_id, "reason": "missing_uniprot_or_chain"})
                    continue
                if _entity_length(entity) < args.min_sequence_length:
                    skipped.append({"pdb_id": pdb_id, "reason": "short_entity"})
                    continue
                if _entity_coverage(entity) < args.min_reference_coverage:
                    skipped.append({"pdb_id": pdb_id, "reason": "low_reference_coverage"})
                    continue
                if _entity_deletions(entity) > args.max_deletions or _entity_mutations(entity) > args.max_mutations:
                    skipped.append({"pdb_id": pdb_id, "reason": "high_deletion_or_mutation_count"})
                    continue

                family = _family_from_text(entry_text + " " + json.dumps(entity.get("rcsb_polymer_entity", {})))
                substrate = _substrate_class(ccds, entry_text)
                state = _state_label(ccds, substrate, has_tonb)

                af2_rel = f"raw_af2/AF-{uniprot_id}-F1-model_v{args.af_version}.pdb"
                pdb_rel = f"raw_pdb/{pdb_id}.pdb"
                af2_ok = True
                if args.download:
                    try:
                        downloads.append({"kind": "pdb", "id": pdb_id, **_download_pdb(pdb_id, raw_pdb_dir, overwrite=args.overwrite)})
                    except Exception as exc:
                        skipped.append({"pdb_id": pdb_id, "reason": f"pdb_download_failed: {exc}"})
                        continue
                    try:
                        downloads.append(
                            {
                                "kind": "afdb",
                                "id": uniprot_id,
                                **_download_afdb(
                                    uniprot_id,
                                    raw_af2_dir,
                                    af_version=args.af_version,
                                    download_pae=args.download_pae,
                                    overwrite=args.overwrite,
                                ),
                            }
                        )
                    except Exception as exc:
                        af2_ok = False
                        skipped.append({"pdb_id": pdb_id, "reason": f"afdb_download_failed: {exc}"})
                if args.require_af2 and not af2_ok:
                    continue

                target_id = f"{family}_{uniprot_id.lower()}"
                pair_id = f"{target_id}_{pdb_id.lower()}_{chain_id.lower()}"
                rows.append(
                    {
                        "target_id": target_id,
                        "pair_id": pair_id,
                        "family": family,
                        "gene_name": family if family != "tbdt" else "",
                        "uniprot_id": uniprot_id,
                        "pdb_id": pdb_id,
                        "pdb_chain": chain_id,
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
                        "resolution": _entry_resolution(entry) or "",
                        "method": _entry_method(entry),
                        "region_annotation_json": "",
                        "split": "expansion",
                        "notes": "RCSB Search API candidate; requires region annotation before supervised training.",
                    }
                )
        except Exception as exc:
            skipped.append({"pdb_id": pdb_id, "reason": str(exc)})

    _write_csv(Path(args.out_manifest), _manifest_fieldnames(), rows)
    report = {
        "query_terms": terms,
        "af_version": int(args.af_version),
        "resolution_cutoff": float(args.resolution_cutoff),
        "search_total_count": int(search.get("total_count", 0)),
        "searched_entries": len(entry_ids),
        "retained_rows": len(rows),
        "output_manifest": args.out_manifest,
        "downloads": downloads,
        "skipped": skipped,
    }
    _write_json(Path(args.report_path), report)
    return report


def main() -> None:
    args = parse_args()
    if args.command == "download-manifest":
        report = download_manifest(args)
    elif args.command == "discover":
        report = discover(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
