from __future__ import annotations

from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1

STANDARD_AA = {
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
}


class StructureParser:
    def __init__(self) -> None:
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)
        self.last_error: Exception | None = None

    def _get_structure(self, file_path: str):
        lower_path = file_path.lower()
        parser = self.pdb_parser if lower_path.endswith((".pdb", ".ent")) else self.cif_parser
        return parser.get_structure("protein", file_path)

    def parse_ca_structure(self, file_path: str, *, strict: bool = False) -> dict[str, dict[str, Any]] | None:
        self.last_error = None
        try:
            structure = self._get_structure(file_path)
            model = next(structure.get_models())
        except Exception as exc:
            self.last_error = exc
            if strict:
                raise ValueError(f"Failed to parse structure {file_path!r}: {exc}") from exc
            return None

        chains_data: dict[str, dict[str, Any]] = {}
        for chain in model:
            coords, plddts, residue_ids, residue_names, seq_chars = [], [], [], [], []
            for res in chain:
                resname = res.get_resname().strip().upper()
                if res.id[0] != " " or resname not in STANDARD_AA or "CA" not in res:
                    continue
                ca = res["CA"]
                coords.append(ca.get_coord())
                plddts.append(float(ca.get_bfactor()))

                ins_code = res.id[2].strip()
                residue_ids.append(format_residue_id(chain.id, int(res.id[1]), ins_code))
                residue_names.append(resname)
                try:
                    seq_chars.append(seq1(resname))
                except Exception:
                    seq_chars.append("X")

            if len(coords) < 15:
                continue

            chains_data[chain.id] = {
                "coords": np.asarray(coords, dtype=np.float32),
                "plddts": np.asarray(plddts, dtype=np.float32),
                "residue_ids": residue_ids,
                "residue_names": residue_names,
                "sequence": "".join(seq_chars),
            }
        return chains_data if chains_data else None


def format_residue_id(chain_id: str, resseq: int, insertion_code: str = "") -> str:
    return f"{chain_id}_{resseq}{insertion_code.strip()}"


def parse_residue_id(full_id: str) -> tuple[str, int, str]:
    """Split parser residue_id format '<chain>_<resseq><icode>'.

    Chain IDs may contain underscores in mmCIF files, so split from the right.
    """
    if "_" not in full_id:
        return full_id, 0, ""
    chain, raw = full_id.rsplit("_", 1)
    idx = 0
    while idx < len(raw) and (raw[idx].isdigit() or (idx == 0 and raw[idx] == "-")):
        idx += 1
    resseq = int(raw[:idx]) if idx > 0 else 0
    icode = raw[idx:].strip() if idx < len(raw) else ""
    return chain, resseq, icode[:1]


def select_chain(chains: dict[str, dict[str, Any]], chain_id: str | None = None) -> tuple[str, dict[str, Any]]:
    if not chains:
        raise ValueError("No parsed chains available.")
    if chain_id is not None:
        if chain_id not in chains:
            raise ValueError(f"Requested chain_id={chain_id} not found. Available: {list(chains.keys())}")
        return chain_id, chains[chain_id]
    best_id = max(chains.keys(), key=lambda cid: len(chains[cid]["coords"]))
    return best_id, chains[best_id]
