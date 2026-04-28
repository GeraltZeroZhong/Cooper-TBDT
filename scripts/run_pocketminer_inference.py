#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PocketMiner inference on a manifest of PDB structures.")
    p.add_argument("--manifest", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_cryptic_manifest.csv"))
    p.add_argument("--pocketminer-root", type=Path, default=Path("outputs/pocketminer_holoshift/external/pocketminer_gvp"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pocketminer_holoshift/pocketminer_predictions"))
    p.add_argument("--structure-col", default="apo_pdb")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def residue_order_from_pdb(path: Path, chain_id: str | None) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "_"
            if chain_id and chain != chain_id:
                continue
            key = (chain, int(line[22:26]), line[26].strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "chain": chain,
                    "resseq": key[1],
                    "icode": key[2],
                    "resname": line[17:20].strip(),
                }
            )
    return rows


def import_pocketminer(pm_root: Path):
    src = pm_root / "src"
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    try:
        import tensorflow as tf  # noqa: F401
        from models import MQAModel
        from xtal_predict import make_predictions
    except Exception as exc:  # pragma: no cover - environment-dependent wrapper
        raise RuntimeError(
            "PocketMiner inference requires tensorflow and mdtraj in the active Python environment. "
            "Create the PocketMiner environment from pocketminer.yml or run this wrapper with that Python."
        ) from exc
    return MQAModel, make_predictions


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    MQAModel, make_predictions = import_pocketminer(args.pocketminer_root.resolve())
    model = MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=(16, 100),
        num_layers=4,
        dropout=0.1,
    )
    nn_path = str((args.pocketminer_root / "models" / "pocketminer").resolve())

    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        target_id = row["target_id"]
        pdb_path = Path(row.get(args.structure_col, ""))
        if not pdb_path.exists():
            failures.append({"target_id": target_id, "error": f"missing PDB: {pdb_path}"})
            continue
        try:
            preds = make_predictions([str(pdb_path)], model, nn_path).reshape(-1)
            residues = residue_order_from_pdb(pdb_path, row.get("apo_chain") or None)
            if len(preds) != len(residues):
                failures.append(
                    {
                        "target_id": target_id,
                        "error": f"prediction/residue length mismatch: {len(preds)} vs {len(residues)}",
                    }
                )
                continue
            for idx, (score, residue) in enumerate(zip(preds, residues, strict=True)):
                out_rows.append(
                    {
                        "target_id": target_id,
                        "structure": args.structure_col,
                        "residue_index": idx,
                        "score": float(score),
                        **residue,
                    }
                )
        except Exception as exc:
            failures.append({"target_id": target_id, "error": str(exc)})

    write_csv(args.out_dir / "pocketminer_predictions.csv", out_rows)
    write_csv(args.out_dir / "pocketminer_prediction_failures.csv", failures)
    status = {"n_targets": len(rows), "n_prediction_rows": len(out_rows), "n_failures": len(failures)}
    (args.out_dir / "pocketminer_prediction_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
