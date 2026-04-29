"""Build non-neural template-transfer baselines for TBDT displacement prediction.

The baselines use only donor samples from training-like splits. Donor AF2
scaffolds are aligned to the target AF2 scaffold, donor displacement vectors
are rotated into the target frame, and vectors are copied onto sequence-aligned
target residues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio import Align

from evopoint_da.data.alignment import kabsch_rotation
from evopoint_da.data.dataset import build_split_file_lists


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}


@dataclass(frozen=True)
class PairSample:
    path: Path
    stem: str
    sequence: str
    af2_pos: torch.Tensor
    y_delta: torch.Tensor
    barrel_core_mask: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateTransfer:
    donor: PairSample
    identity: float
    target_coverage: float
    aligned_residues: int
    same_family: bool
    same_state: bool
    same_substrate: bool
    prediction: torch.Tensor
    n_predicted_residues: int
    fit_region: str
    fit_residues: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TBDT template-transfer coordinate baselines.")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--pair-dir", default="data/processed_tbdt_gold_pairs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--donor-split", action="append", default=[], choices=["train", "val", "test", "all"])
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-identity", type=float, default=0.15)
    parser.add_argument("--min-target-coverage", type=float, default=0.30)
    parser.add_argument("--average-top-k", type=int, default=10)
    parser.add_argument("--allow-same-uniprot", action="store_true")
    parser.add_argument("--output-root", default="artifacts/tbdt_v1/template_baselines")
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _load_pair(path: Path) -> PairSample:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    sequence = str(raw["sequence"])
    n = len(sequence)
    af2_pos = torch.as_tensor(raw["af2_pos"], dtype=torch.float32).reshape(-1, 3)
    y_delta = torch.as_tensor(raw["y_delta"], dtype=torch.float32).reshape(-1, 3)
    if af2_pos.size(0) != n or y_delta.size(0) != n:
        raise ValueError(f"{path} has inconsistent sequence/coordinate lengths.")
    barrel = torch.as_tensor(raw.get("barrel_core_mask", torch.zeros(n)), dtype=torch.bool).reshape(-1)
    if barrel.numel() != n:
        barrel = torch.zeros(n, dtype=torch.bool)
    return PairSample(
        path=path,
        stem=path.stem,
        sequence=sequence,
        af2_pos=af2_pos,
        y_delta=y_delta,
        barrel_core_mask=barrel,
        metadata=metadata,
    )


def _row(sample: PairSample) -> dict[str, Any]:
    row = sample.metadata.get("manifest_row", {})
    return row if isinstance(row, dict) else {}


def _field(sample: PairSample, key: str) -> str:
    row = _row(sample)
    value = row.get(key) or sample.metadata.get(key)
    return str(value or "").strip().lower()


def _uniprot(sample: PairSample) -> str:
    return _field(sample, "uniprot_id").upper()


def _align_pairs(query: str, target: str) -> tuple[list[tuple[int, int]], float, float]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -3.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = next(iter(aligner.align(query, target)))
    query_blocks, target_blocks = alignment.aligned
    pairs: list[tuple[int, int]] = []
    for (q_start, q_end), (t_start, t_end) in zip(query_blocks, target_blocks, strict=False):
        length = min(q_end - q_start, t_end - t_start)
        pairs.extend((q_start + offset, t_start + offset) for offset in range(length))
    if not pairs:
        return [], 0.0, 0.0
    matches = sum(1 for q_idx, t_idx in pairs if query[q_idx] == target[t_idx])
    identity = matches / float(len(pairs))
    target_coverage = len({t_idx for _q_idx, t_idx in pairs}) / float(max(1, len(target)))
    return pairs, identity, target_coverage


def _rotate_vectors(vectors: torch.Tensor, rotation: np.ndarray) -> torch.Tensor:
    r = torch.as_tensor(rotation, dtype=torch.float32)
    return vectors.float() @ r.T


def _transfer_prediction(donor: PairSample, target: PairSample) -> dict[str, Any] | None:
    pairs, identity, target_coverage = _align_pairs(donor.sequence, target.sequence)
    if len(pairs) < 3:
        return None

    donor_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    target_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    core_mask = donor.barrel_core_mask[donor_idx] & target.barrel_core_mask[target_idx]
    if int(core_mask.sum().item()) >= 3:
        fit_donor = donor_idx[core_mask]
        fit_target = target_idx[core_mask]
        fit_region = "barrel_core"
    else:
        fit_donor = donor_idx
        fit_target = target_idx
        fit_region = "all_aligned"

    if fit_donor.numel() < 3:
        return None
    rotation, _translation = kabsch_rotation(
        donor.af2_pos[fit_donor].numpy(),
        target.af2_pos[fit_target].numpy(),
    )
    rotated = _rotate_vectors(donor.y_delta, rotation)

    pred = torch.zeros_like(target.y_delta)
    pred[target_idx] = rotated[donor_idx]
    return {
        "prediction": pred,
        "identity": float(identity),
        "target_coverage": float(target_coverage),
        "aligned_residues": int(len(pairs)),
        "n_predicted_residues": int(torch.unique(target_idx).numel()),
        "fit_region": fit_region,
        "fit_residues": int(fit_donor.numel()),
    }


def _candidate(donor: PairSample, target: PairSample, args: argparse.Namespace) -> CandidateTransfer | None:
    if not args.allow_same_uniprot and _uniprot(donor) and _uniprot(donor) == _uniprot(target):
        return None
    transfer = _transfer_prediction(donor, target)
    if transfer is None:
        return None
    identity = float(transfer["identity"])
    target_coverage = float(transfer["target_coverage"])
    if identity < float(args.min_identity) or target_coverage < float(args.min_target_coverage):
        return None
    same_family = _field(donor, "family") == _field(target, "family")
    same_state = (_field(donor, "state_label") or _field(donor, "state")) == (
        _field(target, "state_label") or _field(target, "state")
    )
    same_substrate = (_field(donor, "substrate_class") or _field(donor, "substrate")) == (
        _field(target, "substrate_class") or _field(target, "substrate")
    )
    return CandidateTransfer(
        donor=donor,
        identity=identity,
        target_coverage=target_coverage,
        aligned_residues=int(transfer["aligned_residues"]),
        same_family=same_family,
        same_state=same_state,
        same_substrate=same_substrate,
        prediction=transfer["prediction"],
        n_predicted_residues=int(transfer["n_predicted_residues"]),
        fit_region=str(transfer["fit_region"]),
        fit_residues=int(transfer["fit_residues"]),
    )


def _candidate_sort_key(c: CandidateTransfer) -> tuple[int, int, int, float, float, int]:
    return (
        int(c.same_family),
        int(c.same_state),
        int(c.same_substrate),
        c.identity,
        c.target_coverage,
        c.aligned_residues,
    )


def _average_candidates(candidates: list[CandidateTransfer], top_k: int) -> tuple[torch.Tensor, list[CandidateTransfer]]:
    if not candidates:
        raise ValueError("No template candidates available for averaging.")
    same_family_state = [c for c in candidates if c.same_family and c.same_state]
    same_family = [c for c in candidates if c.same_family]
    pool = same_family_state or same_family or candidates
    selected = sorted(pool, key=_candidate_sort_key, reverse=True)[: max(1, int(top_k))]
    stacked = torch.stack([c.prediction for c in selected], dim=0)
    nonzero = torch.linalg.vector_norm(stacked, dim=-1) > 0
    counts = nonzero.sum(dim=0).clamp(min=1).unsqueeze(-1)
    pred = stacked.sum(dim=0) / counts
    pred[nonzero.sum(dim=0) == 0] = 0.0
    return pred, selected


def _split_pair_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    splits = build_split_file_lists(
        args.data_dir,
        DEFAULT_SPLIT_RANGES,
        int(args.split_seed),
        split_source=args.split_source,
    )
    pair_dir = Path(args.pair_dir)
    target_paths = [pair_dir / f"{Path(path).stem}.pt" for path in splits.get(args.split, [])]
    donor_splits = args.donor_split or ["train", "val"]
    donor_stems: set[str] = set()
    for split in donor_splits:
        donor_stems.update(Path(path).stem for path in splits.get(split, []))
    donor_paths = [pair_dir / f"{stem}.pt" for stem in sorted(donor_stems)]
    missing = [str(path) for path in [*target_paths, *donor_paths] if not path.exists()]
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(f"Missing pair files for template baseline: {preview}")
    return target_paths, donor_paths


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_baselines(args: argparse.Namespace) -> dict[str, Any]:
    target_paths, donor_paths = _split_pair_paths(args)
    targets = [_load_pair(path) for path in target_paths]
    donors = [_load_pair(path) for path in donor_paths]
    output_root = Path(args.output_root)
    nearest_dir = output_root / "nearest_template"
    average_dir = output_root / "family_state_average"
    nearest_dir.mkdir(parents=True, exist_ok=True)
    average_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target in targets:
        candidates = [candidate for donor in donors if (candidate := _candidate(donor, target, args)) is not None]
        if not candidates:
            nearest_pred = torch.zeros_like(target.y_delta)
            average_pred = torch.zeros_like(target.y_delta)
            selected_nearest = None
            selected_average: list[CandidateTransfer] = []
        else:
            selected_nearest = sorted(candidates, key=_candidate_sort_key, reverse=True)[0]
            nearest_pred = selected_nearest.prediction
            average_pred, selected_average = _average_candidates(candidates, int(args.average_top_k))

        torch.save(
            {
                "pair_id": target.stem,
                "pred_delta": nearest_pred,
                "metadata": {
                    "baseline": "nearest_template_transfer",
                    "donor": selected_nearest.donor.stem if selected_nearest else "",
                    "candidate_count": len(candidates),
                },
            },
            nearest_dir / f"{target.stem}.pt",
        )
        torch.save(
            {
                "pair_id": target.stem,
                "pred_delta": average_pred,
                "metadata": {
                    "baseline": "family_state_average_template_transfer",
                    "donors": [candidate.donor.stem for candidate in selected_average],
                    "candidate_count": len(candidates),
                },
            },
            average_dir / f"{target.stem}.pt",
        )
        rows.append(
            {
                "target": target.stem,
                "target_uniprot": _uniprot(target),
                "target_family": _field(target, "family"),
                "target_state": _field(target, "state_label") or _field(target, "state"),
                "candidate_count": len(candidates),
                "nearest_donor": selected_nearest.donor.stem if selected_nearest else "",
                "nearest_uniprot": _uniprot(selected_nearest.donor) if selected_nearest else "",
                "nearest_family": _field(selected_nearest.donor, "family") if selected_nearest else "",
                "nearest_state": (_field(selected_nearest.donor, "state_label") or _field(selected_nearest.donor, "state"))
                if selected_nearest
                else "",
                "nearest_identity": selected_nearest.identity if selected_nearest else "",
                "nearest_target_coverage": selected_nearest.target_coverage if selected_nearest else "",
                "nearest_fit_region": selected_nearest.fit_region if selected_nearest else "",
                "nearest_fit_residues": selected_nearest.fit_residues if selected_nearest else "",
                "average_donor_count": len(selected_average),
                "average_donors": ";".join(candidate.donor.stem for candidate in selected_average),
            }
        )

    csv_path = output_root / "template_baseline_selection.csv"
    _write_csv(
        csv_path,
        rows,
        [
            "target",
            "target_uniprot",
            "target_family",
            "target_state",
            "candidate_count",
            "nearest_donor",
            "nearest_uniprot",
            "nearest_family",
            "nearest_state",
            "nearest_identity",
            "nearest_target_coverage",
            "nearest_fit_region",
            "nearest_fit_residues",
            "average_donor_count",
            "average_donors",
        ],
    )

    report = {
        "data_dir": args.data_dir,
        "pair_dir": args.pair_dir,
        "split": args.split,
        "donor_splits": args.donor_split or ["train", "val"],
        "split_source": args.split_source,
        "min_identity": float(args.min_identity),
        "min_target_coverage": float(args.min_target_coverage),
        "average_top_k": int(args.average_top_k),
        "allow_same_uniprot": bool(args.allow_same_uniprot),
        "target_count": len(targets),
        "donor_count": len(donors),
        "nearest_template_dir": str(nearest_dir),
        "family_state_average_dir": str(average_dir),
        "selection_csv": str(csv_path),
        "selection": rows,
    }
    report_path = Path(args.report_path) if args.report_path else output_root / "template_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    report = build_baselines(parse_args())
    print(json.dumps(_json_safe({k: report[k] for k in ("report_path", "target_count", "donor_count")}), indent=2))


if __name__ == "__main__":
    main()
