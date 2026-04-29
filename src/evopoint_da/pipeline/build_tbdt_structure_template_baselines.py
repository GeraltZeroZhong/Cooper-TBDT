"""Build external structure-search template-transfer baselines for TBDT.

These baselines use external structure search/alignment tools only to choose a
nearest train/val donor template. The coordinate prediction itself stays the
same displacement-transfer endpoint used by the internal nearest-template
baseline: align donor/target AF2 scaffolds, rotate donor displacement vectors
into the target frame, and copy vectors through the sequence alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evopoint_da.pipeline.build_tbdt_template_baselines import (
    DEFAULT_SPLIT_RANGES,
    PairSample,
    _field,
    _load_pair,
    _split_pair_paths,
    _transfer_prediction,
    _uniprot,
)
from evopoint_da.pipeline.eval_tbdt_state import evaluate as evaluate_regions


ALL_BASELINES = ("foldseek_nearest_template", "usalign_nearest_template")
DEFAULT_TOOL_ROOT = Path("artifacts/tbdt_v1/external_tools")
EPS = 1e-12


@dataclass(frozen=True)
class RankedHit:
    donor_stem: str
    score: float
    rank: int
    metadata: dict[str, Any]
    query_alignment: str = ""
    target_alignment: str = ""


@dataclass(frozen=True)
class SelectedTransfer:
    donor: PairSample | None
    prediction: torch.Tensor
    hit: RankedHit | None
    transfer: dict[str, Any] | None
    candidate_count: int
    rejected_count: int
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Foldseek/US-align nearest-template coordinate baselines.")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--pair-dir", default="data/processed_tbdt_gold_pairs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--donor-split", action="append", default=[], choices=["train", "val", "test", "all"])
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--baseline", action="append", default=[], choices=ALL_BASELINES)
    parser.add_argument("--allow-same-uniprot", action="store_true")
    parser.add_argument("--min-identity", type=float, default=0.0)
    parser.add_argument("--min-target-coverage", type=float, default=0.30)
    parser.add_argument("--output-root", default="artifacts/tbdt_v1/template_baselines")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--foldseek-bin", default=None)
    parser.add_argument("--foldseek-max-seqs", type=int, default=10000)
    parser.add_argument("--usalign-bin", default=None)
    parser.add_argument("--usalign-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tool-timeout", type=float, default=600.0)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--include-all-region", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-derived-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direction-threshold", type=float, default=1.0)
    parser.add_argument("--plug-apical-fraction", type=float, default=0.35)
    parser.add_argument("--plug-extension-residues", type=int, default=12)
    parser.add_argument("--bootstrap-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--tonb-exposure-threshold", type=float, default=1.0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
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
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resolve_executable(explicit: str | None, names: tuple[str, ...], candidates: tuple[Path, ...]) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path.resolve()
        found = shutil.which(explicit)
        if found:
            return Path(found).resolve()
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _foldseek_bin(args: argparse.Namespace) -> Path | None:
    return _resolve_executable(
        args.foldseek_bin,
        ("foldseek",),
        (DEFAULT_TOOL_ROOT / "foldseek/bin/foldseek",),
    )


def _usalign_bin(args: argparse.Namespace) -> Path | None:
    return _resolve_executable(
        args.usalign_bin,
        ("USalign", "TMalign"),
        (DEFAULT_TOOL_ROOT / "USalign/USalign",),
    )


def _row(sample: PairSample) -> dict[str, Any]:
    row = sample.metadata.get("manifest_row", {})
    return row if isinstance(row, dict) else {}


def _structure_path(sample: PairSample, key: str = "af2_pdb") -> Path:
    candidates: list[Any] = [sample.metadata.get(key), _row(sample).get(key)]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw))
        possible = [path] if path.is_absolute() else [Path.cwd() / path, Path.cwd() / "data" / path, path]
        for item in possible:
            if item.exists():
                return item.resolve()
    raise FileNotFoundError(f"{sample.stem} has no readable {key} path in metadata.")


def _safe_symlink_or_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copyfile(source, dest)


def _prepare_donor_directory(donors: list[PairSample], work_dir: Path) -> tuple[Path, Path]:
    donor_dir = work_dir / "donor_pdbs"
    donor_dir.mkdir(parents=True, exist_ok=True)
    list_path = work_dir / "donor_list.txt"
    names: list[str] = []
    for donor in donors:
        source = _structure_path(donor)
        _safe_symlink_or_copy(source, donor_dir / f"{donor.stem}.pdb")
        names.append(donor.stem)
    list_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return donor_dir, list_path


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _hit_stem(value: str) -> str:
    head = value.split(":", 1)[0]
    return Path(head).stem


def _parse_foldseek_hits(path: Path) -> list[RankedHit]:
    hits: list[RankedHit] = []
    if not path.exists():
        return hits
    with path.open("r", encoding="utf-8") as handle:
        for rank, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            alntmscore = _finite_float(parts[2])
            qtmscore = _finite_float(parts[3])
            ttmscore = _finite_float(parts[4])
            bits = _finite_float(parts[5])
            evalue = _finite_float(parts[6])
            score = next(
                (value for value in (alntmscore, qtmscore, ttmscore, bits) if math.isfinite(value)),
                float("-inf"),
            )
            hits.append(
                RankedHit(
                    donor_stem=_hit_stem(parts[1]),
                    score=float(score),
                    rank=rank,
                    metadata={
                        "external_query": parts[0],
                        "external_target": parts[1],
                        "foldseek_alntmscore": alntmscore,
                        "foldseek_qtmscore": qtmscore,
                        "foldseek_ttmscore": ttmscore,
                        "foldseek_bits": bits,
                        "foldseek_evalue": evalue,
                    },
                    query_alignment=parts[7] if len(parts) > 7 else "",
                    target_alignment=parts[8] if len(parts) > 8 else "",
                )
            )
    return sorted(hits, key=lambda hit: (hit.score, -hit.rank), reverse=True)


def _run_foldseek_hits(
    target: PairSample,
    donors: list[PairSample],
    *,
    args: argparse.Namespace,
    work_dir: Path,
    donor_dir: Path,
    foldseek_bin: Path,
) -> list[RankedHit]:
    query = _structure_path(target)
    target_work = work_dir / "queries" / target.stem
    target_work.mkdir(parents=True, exist_ok=True)
    hits_path = target_work / "hits.tsv"
    tmp_dir = target_work / "tmp"
    cmd = [
        str(foldseek_bin),
        "easy-search",
        str(query),
        str(donor_dir),
        str(hits_path),
        str(tmp_dir),
        "--format-output",
        "query,target,alntmscore,qtmscore,ttmscore,bits,evalue,qaln,taln",
        "--threads",
        str(max(1, int(args.threads))),
        "--max-seqs",
        str(max(len(donors), int(args.foldseek_max_seqs))),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(args.tool_timeout),
    )
    return _parse_foldseek_hits(hits_path)


def _parse_usalign_tabular(text: str) -> list[RankedHit]:
    hits: list[RankedHit] = []
    rank = 0
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        rank += 1
        tm1 = _finite_float(parts[2])
        tm2 = _finite_float(parts[3])
        rmsd = _finite_float(parts[4])
        id1 = _finite_float(parts[5])
        id2 = _finite_float(parts[6])
        idali = _finite_float(parts[7])
        l1 = _finite_float(parts[8])
        l2 = _finite_float(parts[9])
        lali = _finite_float(parts[10])
        score = tm1 if math.isfinite(tm1) else tm2
        hits.append(
            RankedHit(
                donor_stem=_hit_stem(parts[1]),
                score=float(score),
                rank=rank,
                metadata={
                    "external_query": parts[0],
                    "external_target": parts[1],
                    "usalign_tm_target_norm": tm1,
                    "usalign_tm_donor_norm": tm2,
                    "usalign_rmsd": rmsd,
                    "usalign_identity_target_norm": id1,
                    "usalign_identity_donor_norm": id2,
                    "usalign_identity_aligned": idali,
                    "usalign_target_length": l1,
                    "usalign_donor_length": l2,
                    "usalign_aligned_length": lali,
                },
            )
        )
    return sorted(hits, key=lambda hit: (hit.score, -hit.rank), reverse=True)


def _run_usalign_hits(
    target: PairSample,
    *,
    args: argparse.Namespace,
    donor_dir: Path,
    donor_list: Path,
    usalign_bin: Path,
) -> list[RankedHit]:
    query = _structure_path(target)
    cmd = [
        str(usalign_bin),
        str(query),
        "-dir2",
        str(donor_dir) + "/",
        str(donor_list),
        "-suffix",
        ".pdb",
        "-outfmt",
        "2",
        "-ter",
        "0",
    ]
    if bool(args.usalign_fast):
        cmd.append("-fast")
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(args.tool_timeout),
    )
    return _parse_usalign_tabular(result.stdout)


def _valid_transfer(
    donor: PairSample,
    target: PairSample,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.allow_same_uniprot and _uniprot(donor) and _uniprot(donor) == _uniprot(target):
        return None
    transfer = _transfer_prediction(donor, target)
    if transfer is None:
        return None
    identity = float(transfer["identity"])
    target_coverage = float(transfer["target_coverage"])
    if identity + EPS < float(args.min_identity) or target_coverage + EPS < float(args.min_target_coverage):
        return None
    return transfer


def _af2_index_lookup(sample: PairSample) -> dict[int, int]:
    raw = torch.load(sample.path, map_location="cpu", weights_only=False)
    value = raw.get("af2_indices") if isinstance(raw, dict) else None
    if value is None:
        return {idx: idx for idx in range(len(sample.sequence))}
    indices = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if indices.numel() != len(sample.sequence):
        return {idx: idx for idx in range(len(sample.sequence))}
    return {int(full_idx): int(local_idx) for local_idx, full_idx in enumerate(indices.tolist())}


def _aligned_full_index_pairs(
    donor_alignment: str,
    target_alignment: str,
    donor: PairSample,
    target: PairSample,
) -> list[tuple[int, int]]:
    donor_lookup = _af2_index_lookup(donor)
    target_lookup = _af2_index_lookup(target)
    donor_full = -1
    target_full = -1
    pairs: list[tuple[int, int]] = []
    for donor_char, target_char in zip(donor_alignment, target_alignment, strict=False):
        donor_has_residue = donor_char != "-"
        target_has_residue = target_char != "-"
        if donor_has_residue:
            donor_full += 1
        if target_has_residue:
            target_full += 1
        if not donor_has_residue or not target_has_residue:
            continue
        donor_idx = donor_lookup.get(donor_full)
        target_idx = target_lookup.get(target_full)
        if donor_idx is not None and target_idx is not None:
            pairs.append((donor_idx, target_idx))
    return pairs


def _transfer_from_pairs(donor: PairSample, target: PairSample, pairs: list[tuple[int, int]]) -> dict[str, Any] | None:
    if len(pairs) < 3:
        return None
    unique_pairs: dict[int, int] = {}
    for donor_i, target_i in pairs:
        unique_pairs.setdefault(int(target_i), int(donor_i))
    if len(unique_pairs) < 3:
        return None

    target_idx = torch.tensor(sorted(unique_pairs), dtype=torch.long)
    donor_idx = torch.tensor([unique_pairs[int(target_i)] for target_i in target_idx.tolist()], dtype=torch.long)
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

    from evopoint_da.data.alignment import kabsch_rotation

    rotation, _translation = kabsch_rotation(
        donor.af2_pos[fit_donor].numpy(),
        target.af2_pos[fit_target].numpy(),
    )
    rotated = donor.y_delta.float() @ torch.as_tensor(rotation, dtype=torch.float32).T
    pred = torch.zeros_like(target.y_delta)
    pred[target_idx] = rotated[donor_idx]
    matches = sum(
        1
        for donor_i, target_i in zip(donor_idx.tolist(), target_idx.tolist(), strict=False)
        if donor.sequence[donor_i] == target.sequence[target_i]
    )
    return {
        "prediction": pred,
        "identity": float(matches / max(1, int(donor_idx.numel()))),
        "target_coverage": float(torch.unique(target_idx).numel() / max(1, len(target.sequence))),
        "aligned_residues": int(donor_idx.numel()),
        "n_predicted_residues": int(torch.unique(target_idx).numel()),
        "fit_region": fit_region,
        "fit_residues": int(fit_donor.numel()),
        "mapping_source": "external_structure_alignment",
    }


def _parse_usalign_alignment(text: str) -> tuple[str, str] | None:
    lines = [line.strip() for line in text.splitlines()]
    seq_re = re.compile(r"^[A-Za-z\-]+$")
    mid_re = re.compile(r"^[\s:.]+$")
    best: tuple[str, str] | None = None
    for idx in range(len(lines) - 2):
        if seq_re.match(lines[idx]) and mid_re.match(lines[idx + 1]) and seq_re.match(lines[idx + 2]):
            if len(lines[idx]) == len(lines[idx + 2]):
                best = (lines[idx], lines[idx + 2])
    return best


def _usalign_pair_alignment(
    donor: PairSample,
    target: PairSample,
    *,
    args: argparse.Namespace,
    usalign_bin: Path,
) -> tuple[str, str] | None:
    cmd = [str(usalign_bin), str(_structure_path(donor)), str(_structure_path(target)), "-ter", "0"]
    if bool(args.usalign_fast):
        cmd.append("-fast")
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(args.tool_timeout),
    )
    return _parse_usalign_alignment(result.stdout)


def _external_transfer(
    baseline: str,
    donor: PairSample,
    target: PairSample,
    hit: RankedHit,
    args: argparse.Namespace,
    *,
    usalign_bin: Path | None,
) -> dict[str, Any] | None:
    if not args.allow_same_uniprot and _uniprot(donor) and _uniprot(donor) == _uniprot(target):
        return None

    transfer: dict[str, Any] | None = None
    if baseline == "foldseek_nearest_template" and hit.query_alignment and hit.target_alignment:
        pairs = _aligned_full_index_pairs(hit.target_alignment, hit.query_alignment, donor, target)
        transfer = _transfer_from_pairs(donor, target, pairs)
    elif baseline == "usalign_nearest_template" and usalign_bin is not None:
        alignment = _usalign_pair_alignment(donor, target, args=args, usalign_bin=usalign_bin)
        if alignment is not None:
            donor_alignment, target_alignment = alignment
            pairs = _aligned_full_index_pairs(donor_alignment, target_alignment, donor, target)
            transfer = _transfer_from_pairs(donor, target, pairs)

    if transfer is None:
        transfer = _valid_transfer(donor, target, args)
    if transfer is None:
        return None
    identity = float(transfer["identity"])
    target_coverage = float(transfer["target_coverage"])
    if identity + EPS < float(args.min_identity) or target_coverage + EPS < float(args.min_target_coverage):
        return None
    return transfer


def _select_transfer(
    baseline: str,
    target: PairSample,
    donors_by_stem: dict[str, PairSample],
    hits: list[RankedHit],
    args: argparse.Namespace,
    *,
    usalign_bin: Path | None = None,
) -> SelectedTransfer:
    rejected = 0
    seen: set[str] = set()
    for hit in hits:
        if hit.donor_stem in seen:
            continue
        seen.add(hit.donor_stem)
        donor = donors_by_stem.get(hit.donor_stem)
        if donor is None:
            rejected += 1
            continue
        transfer = _external_transfer(baseline, donor, target, hit, args, usalign_bin=usalign_bin)
        if transfer is None:
            rejected += 1
            continue
        return SelectedTransfer(
            donor=donor,
            prediction=transfer["prediction"],
            hit=hit,
            transfer=transfer,
            candidate_count=len(hits),
            rejected_count=rejected,
            status="ok",
        )
    return SelectedTransfer(
        donor=None,
        prediction=torch.zeros_like(target.y_delta),
        hit=None,
        transfer=None,
        candidate_count=len(hits),
        rejected_count=rejected,
        status="no_valid_transfer",
    )


def _save_prediction(output_dir: Path, target: PairSample, selected: SelectedTransfer, *, baseline: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{target.stem}.pt"
    metadata: dict[str, Any] = {
        "baseline": baseline,
        "donor": selected.donor.stem if selected.donor else "",
        "candidate_count": selected.candidate_count,
        "rejected_count": selected.rejected_count,
        "status": selected.status,
    }
    if selected.hit:
        metadata.update(selected.hit.metadata)
        metadata["external_score"] = selected.hit.score
        metadata["external_rank"] = selected.hit.rank
    if selected.transfer:
        metadata.update(
            {
                "identity": selected.transfer["identity"],
                "target_coverage": selected.transfer["target_coverage"],
                "aligned_residues": selected.transfer["aligned_residues"],
                "fit_region": selected.transfer["fit_region"],
                "fit_residues": selected.transfer["fit_residues"],
                "n_predicted_residues": selected.transfer["n_predicted_residues"],
                "mapping_source": selected.transfer.get("mapping_source", "sequence_alignment"),
            }
        )
    if selected.error:
        metadata["error"] = selected.error
    torch.save({"pair_id": target.stem, "pred_delta": selected.prediction, "metadata": metadata}, path)
    return path


def _selection_row(baseline: str, target: PairSample, selected: SelectedTransfer, path: Path) -> dict[str, Any]:
    row = {
        "baseline": baseline,
        "target": target.stem,
        "target_uniprot": _uniprot(target),
        "target_family": _field(target, "family"),
        "target_state": _field(target, "state_label") or _field(target, "state"),
        "prediction_file": str(path),
        "status": selected.status,
        "candidate_count": selected.candidate_count,
        "rejected_count": selected.rejected_count,
        "selected_donor": selected.donor.stem if selected.donor else "",
        "selected_uniprot": _uniprot(selected.donor) if selected.donor else "",
        "selected_family": _field(selected.donor, "family") if selected.donor else "",
        "selected_state": (_field(selected.donor, "state_label") or _field(selected.donor, "state"))
        if selected.donor
        else "",
        "external_score": selected.hit.score if selected.hit else "",
        "external_rank": selected.hit.rank if selected.hit else "",
        "error": selected.error,
    }
    if selected.hit:
        row.update(selected.hit.metadata)
    if selected.transfer:
        row.update(
            {
                "transfer_identity": selected.transfer["identity"],
                "transfer_target_coverage": selected.transfer["target_coverage"],
                "transfer_aligned_residues": selected.transfer["aligned_residues"],
                "transfer_fit_region": selected.transfer["fit_region"],
                "transfer_fit_residues": selected.transfer["fit_residues"],
                "transfer_n_predicted_residues": selected.transfer["n_predicted_residues"],
                "transfer_mapping_source": selected.transfer.get("mapping_source", "sequence_alignment"),
            }
        )
    return row


def _evaluate_prediction_dir(args: argparse.Namespace, target_paths: list[Path], prediction_dir: Path, output_root: Path) -> dict[str, Any]:
    output_json = output_root / f"{prediction_dir.name}_region_metrics.json"
    output_csv = output_root / f"{prediction_dir.name}_region_metrics.csv"
    eval_args = argparse.Namespace(
        inputs=[str(path) for path in target_paths],
        predictions=str(prediction_dir),
        output_json=str(output_json),
        output_csv=str(output_csv),
        region_json=None,
        include_all_region=bool(args.include_all_region),
        direction_threshold=float(args.direction_threshold),
        add_derived_regions=bool(args.add_derived_regions),
        plug_apical_fraction=float(args.plug_apical_fraction),
        plug_extension_residues=int(args.plug_extension_residues),
        bootstrap_iter=int(args.bootstrap_iter),
        bootstrap_seed=int(args.bootstrap_seed),
        paired_delta_csv=None,
        tonb_metrics_csv=None,
        tonb_exposure_threshold=float(args.tonb_exposure_threshold),
    )
    report = evaluate_regions(eval_args)
    report["output_json"] = str(output_json)
    report["output_csv"] = str(output_csv)
    return report


def build_baselines(args: argparse.Namespace) -> dict[str, Any]:
    target_paths, donor_paths = _split_pair_paths(args)
    targets = [_load_pair(path) for path in target_paths]
    donors = [_load_pair(path) for path in donor_paths]
    donors_by_stem = {donor.stem: donor for donor in donors}
    selected_baselines = tuple(args.baseline or ALL_BASELINES)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else output_root / "_external_template_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    donor_dir, donor_list = _prepare_donor_directory(donors, work_dir)

    foldseek_bin = _foldseek_bin(args)
    usalign_bin = _usalign_bin(args)
    skipped: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    prediction_dirs: dict[str, Path] = {}
    metrics: dict[str, Any] = {}
    hits_cache: dict[tuple[str, str], list[RankedHit]] = {}

    for baseline in selected_baselines:
        if baseline == "foldseek_nearest_template" and foldseek_bin is None:
            skipped.append({"baseline": baseline, "reason": "Foldseek executable not found."})
            continue
        if baseline == "usalign_nearest_template" and usalign_bin is None:
            skipped.append({"baseline": baseline, "reason": "USalign/TMalign executable not found."})
            continue

        baseline_dir = output_root / baseline
        prediction_dirs[baseline] = baseline_dir
        for target in targets:
            try:
                cache_key = (baseline, str(_structure_path(target)))
                if baseline == "foldseek_nearest_template":
                    if cache_key not in hits_cache:
                        hits_cache[cache_key] = _run_foldseek_hits(
                            target,
                            donors,
                            args=args,
                            work_dir=work_dir / "foldseek",
                            donor_dir=donor_dir,
                            foldseek_bin=foldseek_bin,
                        )
                    hits = hits_cache[cache_key]
                elif baseline == "usalign_nearest_template":
                    if cache_key not in hits_cache:
                        hits_cache[cache_key] = _run_usalign_hits(
                            target,
                            args=args,
                            donor_dir=donor_dir,
                            donor_list=donor_list,
                            usalign_bin=usalign_bin,
                        )
                    hits = hits_cache[cache_key]
                else:
                    raise ValueError(f"Unknown baseline: {baseline}")
                selected = _select_transfer(
                    baseline,
                    target,
                    donors_by_stem,
                    hits,
                    args,
                    usalign_bin=usalign_bin,
                )
            except Exception as exc:
                selected = SelectedTransfer(
                    donor=None,
                    prediction=torch.zeros_like(target.y_delta),
                    hit=None,
                    transfer=None,
                    candidate_count=0,
                    rejected_count=0,
                    status="failed_zero_fallback",
                    error=str(exc),
                )
            prediction_path = _save_prediction(baseline_dir, target, selected, baseline=baseline)
            rows.append(_selection_row(baseline, target, selected, prediction_path))

        if not bool(args.skip_eval):
            metrics[baseline] = _evaluate_prediction_dir(args, target_paths, baseline_dir, output_root)

    selection_csv = output_root / "external_template_baseline_selection.csv"
    if rows:
        _write_csv(selection_csv, rows)

    report = {
        "data_dir": args.data_dir,
        "pair_dir": args.pair_dir,
        "split": args.split,
        "donor_splits": args.donor_split or ["train", "val"],
        "split_source": args.split_source,
        "target_count": len(targets),
        "donor_count": len(donors),
        "requested_baselines": list(selected_baselines),
        "prediction_dirs": {name: str(path) for name, path in sorted(prediction_dirs.items())},
        "selection_csv": str(selection_csv) if rows else "",
        "skipped": skipped,
        "tool_paths": {
            "foldseek": str(foldseek_bin) if foldseek_bin else "",
            "usalign": str(usalign_bin) if usalign_bin else "",
        },
        "tool_settings": {
            "min_identity": float(args.min_identity),
            "min_target_coverage": float(args.min_target_coverage),
            "allow_same_uniprot": bool(args.allow_same_uniprot),
            "usalign_fast": bool(args.usalign_fast),
            "threads": int(args.threads),
        },
        "metrics": metrics,
        "scientific_contract": {
            "selector_only": (
                "Foldseek and US-align choose the donor template from train/val AF2 structures. "
                "The per-residue coordinate endpoint remains displacement transfer from the selected donor."
            ),
            "residue_mapping": (
                "Foldseek uses its qaln/taln alignment strings. US-align uses a pairwise alignment rerun for the "
                "selected donor. Both mappings are converted through processed AF2 residue indices before vector transfer."
            ),
            "leakage_control": "Default excludes donors with the same UniProt accession as the target.",
        },
    }
    report_path = Path(args.report_path) if args.report_path else output_root / "external_template_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    report = build_baselines(parse_args())
    print(
        json.dumps(
            _json_safe(
                {
                    "report_path": report["report_path"],
                    "target_count": report["target_count"],
                    "donor_count": report["donor_count"],
                    "prediction_dirs": report["prediction_dirs"],
                    "skipped": report["skipped"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
