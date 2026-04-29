"""Build external score-only TBDT baselines.

The baselines in this module answer residue-localization questions rather than
the coordinate RMSD endpoint: they produce one scalar score per residue and are
evaluated by the existing ROC/PR displacement-localization task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio import Align
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from evopoint_da.data.dataset import build_split_file_lists
from evopoint_da.pipeline.eval_tbdt_classification_curves import evaluate as evaluate_classification
from evopoint_da.pipeline.tbdt_state_eval_metrics import _json_safe
from evopoint_da.pipeline.tbdt_state_eval_utils import (
    _as_mapping,
    _extract_target,
    _first_present,
    _load_pt,
    _vector_tensor,
)


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}
ALL_BASELINES = (
    "modeller_nearest_holo_template",
    "prody_anm_mobility",
    "prody_gnm_mobility",
    "dynamine_sequence_dynamics",
    "iupred2a_long",
    "p2rank_pocket_score",
    "fpocket_pocket_score",
    "protcross_pocket_score",
)
FEASIBLE_BASELINES = {
    "prody_anm_mobility",
    "prody_gnm_mobility",
    "iupred2a_long",
    "p2rank_pocket_score",
    "fpocket_pocket_score",
    "protcross_pocket_score",
}
INFEASIBLE_REASONS = {
    "modeller_nearest_holo_template": (
        "Skipped: MODELLER is not installed and requires a MODELLER license key. "
        "The repository already has a license-free nearest-template transfer baseline, but it is not a MODELLER run."
    ),
    "dynamine_sequence_dynamics": (
        "Skipped: the Bio2Byte DynaMine API requires an authenticated token and was not available as a local package. "
        "Using an amino-acid propensity proxy would not be a DynaMine baseline."
    ),
}
DEFAULT_REGIONS = ("eval", "plug", "tonb_box", "barrel_core", "all")
DEFAULT_BUILTIN_BASELINES = ("af2_low_plddt", "af2_surface_rsa")
DEFAULT_PREDICTIONS = (
    (
        "cooper_tbdt_scaffold_blend",
        Path("artifacts/tbdt_v1/report_models/predictions/validation_calibrated_region_blend_test"),
    ),
)
IUPRED_URL_TEMPLATE = "https://iupred2a.elte.hu/iupred2a/{kind}/{accession}.json"
DEFAULT_TOOL_ROOT = Path("artifacts/tbdt_v1/external_tools")
DEFAULT_PROTCROSS_ROOT = Path("/home/zero/ProtCross")
EPS = 1e-12


@dataclass(frozen=True)
class Sample:
    path: Path
    stem: str
    raw: dict[str, Any]
    pos: torch.Tensor
    target: torch.Tensor
    sequence: str
    uniprot_id: str
    residue_ids: list[str]
    af2_indices: torch.Tensor
    af2_pdb: Path | None
    af2_chain: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external score-only baselines for TBDT residue shift localization.")
    parser.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    parser.add_argument("--pair-dir", default="data/processed_tbdt_gold_pairs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--baseline", action="append", default=[], choices=ALL_BASELINES)
    parser.add_argument("--output-root", default="artifacts/tbdt_v1/external_score_baselines")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--classification-out-dir", default="artifacts/tbdt_v1/external_baseline_curves")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--anm-cutoff", type=float, default=15.0)
    parser.add_argument("--gnm-cutoff", type=float, default=10.0)
    parser.add_argument("--mode-count", type=int, default=10)
    parser.add_argument("--iupred-kind", default="long", choices=["long", "short", "glob"])
    parser.add_argument("--iupred-cache-dir", default=None)
    parser.add_argument("--iupred-timeout", type=float, default=30.0)
    parser.add_argument("--iupred-sleep", type=float, default=0.25)
    parser.add_argument("--tool-work-dir", default=None)
    parser.add_argument("--tool-timeout", type=float, default=900.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--p2rank-bin", default=None)
    parser.add_argument("--p2rank-config", default="alphafold")
    parser.add_argument("--fpocket-bin", default=None)
    parser.add_argument(
        "--allow-failed-zero-fallback",
        action="store_true",
        help="Write zero-valued per-residue scores for per-sample external-tool failures instead of failing the run.",
    )
    parser.add_argument("--protcross-root", default=str(DEFAULT_PROTCROSS_ROOT))
    parser.add_argument("--protcross-checkpoint", default=None)
    parser.add_argument("--protcross-esm-weights", default="esmc_weights/esmc_600m_2024_12_v0.pth")
    parser.add_argument("--protcross-pca", default=None)
    parser.add_argument("--protcross-device", default="auto")
    parser.add_argument("--prediction", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--external-baseline", action="append", default=[])
    return parser.parse_args()


def _split_paths(args: argparse.Namespace) -> list[Path]:
    splits = build_split_file_lists(
        args.data_dir,
        DEFAULT_SPLIT_RANGES,
        int(args.split_seed),
        split_source=args.split_source,
    )
    paths = [Path(path) for path in splits.get(args.split, [])]
    if not paths:
        raise FileNotFoundError(f"No samples found for split={args.split!r} in {args.data_dir}")
    return paths


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    row = _metadata(sample).get("manifest_row", {})
    return row if isinstance(row, dict) else {}


def _uniprot_id(sample: dict[str, Any]) -> str:
    row = _manifest_row(sample)
    metadata = _metadata(sample)
    for source in (row, metadata, sample):
        value = source.get("uniprot_id") if isinstance(source, dict) else None
        if value:
            return str(value).strip().upper()
    return ""


def _pair_sequence(pair_dir: Path, stem: str) -> str:
    pair_path = pair_dir / f"{stem}.pt"
    if not pair_path.exists():
        return ""
    pair = _as_mapping(_load_pt(pair_path))
    return str(pair.get("sequence") or "")


def _load_pair_mapping(pair_dir: Path, stem: str) -> dict[str, Any]:
    pair_path = pair_dir / f"{stem}.pt"
    if not pair_path.exists():
        return {}
    return _as_mapping(_load_pt(pair_path))


def _first_metadata_value(raw: dict[str, Any], pair: dict[str, Any], key: str) -> str:
    sources: list[Any] = [
        _metadata(raw),
        _manifest_row(raw),
        _metadata(pair),
        _manifest_row(pair),
        raw,
        pair,
    ]
    for source in sources:
        if isinstance(source, dict):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _resolve_path_value(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path.cwd() / "data" / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _residue_ids(raw: dict[str, Any], pair: dict[str, Any], n: int) -> list[str]:
    value = raw.get("residue_ids") or pair.get("residue_ids")
    if isinstance(value, (list, tuple)) and len(value) == n:
        return [str(item) for item in value]
    return [f"A_{idx + 1}" for idx in range(n)]


def _af2_indices(pair: dict[str, Any], n: int) -> torch.Tensor:
    value = pair.get("af2_indices")
    if value is None:
        return torch.arange(n, dtype=torch.long)
    tensor = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if tensor.numel() != n:
        return torch.arange(n, dtype=torch.long)
    return tensor


def _load_sample(path: Path, pair_dir: Path) -> Sample:
    raw = _as_mapping(_load_pt(path))
    pair = _load_pair_mapping(pair_dir, path.stem)
    target = _extract_target(raw, path)
    pos_value = _first_present(raw, ("pos", "af2_pos"))
    if pos_value is None:
        raise ValueError(f"{path} has no pos/af2_pos coordinates.")
    pos = _vector_tensor(pos_value, f"pos in {path}")
    if pos.size(0) != target.size(0):
        raise ValueError(f"{path} coordinate length {pos.size(0)} != target length {target.size(0)}")
    sequence = str(raw.get("sequence") or pair.get("sequence") or _pair_sequence(pair_dir, path.stem))
    af2_pdb = _resolve_path_value(_first_metadata_value(raw, pair, "af2_pdb"))
    af2_chain = _first_metadata_value(raw, pair, "af2_chain") or _first_metadata_value(raw, pair, "chain")
    return Sample(
        path=path,
        stem=path.stem,
        raw=raw,
        pos=pos.float(),
        target=target.float(),
        sequence=sequence,
        uniprot_id=_uniprot_id(raw),
        residue_ids=_residue_ids(raw, pair, int(pos.size(0))),
        af2_indices=_af2_indices(pair, int(pos.size(0))),
        af2_pdb=af2_pdb,
        af2_chain=af2_chain.strip(),
    )


def _normalize_score(score: np.ndarray) -> torch.Tensor:
    arr = np.asarray(score, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    if np.isnan(arr).all():
        arr = np.zeros_like(arr)
    else:
        fill = float(np.nanmedian(arr))
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return torch.from_numpy(arr.astype(np.float32))


def _contact_pairs(pos: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(pos)
    pairs = np.asarray(list(tree.query_pairs(float(cutoff))), dtype=np.int64)
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    i = pairs[:, 0]
    j = pairs[:, 1]
    dist = np.linalg.norm(pos[j] - pos[i], axis=1)
    valid = dist > EPS
    return i[valid], j[valid], dist[valid]


def _low_modes(matrix: csr_matrix, *, skip_modes: int, mode_count: int) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if n <= skip_modes + 1:
        return np.empty(0, dtype=float), np.empty((n, 0), dtype=float)
    k = min(n - 2, max(1, int(skip_modes) + int(mode_count)))
    try:
        values, vectors = eigsh(matrix, k=k, which="SM", tol=1e-5, maxiter=max(300, n * 5))
    except Exception:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=float)
    valid = values > 1e-8
    values = values[valid]
    vectors = vectors[:, valid]
    return values[:mode_count], vectors[:, :mode_count]


def _gnm_mobility(pos: torch.Tensor, *, cutoff: float, mode_count: int) -> torch.Tensor:
    xyz = pos.detach().cpu().numpy().astype(float)
    n = xyz.shape[0]
    i, j, _dist = _contact_pairs(xyz, cutoff)
    if i.size == 0:
        return torch.zeros(n, dtype=torch.float32)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = -np.ones(rows.size, dtype=float)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    degree = -np.asarray(adjacency.sum(axis=1)).reshape(-1)
    kirchhoff = adjacency + diags(degree, offsets=0, format="csr")
    values, vectors = _low_modes(kirchhoff, skip_modes=1, mode_count=mode_count)
    if values.size == 0:
        return torch.zeros(n, dtype=torch.float32)
    fluct = np.sum((vectors * vectors) / values.reshape(1, -1), axis=1)
    return _normalize_score(fluct)


def _anm_mobility(pos: torch.Tensor, *, cutoff: float, mode_count: int) -> torch.Tensor:
    xyz = pos.detach().cpu().numpy().astype(float)
    n = xyz.shape[0]
    i_idx, j_idx, dist = _contact_pairs(xyz, cutoff)
    if i_idx.size == 0:
        return torch.zeros(n, dtype=torch.float32)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros((n, 3, 3), dtype=float)
    for i, j, d in zip(i_idx.tolist(), j_idx.tolist(), dist.tolist(), strict=False):
        unit = (xyz[j] - xyz[i]) / d
        block = -np.outer(unit, unit)
        diag[i] -= block
        diag[j] -= block
        for a in range(3):
            for b in range(3):
                rows.append(3 * i + a)
                cols.append(3 * j + b)
                data.append(block[a, b])
                rows.append(3 * j + a)
                cols.append(3 * i + b)
                data.append(block[a, b])
    for i in range(n):
        for a in range(3):
            for b in range(3):
                rows.append(3 * i + a)
                cols.append(3 * i + b)
                data.append(diag[i, a, b])
    hessian = coo_matrix((data, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
    values, vectors = _low_modes(hessian, skip_modes=6, mode_count=mode_count)
    if values.size == 0:
        return torch.zeros(n, dtype=torch.float32)
    mode_fluct = vectors.reshape(n, 3, -1)
    fluct = np.sum(np.sum(mode_fluct * mode_fluct, axis=1) / values.reshape(1, -1), axis=1)
    return _normalize_score(fluct)


def _fetch_iupred(accession: str, *, kind: str, cache_dir: Path, timeout: float, sleep: float) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{accession}_{kind}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    url = IUPRED_URL_TEMPLATE.format(kind=kind, accession=accession)
    request = urllib.request.Request(url, headers={"User-Agent": "Cooper-TBDT external baseline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"IUPred2A request failed for {accession}: {exc}") from exc
    data = json.loads(payload)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    if sleep > 0:
        time.sleep(float(sleep))
    return data


def _align_scores(source_sequence: str, source_scores: list[float], target_sequence: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if not target_sequence:
        raise ValueError("Cannot align sequence baseline without a target sample sequence.")
    source_sequence = "".join(source_sequence.split()).upper()
    target_sequence = "".join(target_sequence.split()).upper()
    if len(source_sequence) != len(source_scores):
        raise ValueError(f"Source sequence length {len(source_sequence)} != score length {len(source_scores)}")
    start = source_sequence.find(target_sequence)
    if start >= 0:
        scores = source_scores[start : start + len(target_sequence)]
        return torch.tensor(scores, dtype=torch.float32), {
            "mapping": "substring",
            "source_length": len(source_sequence),
            "target_length": len(target_sequence),
            "mapped_residues": len(target_sequence),
            "identity": 1.0,
        }

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -2.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = next(iter(aligner.align(source_sequence, target_sequence)))
    source_blocks, target_blocks = alignment.aligned
    out = np.full(len(target_sequence), np.nan, dtype=float)
    matches = 0
    mapped = 0
    for (s_start, s_end), (t_start, t_end) in zip(source_blocks, target_blocks, strict=False):
        length = min(s_end - s_start, t_end - t_start)
        for offset in range(length):
            s_idx = s_start + offset
            t_idx = t_start + offset
            out[t_idx] = float(source_scores[s_idx])
            mapped += 1
            matches += int(source_sequence[s_idx] == target_sequence[t_idx])
    fill = float(np.nanmedian(out)) if not np.isnan(out).all() else float(np.mean(source_scores))
    out = np.nan_to_num(out, nan=fill)
    return torch.from_numpy(out.astype(np.float32)), {
        "mapping": "pairwise_alignment",
        "source_length": len(source_sequence),
        "target_length": len(target_sequence),
        "mapped_residues": mapped,
        "identity": float(matches / mapped) if mapped else 0.0,
    }


def _iupred_score(
    sample: Sample,
    *,
    kind: str,
    cache_dir: Path,
    timeout: float,
    sleep: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not sample.uniprot_id:
        raise ValueError(f"{sample.stem} has no UniProt accession for IUPred2A.")
    data = _fetch_iupred(sample.uniprot_id, kind=kind, cache_dir=cache_dir, timeout=timeout, sleep=sleep)
    source_sequence = str(data.get("sequence") or "")
    source_scores = [float(value) for value in data.get("iupred2") or []]
    score, mapping = _align_scores(source_sequence, source_scores, sample.sequence)
    return score, {"uniprot_id": sample.uniprot_id, "iupred_kind": kind, **mapping}


def _save_score(output_dir: Path, sample: Sample, score: torch.Tensor, *, baseline: str, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample.stem}.pt"
    torch.save(
        {
            "pair_id": sample.stem,
            "score": score.detach().cpu().float().reshape(-1),
            "metadata": {"baseline": baseline, **metadata},
        },
        path,
    )
    return path


def _resolve_executable(explicit: str | None, names: tuple[str, ...], candidates: tuple[Path, ...]) -> Path | None:
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path.resolve()
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


def _p2rank_bin(args: argparse.Namespace) -> Path | None:
    return _resolve_executable(
        args.p2rank_bin,
        ("prank",),
        (DEFAULT_TOOL_ROOT / "p2rank_2.5.1/prank",),
    )


def _fpocket_bin(args: argparse.Namespace) -> Path | None:
    return _resolve_executable(
        args.fpocket_bin,
        ("fpocket",),
        (DEFAULT_TOOL_ROOT / "fpocket/bin/fpocket",),
    )


def _protcross_assets(args: argparse.Namespace) -> tuple[Path, Path, Path, Path] | None:
    root = Path(args.protcross_root)
    checkpoint = Path(args.protcross_checkpoint) if args.protcross_checkpoint else root / "checkpoints/protcross-0.1.2-binding-moad-final.ckpt"
    pca = Path(args.protcross_pca) if args.protcross_pca else root / "data/pca_esmc_128_binding_moad_0.1.2.pkl"
    esm_weights = Path(args.protcross_esm_weights)
    if not esm_weights.is_absolute():
        esm_weights = Path.cwd() / esm_weights
    missing = [path for path in (root, checkpoint, pca, esm_weights) if not path.exists()]
    if missing:
        return None
    return root.resolve(), checkpoint.resolve(), esm_weights.resolve(), pca.resolve()


def _residue_id_from_parts(chain: str, label: str) -> str:
    chain = (chain or "A").strip() or "A"
    label = str(label).strip()
    return f"{chain}_{label}"


def _pdb_residue_ids(path: Path, chain_id: str = "") -> list[str]:
    residues: list[str] = []
    seen: set[str] = set()
    chain_filter = chain_id.strip()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip() or "A"
            if chain_filter and chain != chain_filter:
                continue
            number = line[22:26].strip()
            insertion = line[26].strip()
            if not number:
                continue
            residue_id = _residue_id_from_parts(chain, f"{number}{insertion}")
            if residue_id not in seen:
                residues.append(residue_id)
                seen.add(residue_id)
    return residues


def _map_af2_score(sample: Sample, score_by_residue_id: dict[str, float]) -> tuple[torch.Tensor, dict[str, Any]]:
    if sample.af2_pdb is None:
        raise ValueError(f"{sample.stem} has no AF2 PDB path for structure-score mapping.")
    af2_ids = _pdb_residue_ids(sample.af2_pdb, sample.af2_chain)
    if not af2_ids:
        raise ValueError(f"No residues parsed from AF2 PDB for {sample.stem}: {sample.af2_pdb}")
    full = np.zeros(len(af2_ids), dtype=np.float32)
    mapped_af2 = 0
    for idx, residue_id in enumerate(af2_ids):
        if residue_id in score_by_residue_id:
            full[idx] = float(score_by_residue_id[residue_id])
            mapped_af2 += 1
    indices = sample.af2_indices.detach().cpu().long().numpy()
    valid = (indices >= 0) & (indices < len(full))
    out = np.zeros(sample.pos.size(0), dtype=np.float32)
    out[valid] = full[indices[valid]]
    return torch.from_numpy(out), {
        "af2_residues": len(af2_ids),
        "af2_scored_residues": int(mapped_af2),
        "mapped_sample_residues": int(valid.sum()),
        "af2_pdb": str(sample.af2_pdb),
        "af2_chain": sample.af2_chain,
    }


def _p2rank_score(sample: Sample, args: argparse.Namespace, p2rank_bin: Path, work_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if sample.af2_pdb is None:
        raise ValueError(f"{sample.stem} has no AF2 PDB path.")
    out_dir = work_dir / "p2rank" / sample.stem
    residues_csv = out_dir / f"{sample.af2_pdb.name}_residues.csv"
    if not residues_csv.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(p2rank_bin),
            "predict",
            "-f",
            str(sample.af2_pdb),
            "-c",
            str(args.p2rank_config),
            "-o",
            str(out_dir),
            "-threads",
            str(max(1, int(args.threads))),
            "-visualizations",
            "0",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=float(args.tool_timeout))
    score_by_id: dict[str, float] = {}
    with residues_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            residue_id = _residue_id_from_parts(str(row.get("chain") or ""), str(row.get("residue_label") or ""))
            score = row.get("probability") or row.get("score") or "0"
            score_by_id[residue_id] = max(score_by_id.get(residue_id, 0.0), float(score))
    score, mapping = _map_af2_score(sample, score_by_id)
    return score, {"method": "P2Rank ligand-binding residue probability", "tool_output": str(residues_csv), **mapping}


def _copy_structure_for_tool(sample: Sample, run_dir: Path) -> Path:
    if sample.af2_pdb is None:
        raise ValueError(f"{sample.stem} has no AF2 PDB path.")
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"{sample.stem}{sample.af2_pdb.suffix or '.pdb'}"
    if not dest.exists():
        shutil.copyfile(sample.af2_pdb, dest)
    return dest


def _parse_fpocket_info(info_path: Path) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = {}
    current: int | None = None
    pocket_re = re.compile(r"^Pocket\s+(\d+)\s*:")
    value_re = re.compile(r"^\s*(Score|Druggability Score)\s*:\s*([-+0-9.eE]+)")
    with info_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            pocket_match = pocket_re.match(line)
            if pocket_match:
                current = int(pocket_match.group(1))
                scores.setdefault(current, {})
                continue
            if current is None:
                continue
            value_match = value_re.match(line)
            if not value_match:
                continue
            key = "score" if value_match.group(1) == "Score" else "drug_score"
            scores.setdefault(current, {})[key] = float(value_match.group(2))
    return scores


def _parse_pdb_residue_set(path: Path) -> set[str]:
    residues: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip() or "A"
            number = line[22:26].strip()
            insertion = line[26].strip()
            if number:
                residues.add(_residue_id_from_parts(chain, f"{number}{insertion}"))
    return residues


def _fpocket_score(sample: Sample, args: argparse.Namespace, fpocket_bin: Path, work_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    run_dir = work_dir / "fpocket" / sample.stem
    input_pdb = _copy_structure_for_tool(sample, run_dir)
    out_dir = run_dir / f"{input_pdb.stem}_out"
    info_path = out_dir / f"{input_pdb.stem}_info.txt"
    if not info_path.exists():
        subprocess.run(
            [str(fpocket_bin), "-f", str(input_pdb.name)],
            cwd=run_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(args.tool_timeout),
        )
    pocket_scores = _parse_fpocket_info(info_path)
    score_by_id: dict[str, float] = {}
    for pocket_file in sorted((out_dir / "pockets").glob("pocket*_atm.pdb")):
        match = re.search(r"pocket(\d+)_atm\.pdb$", pocket_file.name)
        if not match:
            continue
        pocket_id = int(match.group(1))
        pocket_score = float(pocket_scores.get(pocket_id, {}).get("score", pocket_scores.get(pocket_id, {}).get("drug_score", 0.0)))
        for residue_id in _parse_pdb_residue_set(pocket_file):
            score_by_id[residue_id] = max(score_by_id.get(residue_id, 0.0), pocket_score)
    score, mapping = _map_af2_score(sample, score_by_id)
    return score, {
        "method": "fpocket max pocket score assigned to contacting pocket residues",
        "tool_output": str(info_path),
        "pocket_count": len(pocket_scores),
        **mapping,
    }


def _protcross_score(sample: Sample, args: argparse.Namespace, assets: tuple[Path, Path, Path, Path], work_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if sample.af2_pdb is None:
        raise ValueError(f"{sample.stem} has no AF2 PDB path.")
    root, checkpoint, esm_weights, pca = assets
    out_dir = (work_dir / "protcross" / sample.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    score_tsv = out_dir / "scores.tsv"
    output_pdb = out_dir / "protcross_bfactor.pdb"
    if not score_tsv.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [
            sys.executable,
            "-m",
            "evopoint_da.cli.predict",
            str(sample.af2_pdb),
            "--checkpoint",
            str(checkpoint),
            "--esm-weights",
            str(esm_weights),
            "--pca",
            str(pca),
            "--scores-tsv",
            str(score_tsv),
            "--output",
            str(output_pdb),
            "--device",
            str(args.protcross_device),
            "--quiet",
        ]
        if sample.af2_chain:
            cmd.extend(["--chain", sample.af2_chain])
        subprocess.run(cmd, cwd=root, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=float(args.tool_timeout))
    score_by_id: dict[str, float] = {}
    with score_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            residue_id = str(row.get("residue_id") or "").strip()
            if not residue_id:
                residue_id = _residue_id_from_parts(str(row.get("chain_id") or ""), str(row.get("residue_number") or ""))
            score_by_id[residue_id] = max(score_by_id.get(residue_id, 0.0), float(row.get("probability") or 0.0))
    score, mapping = _map_af2_score(sample, score_by_id)
    return score, {
        "method": "ProtCross binding-site probability",
        "tool_output": str(score_tsv),
        "protcross_root": str(root),
        "checkpoint": str(checkpoint),
        "pca": str(pca),
        **mapping,
    }


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
            writer.writerow({key: row.get(key, "") for key in columns})


def _parse_prediction_specs(specs: list[str]) -> list[str]:
    parsed = list(specs)
    if parsed:
        return parsed
    for name, path in DEFAULT_PREDICTIONS:
        if path.is_dir():
            parsed.append(f"{name}={path}")
    return parsed


def build_baselines(args: argparse.Namespace) -> dict[str, Any]:
    sample_paths = _split_paths(args)
    samples = [_load_sample(path, Path(args.pair_dir)) for path in sample_paths]
    selected = tuple(args.baseline or ALL_BASELINES)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    iupred_cache = Path(args.iupred_cache_dir) if args.iupred_cache_dir else output_root / "iupred_cache"
    tool_work_dir = Path(args.tool_work_dir) if args.tool_work_dir else output_root / "tool_runs"
    p2rank_bin = _p2rank_bin(args)
    fpocket_bin = _fpocket_bin(args)
    protcross_assets = _protcross_assets(args)

    score_dirs: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for baseline in selected:
        if baseline not in FEASIBLE_BASELINES:
            skipped.append({"baseline": baseline, "reason": INFEASIBLE_REASONS.get(baseline, "Not implemented.")})
            continue
        if baseline == "p2rank_pocket_score" and p2rank_bin is None:
            skipped.append({"baseline": baseline, "reason": "Skipped: P2Rank prank executable not found."})
            continue
        if baseline == "fpocket_pocket_score" and fpocket_bin is None:
            skipped.append({"baseline": baseline, "reason": "Skipped: fpocket executable not found."})
            continue
        if baseline == "protcross_pocket_score" and protcross_assets is None:
            skipped.append(
                {
                    "baseline": baseline,
                    "reason": "Skipped: ProtCross root/checkpoint/ESM-C weights/PCA assets were not all readable.",
                }
            )
            continue
        output_dir = output_root / baseline
        score_dirs[baseline] = output_dir
        for sample in samples:
            metadata: dict[str, Any]
            try:
                if baseline == "prody_gnm_mobility":
                    score = _gnm_mobility(sample.pos, cutoff=float(args.gnm_cutoff), mode_count=int(args.mode_count))
                    metadata = {
                        "method": "GNM square fluctuation from low-frequency modes",
                        "cutoff": float(args.gnm_cutoff),
                        "mode_count": int(args.mode_count),
                        "status": "ok",
                    }
                elif baseline == "prody_anm_mobility":
                    score = _anm_mobility(sample.pos, cutoff=float(args.anm_cutoff), mode_count=int(args.mode_count))
                    metadata = {
                        "method": "ANM square fluctuation from low-frequency modes",
                        "cutoff": float(args.anm_cutoff),
                        "mode_count": int(args.mode_count),
                        "status": "ok",
                    }
                elif baseline == "iupred2a_long":
                    score, metadata = _iupred_score(
                        sample,
                        kind=str(args.iupred_kind),
                        cache_dir=iupred_cache,
                        timeout=float(args.iupred_timeout),
                        sleep=float(args.iupred_sleep),
                    )
                    metadata = {**metadata, "status": "ok"}
                elif baseline == "p2rank_pocket_score":
                    score, metadata = _p2rank_score(sample, args, p2rank_bin, tool_work_dir)
                    metadata = {**metadata, "status": "ok"}
                elif baseline == "fpocket_pocket_score":
                    score, metadata = _fpocket_score(sample, args, fpocket_bin, tool_work_dir)
                    metadata = {**metadata, "status": "ok"}
                elif baseline == "protcross_pocket_score":
                    score, metadata = _protcross_score(sample, args, protcross_assets, tool_work_dir)
                    metadata = {**metadata, "status": "ok"}
                else:
                    raise ValueError(f"Unsupported feasible baseline: {baseline}")
            except Exception as exc:
                if not bool(args.allow_failed_zero_fallback):
                    raise RuntimeError(f"{baseline} failed for sample {sample.stem}: {exc}") from exc
                score = torch.zeros(sample.pos.size(0), dtype=torch.float32)
                metadata = {"method": baseline, "status": "failed_zero_fallback", "error": str(exc)}
            if score.numel() != sample.pos.size(0):
                raise ValueError(f"{baseline}/{sample.stem} score length {score.numel()} != residues {sample.pos.size(0)}")
            path = _save_score(output_dir, sample, score, baseline=baseline, metadata=metadata)
            rows.append(
                {
                    "baseline": baseline,
                    "sample_id": sample.stem,
                    "score_file": str(path),
                    "n_residues": int(score.numel()),
                    "score_mean": float(score.float().mean().item()) if score.numel() else float("nan"),
                    "score_median": float(score.float().median().item()) if score.numel() else float("nan"),
                    "status": metadata.get("status", ""),
                    "error": metadata.get("error", ""),
                    **metadata,
                }
            )

    if rows:
        _write_csv(output_root / "external_score_baseline_samples.csv", rows)

    classification_report = None
    if score_dirs and not bool(args.skip_classification):
        eval_args = argparse.Namespace(
            inputs=[str(path) for path in sample_paths],
            sample_list=None,
            prediction=_parse_prediction_specs(args.prediction),
            score_baseline=[f"{name}={path}" for name, path in sorted(score_dirs.items())],
            external_baseline=list(args.external_baseline) or list(DEFAULT_BUILTIN_BASELINES),
            region=list(args.region) or list(DEFAULT_REGIONS),
            positive_threshold=float(args.positive_threshold),
            sasa_feature_index=129,
            rsa_feature_index=130,
            out_dir=str(args.classification_out_dir),
            dpi=180,
        )
        classification_report = evaluate_classification(eval_args)

    report = {
        "data_dir": args.data_dir,
        "pair_dir": args.pair_dir,
        "split": args.split,
        "split_source": args.split_source,
        "n_samples": len(samples),
        "requested_baselines": list(selected),
        "score_dirs": {name: str(path) for name, path in sorted(score_dirs.items())},
        "skipped": skipped,
        "sample_summary_csv": str(output_root / "external_score_baseline_samples.csv") if rows else "",
        "tool_paths": {
            "p2rank": str(p2rank_bin) if p2rank_bin else "",
            "fpocket": str(fpocket_bin) if fpocket_bin else "",
            "protcross_root": str(protcross_assets[0]) if protcross_assets else "",
        },
        "classification_report": classification_report,
        "scientific_contract": {
            "coordinate_endpoint": (
                "These external baselines are scalar mobility/disorder scores and are evaluated only as "
                "residue shift localization baselines, not as per-residue 3D displacement vectors."
            ),
            "prody_style_anm_gnm": (
                "Local implementation of ProDy-style elastic-network ANM/GNM square-fluctuation mobility, "
                "not a direct invocation of the ProDy package."
            ),
            "iupred2a": "Uses the public IUPred2A REST endpoint by UniProt accession, then aligns full UniProt scores to sample residues.",
            "p2rank": "Uses P2Rank's AlphaFold configuration and maps AF2 residue probabilities back through the processed AF2 indices.",
            "fpocket": "Assigns each residue the maximum fpocket pocket score among contacting pocket atom files.",
            "protcross": "Runs the local ProtCross project from /home/zero/ProtCross and maps binding probabilities by AF2 residue index.",
            "failure_policy": (
                "Per-sample external-tool failures abort the run by default; zero-valued failure records are written "
                "only when --allow-failed-zero-fallback is explicitly set."
            ),
        },
    }
    report_path = Path(args.report_path) if args.report_path else output_root / "external_score_baseline_report.json"
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
                    "n_samples": report["n_samples"],
                    "score_dirs": report["score_dirs"],
                    "skipped": report["skipped"],
                    "classification_report": (report.get("classification_report") or {}).get("report_path"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
