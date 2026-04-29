from __future__ import annotations

from typing import Any

import numpy as np
from Bio import Align


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rotation (R) and translation (t) from P to Q."""
    if P.shape != Q.shape:
        raise ValueError(f"Kabsch inputs must have identical shapes, got {P.shape} and {Q.shape}.")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Kabsch inputs must be shaped (n, 3), got {P.shape}.")
    if P.shape[0] < 3:
        raise ValueError("Kabsch alignment requires at least 3 points.")

    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q_mean - P_mean @ R.T
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ R.T + t


def iterative_kabsch(
    af2_coords: np.ndarray,
    holo_coords: np.ndarray,
    max_iter: int = 5,
    trim_ratio: float = 0.5,
) -> np.ndarray:
    """Robustly align by iteratively fitting the lowest-distance residue core."""
    if len(af2_coords) != len(holo_coords):
        raise ValueError(f"Alignment arrays must have the same length, got {len(af2_coords)} and {len(holo_coords)}.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not 0.0 < trim_ratio <= 1.0:
        raise ValueError("trim_ratio must be in (0, 1].")

    R, t = kabsch_rotation(af2_coords, holo_coords)
    best_R, best_t = R, t
    min_core_rmsd = float("inf")

    for _ in range(max_iter):
        aligned_iter = apply_transform(af2_coords, best_R, best_t)
        dists = np.linalg.norm(aligned_iter - holo_coords, axis=1)

        n_core = max(10, int(len(dists) * trim_ratio))
        n_core = min(n_core, len(dists))
        core_indices = np.argsort(dists)[:n_core]

        core_af2 = af2_coords[core_indices]
        core_holo = holo_coords[core_indices]
        if len(core_af2) < 3:
            break

        R_new, t_new = kabsch_rotation(core_af2, core_holo)
        core_aligned = apply_transform(core_af2, R_new, t_new)
        core_rmsd = np.sqrt(np.mean(np.sum((core_aligned - core_holo) ** 2, axis=1)))

        if core_rmsd < min_core_rmsd:
            min_core_rmsd = core_rmsd
            best_R, best_t = R_new, t_new
        else:
            break

    return apply_transform(af2_coords, best_R, best_t)


def _resolve_core_alignment_mask(
    common_ids: list[str],
    af2_idx: np.ndarray,
    holo_idx: np.ndarray,
    *,
    alignment_residue_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    alignment_af2_indices: list[int] | np.ndarray | set[int] | tuple[int, ...] | None = None,
    alignment_holo_indices: list[int] | np.ndarray | set[int] | tuple[int, ...] | None = None,
    aligned_position_mask: list[bool] | np.ndarray | tuple[bool, ...] | None = None,
) -> np.ndarray | None:
    """Resolve optional core-alignment selectors onto aligned residue positions."""
    n_aligned = len(common_ids)
    masks: list[np.ndarray] = []

    if alignment_residue_ids is not None:
        residue_id_set = {str(rid) for rid in alignment_residue_ids}
        masks.append(np.asarray([rid in residue_id_set for rid in common_ids], dtype=bool))

    if alignment_af2_indices is not None:
        af2_index_set = {int(idx) for idx in alignment_af2_indices}
        masks.append(np.isin(af2_idx, list(af2_index_set)))

    if alignment_holo_indices is not None:
        holo_index_set = {int(idx) for idx in alignment_holo_indices}
        masks.append(np.isin(holo_idx, list(holo_index_set)))

    if aligned_position_mask is not None:
        mask = np.asarray(aligned_position_mask, dtype=bool)
        if mask.shape != (n_aligned,):
            raise ValueError(f"aligned_position_mask must have length {n_aligned}, got shape {mask.shape}.")
        masks.append(mask)

    if not masks:
        return None

    core_mask = np.ones(n_aligned, dtype=bool)
    for mask in masks:
        core_mask &= mask
    return core_mask


def compute_displacement_target(
    af2_chains: dict[str, dict[str, Any]],
    holo_chains: dict[str, dict[str, Any]],
    *,
    alignment_residue_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    alignment_af2_indices: list[int] | np.ndarray | set[int] | tuple[int, ...] | None = None,
    alignment_holo_indices: list[int] | np.ndarray | set[int] | tuple[int, ...] | None = None,
    aligned_position_mask: list[bool] | np.ndarray | tuple[bool, ...] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, str]:
    """Build displacement targets from the best AF2/holo chain alignment.

    Optional alignment selectors restrict the Kabsch fit to stable core residues
    while displacement targets are still returned for every aligned residue.
    Existing callers that pass only ``af2_chains`` and ``holo_chains`` keep the
    previous robust full-alignment behavior.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -4.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    best_score = -float("inf")
    best_pair: tuple[str, str] | None = None

    for af2_id, af2_data in af2_chains.items():
        for holo_id, holo_data in holo_chains.items():
            len_a = len(af2_data["sequence"])
            len_h = len(holo_data["sequence"])
            if min(len_a, len_h) / max(len_a, len_h) < 0.4:
                continue

            try:
                score = aligner.score(af2_data["sequence"], holo_data["sequence"])
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_pair = (af2_id, holo_id)

    if not best_pair:
        raise ValueError("No alignable chains found.")

    best_af2_id, best_holo_id = best_pair
    seq_af2 = af2_chains[best_af2_id]["sequence"]
    seq_holo = holo_chains[best_holo_id]["sequence"]

    alignment = next(iter(aligner.align(seq_af2, seq_holo)))
    af2_blocks, holo_blocks = alignment.aligned

    af2_idx = []
    holo_idx = []
    for (a_s, a_e), (h_s, h_e) in zip(af2_blocks, holo_blocks):
        length = min(a_e - a_s, h_e - h_s)
        af2_idx.extend(range(a_s, a_s + length))
        holo_idx.extend(range(h_s, h_s + length))

    if len(af2_idx) < 15:
        raise ValueError(f"Alignment too short ({len(af2_idx)} residues).")

    matches = sum(1 for i, j in zip(af2_idx, holo_idx) if seq_af2[i] == seq_holo[j])
    identity = matches / len(af2_idx)
    if identity < 0.90:
        raise ValueError(f"Low sequence identity ({identity:.2%}). Likely wrong chain or severe mutation.")

    af2_idx_np = np.asarray(af2_idx, dtype=np.int64)
    holo_idx_np = np.asarray(holo_idx, dtype=np.int64)

    coords_af2_full = af2_chains[best_af2_id]["coords"]
    coords_holo_full = holo_chains[best_holo_id]["coords"]

    af2_sub = coords_af2_full[af2_idx_np]
    holo_sub = coords_holo_full[holo_idx_np]

    common_ids = [af2_chains[best_af2_id]["residue_ids"][i] for i in af2_idx_np]
    core_mask = _resolve_core_alignment_mask(
        common_ids,
        af2_idx_np,
        holo_idx_np,
        alignment_residue_ids=alignment_residue_ids,
        alignment_af2_indices=alignment_af2_indices,
        alignment_holo_indices=alignment_holo_indices,
        aligned_position_mask=aligned_position_mask,
    )

    if core_mask is None:
        af2_aligned = iterative_kabsch(af2_sub, holo_sub, max_iter=5, trim_ratio=0.6)
    else:
        n_core = int(core_mask.sum())
        if n_core < 3:
            raise ValueError(f"Core alignment selector matched {n_core} residues; at least 3 are required.")
        R, t = kabsch_rotation(af2_sub[core_mask], holo_sub[core_mask])
        af2_aligned = apply_transform(af2_sub, R, t)

    delta_r = holo_sub - af2_aligned

    return (
        delta_r.astype(np.float32),
        common_ids,
        af2_aligned.astype(np.float32),
        af2_idx_np,
        holo_idx_np,
        best_af2_id,
    )
