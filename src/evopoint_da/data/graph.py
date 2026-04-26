from __future__ import annotations

import json
import os

import numpy as np
import torch


def parse_pae_matrix(pae_path: str | None, n: int, *, strict: bool = False) -> np.ndarray:
    def _zeros() -> np.ndarray:
        return np.zeros((n, n), dtype=np.float32)

    def _extract_matrix(raw_obj) -> np.ndarray | None:
        if isinstance(raw_obj, dict):
            if "predicted_aligned_error" in raw_obj:
                return np.asarray(raw_obj["predicted_aligned_error"], dtype=np.float32)
            if "pae" in raw_obj:
                return np.asarray(raw_obj["pae"], dtype=np.float32)
            return None
        if isinstance(raw_obj, list):
            if len(raw_obj) == 0:
                return None
            if isinstance(raw_obj[0], dict):
                nested = _extract_matrix(raw_obj[0])
                if nested is not None:
                    return nested
            return np.asarray(raw_obj, dtype=np.float32)
        return np.asarray(raw_obj, dtype=np.float32)

    if n < 0:
        raise ValueError("n must be non-negative.")
    if pae_path is None or not os.path.exists(pae_path):
        if strict:
            raise FileNotFoundError(f"PAE file not found: {pae_path}")
        return _zeros()

    try:
        if pae_path.lower().endswith(".npy"):
            pae = np.asarray(np.load(pae_path), dtype=np.float32)
        else:
            with open(pae_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            extracted = _extract_matrix(raw)
            if extracted is None:
                raise ValueError("No PAE matrix field found.")
            pae = np.asarray(extracted, dtype=np.float32)
    except Exception:
        if strict:
            raise
        return _zeros()

    if pae.ndim != 2:
        if strict:
            raise ValueError(f"PAE matrix must be 2D, got shape {pae.shape}.")
        return _zeros()

    h, w = int(pae.shape[0]), int(pae.shape[1])
    if h != n or w != n:
        out = _zeros()
        hh, ww = min(n, h), min(n, w)
        out[:hh, :ww] = pae[:hh, :ww]
        return out
    return pae.astype(np.float32)


def build_knn_edges(
    pos: torch.Tensor,
    k: int = 16,
    pae: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pos.dim() != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must have shape (n, 3), got {tuple(pos.shape)}.")

    n = int(pos.shape[0])
    if n == 0 or k <= 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)
        edge_attr = torch.zeros((0, 2), dtype=pos.dtype, device=pos.device)
        return edge_index, edge_attr

    dist = torch.cdist(pos, pos)
    knn_idx = dist.topk(k=min(k + 1, n), largest=False).indices[:, 1:]
    row = torch.arange(n, device=pos.device).unsqueeze(1).repeat(1, knn_idx.shape[1]).reshape(-1)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
    if pae is None:
        edge_pae = torch.zeros_like(edge_dist)
    else:
        pae_t = torch.as_tensor(pae, dtype=pos.dtype, device=pos.device)
        if pae_t.shape[0] < n or pae_t.shape[1] < n:
            raise ValueError(f"PAE matrix shape {tuple(pae_t.shape)} is smaller than required {(n, n)}.")
        edge_pae = pae_t[row, col].unsqueeze(1)
    edge_attr = torch.cat([edge_dist, edge_pae], dim=1)
    return edge_index, edge_attr
