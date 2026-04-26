from __future__ import annotations

import json
import os

import numpy as np
import torch

GVP_DEFAULT_NUM_RBF = 16
GVP_DEFAULT_RBF_MAX = 20.0
GVP_DEFAULT_PAE_SCALE = 30.0
GVP_DEFAULT_PAE_CONF_TAU = 8.0
GVP_DEFAULT_SEQSEP_BINS = (1, 2, 4, 8, 16, 32, 64)


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


def _normalize_vectors(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return torch.where(norms > eps, vectors / norms.clamp_min(eps), torch.zeros_like(vectors))


def build_ca_node_vectors(pos: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build CA-only node vector features with shape (N, 3, 3)."""
    if pos.dim() != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must have shape (n, 3), got {tuple(pos.shape)}.")

    n = int(pos.size(0))
    node_v = torch.zeros((n, 3, 3), dtype=pos.dtype, device=pos.device)
    if n <= 1:
        return node_v

    ca_step = pos[1:] - pos[:-1]
    node_v[1:, 0, :] = ca_step
    node_v[:-1, 1, :] = ca_step
    if n > 2:
        node_v[1:-1, 2, :] = pos[2:] - pos[:-2]
    return _normalize_vectors(node_v, eps=eps)


def gvp_edge_scalar_dim(
    num_rbf: int = GVP_DEFAULT_NUM_RBF,
    seqsep_bins: tuple[int, ...] = GVP_DEFAULT_SEQSEP_BINS,
) -> int:
    return int(num_rbf) + 2 + len(seqsep_bins) + 1


def rbf_expand(
    distances: torch.Tensor,
    num_rbf: int = GVP_DEFAULT_NUM_RBF,
    rbf_max: float = GVP_DEFAULT_RBF_MAX,
) -> torch.Tensor:
    if num_rbf <= 0:
        return distances.new_zeros((distances.numel(), 0))
    centers = torch.linspace(0.0, float(rbf_max), int(num_rbf), dtype=distances.dtype, device=distances.device)
    if num_rbf == 1:
        width = torch.tensor(float(rbf_max), dtype=distances.dtype, device=distances.device).clamp_min(1.0)
    else:
        width = centers[1] - centers[0]
    d = distances.reshape(-1, 1)
    return torch.exp(-((d - centers.reshape(1, -1)) / width.clamp_min(1e-6)) ** 2)


def build_gvp_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    *,
    num_rbf: int = GVP_DEFAULT_NUM_RBF,
    rbf_max: float = GVP_DEFAULT_RBF_MAX,
    pae_scale: float = GVP_DEFAULT_PAE_SCALE,
    pae_conf_tau: float = GVP_DEFAULT_PAE_CONF_TAU,
    seqsep_bins: tuple[int, ...] = GVP_DEFAULT_SEQSEP_BINS,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scalar/vector edge features for a CA graph.

    edge_s = distance RBF + normalized PAE + PAE confidence + sequence-separation bucket.
    edge_v = unit vector from edge source to edge destination, shape (E, 1, 3).
    """
    if pos.dim() != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must have shape (n, 3), got {tuple(pos.shape)}.")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}.")

    edge_count = int(edge_index.size(1))
    scalar_dim = gvp_edge_scalar_dim(num_rbf=num_rbf, seqsep_bins=seqsep_bins)
    if edge_count == 0:
        edge_s = torch.zeros((0, scalar_dim), dtype=pos.dtype, device=pos.device)
        edge_v = torch.zeros((0, 1, 3), dtype=pos.dtype, device=pos.device)
        return edge_s, edge_v

    src, dst = edge_index
    rel = pos[dst] - pos[src]
    distances = torch.linalg.vector_norm(rel, dim=-1)
    edge_v = _normalize_vectors(rel, eps=eps).unsqueeze(1)

    rbf = rbf_expand(distances, num_rbf=num_rbf, rbf_max=rbf_max)
    if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.size(-1) >= 2:
        pae = edge_attr[:, 1].to(dtype=pos.dtype, device=pos.device)
    else:
        pae = torch.zeros((edge_count,), dtype=pos.dtype, device=pos.device)
    pae_norm = (pae / float(pae_scale)).clamp(min=0.0, max=1.0).unsqueeze(-1)
    pae_conf = torch.exp(-pae / float(pae_conf_tau)).unsqueeze(-1)

    sep = torch.abs(dst - src)
    boundaries = torch.tensor(seqsep_bins, dtype=sep.dtype, device=sep.device)
    sep_bucket = torch.bucketize(sep, boundaries, right=True)
    sep_onehot = torch.nn.functional.one_hot(sep_bucket, num_classes=len(seqsep_bins) + 1).to(dtype=pos.dtype)

    edge_s = torch.cat([rbf, pae_norm, pae_conf, sep_onehot], dim=-1)
    return edge_s, edge_v


def build_gvp_graph_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    *,
    num_rbf: int = GVP_DEFAULT_NUM_RBF,
    rbf_max: float = GVP_DEFAULT_RBF_MAX,
    pae_scale: float = GVP_DEFAULT_PAE_SCALE,
    pae_conf_tau: float = GVP_DEFAULT_PAE_CONF_TAU,
    seqsep_bins: tuple[int, ...] = GVP_DEFAULT_SEQSEP_BINS,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_v = build_ca_node_vectors(pos, eps=eps)
    edge_s, edge_v = build_gvp_edge_features(
        pos,
        edge_index,
        edge_attr,
        num_rbf=num_rbf,
        rbf_max=rbf_max,
        pae_scale=pae_scale,
        pae_conf_tau=pae_conf_tau,
        seqsep_bins=seqsep_bins,
        eps=eps,
    )
    return node_v, edge_s, edge_v
