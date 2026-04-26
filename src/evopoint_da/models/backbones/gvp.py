import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.in_channels > 0 and self.out_channels > 0:
            self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.register_parameter("weight", None)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.out_channels == 0:
            return v.new_zeros((*v.shape[:-2], 0, 3))
        if self.in_channels == 0:
            return v.new_zeros((*v.shape[:-2], self.out_channels, 3))
        return torch.einsum("...vc,ov->...oc", v, self.weight)


class VectorLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.channels = int(channels)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.channels == 0:
            return v
        scale = torch.sqrt((v * v).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(self.eps))
        return (v / scale.unsqueeze(-1)) * self.weight.view(*([1] * (v.dim() - 2)), self.channels, 1)


def _vector_dropout(v: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p <= 0.0:
        return v
    keep = 1.0 - p
    mask = torch.empty((*v.shape[:-1], 1), dtype=v.dtype, device=v.device).bernoulli_(keep)
    return v * mask / keep


class GVP(nn.Module):
    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        vector_hidden: int | None = None,
        *,
        scalar_activation: bool = True,
        vector_gate: bool = True,
    ):
        super().__init__()
        self.scalar_in = int(scalar_in)
        self.vector_in = int(vector_in)
        self.scalar_out = int(scalar_out)
        self.vector_out = int(vector_out)
        self.vector_hidden = int(vector_hidden or max(self.vector_in, self.vector_out, 1))
        self.vector_gate = bool(vector_gate and self.vector_out > 0)

        self.vector_hidden_proj = VectorLinear(self.vector_in, self.vector_hidden)
        self.scalar_proj = nn.Linear(self.scalar_in + self.vector_hidden, self.scalar_out)
        self.scalar_activation = nn.SiLU() if scalar_activation else nn.Identity()
        self.vector_out_proj = VectorLinear(self.vector_hidden, self.vector_out)
        self.gate_proj = nn.Linear(self.scalar_out, self.vector_out) if self.vector_gate else None

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vh = self.vector_hidden_proj(v)
        vh_norm = torch.linalg.vector_norm(vh, dim=-1)
        s_out = self.scalar_activation(self.scalar_proj(torch.cat([s, vh_norm], dim=-1)))

        v_out = self.vector_out_proj(vh)
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(s_out)).unsqueeze(-1)
            v_out = v_out * gate
        return s_out, v_out


class GVPLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.vector_dim = int(vector_dim)
        self.dropout = float(dropout)

        self.message_gvp = GVP(
            scalar_in=3 * self.scalar_dim,
            vector_in=3 * self.vector_dim,
            scalar_out=self.scalar_dim,
            vector_out=self.vector_dim,
        )
        self.update_gvp = GVP(
            scalar_in=2 * self.scalar_dim,
            vector_in=2 * self.vector_dim,
            scalar_out=self.scalar_dim,
            vector_out=self.vector_dim,
        )
        self.ffn_gvp = GVP(
            scalar_in=self.scalar_dim,
            vector_in=self.vector_dim,
            scalar_out=self.scalar_dim,
            vector_out=self.vector_dim,
        )
        self.scalar_norm_1 = nn.LayerNorm(self.scalar_dim)
        self.scalar_norm_2 = nn.LayerNorm(self.scalar_dim)
        self.vector_norm_1 = VectorLayerNorm(self.vector_dim)
        self.vector_norm_2 = VectorLayerNorm(self.vector_dim)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(s.size(0))
        if edge_index.numel() == 0:
            agg_s = s.new_zeros((n, self.scalar_dim))
            agg_v = v.new_zeros((n, self.vector_dim, 3))
        else:
            src, dst = edge_index
            msg_s_in = torch.cat([s[src], s[dst], edge_s], dim=-1)
            msg_v_in = torch.cat([v[src], v[dst], edge_v], dim=1)
            msg_s, msg_v = self.message_gvp(msg_s_in, msg_v_in)

            agg_s = s.new_zeros((n, self.scalar_dim))
            agg_v = v.new_zeros((n, self.vector_dim, 3))
            agg_s.index_add_(0, dst, msg_s)
            agg_v.index_add_(0, dst, msg_v)

            deg = torch.bincount(dst, minlength=n).to(dtype=s.dtype).clamp_min(1.0)
            agg_s = agg_s / deg.unsqueeze(-1)
            agg_v = agg_v / deg.view(n, 1, 1)

        upd_s, upd_v = self.update_gvp(torch.cat([s, agg_s], dim=-1), torch.cat([v, agg_v], dim=1))
        s = self.scalar_norm_1(s + F.dropout(upd_s, p=self.dropout, training=self.training))
        v = self.vector_norm_1(v + _vector_dropout(upd_v, p=self.dropout, training=self.training))

        ff_s, ff_v = self.ffn_gvp(s, v)
        s = self.scalar_norm_2(s + F.dropout(ff_s, p=self.dropout, training=self.training))
        v = self.vector_norm_2(v + _vector_dropout(ff_v, p=self.dropout, training=self.training))
        return s, v


class GVPBackbone(nn.Module):
    def __init__(
        self,
        node_scalar_dim: int,
        node_vector_dim: int = 3,
        edge_scalar_dim: int = 26,
        edge_vector_dim: int = 1,
        hidden_dim: int = 128,
        vector_dim: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_init_gain: float = 0.001,
    ):
        super().__init__()
        self.node_proj = GVP(
            scalar_in=node_scalar_dim,
            vector_in=node_vector_dim,
            scalar_out=hidden_dim,
            vector_out=vector_dim,
        )
        self.edge_proj = GVP(
            scalar_in=edge_scalar_dim,
            vector_in=edge_vector_dim,
            scalar_out=hidden_dim,
            vector_out=vector_dim,
        )
        self.layers = nn.ModuleList([GVPLayer(hidden_dim, vector_dim, dropout=dropout) for _ in range(num_layers)])
        self.output_gvp = GVP(
            scalar_in=hidden_dim,
            vector_in=vector_dim,
            scalar_out=hidden_dim,
            vector_out=1,
            vector_gate=False,
        )
        if self.output_gvp.vector_out_proj.weight is not None:
            nn.init.xavier_uniform_(self.output_gvp.vector_out_proj.weight, gain=output_init_gain)

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> torch.Tensor:
        s, v = self.node_proj(node_s, node_v)
        edge_s, edge_v = self.edge_proj(edge_s, edge_v)
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_s, edge_v)
        _, out_v = self.output_gvp(s, v)
        return out_v.squeeze(1)
