import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from evopoint_da.utils.binning import build_bin_ranges as _build_bin_ranges

from .backbones.gvp import GVPBackbone


TBDT_MASK_METRICS = {
    "plug_mask": "plug_rmsd",
    "extracellular_loop_mask": "loop_rmsd",
    "tonb_box_mask": "tonb_box_rmsd",
    "barrel_core_mask": "barrel_core_rmsd_degradation_proxy",
    "substrate_contact_mask": "substrate_contact_rmsd",
    "eval_mask": "eval_rmsd",
}


def _in_disp_range(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Left-closed/right-open displacement range mask."""
    return (values >= float(low)) & (values < float(high))


def _as_raw_plddt(plddt: torch.Tensor) -> torch.Tensor:
    if plddt.dim() > 1:
        plddt = plddt.squeeze(-1)
    if plddt.numel() > 0 and float(plddt.detach().max().item()) <= 1.5:
        plddt = plddt * 100.0
    return plddt.float()


class EvoPointLitModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 144,
        hidden_dim: int = 128,
        num_layers: int = 4,
        node_vector_dim: int = 3,
        edge_scalar_dim: int = 26,
        edge_vector_dim: int = 1,
        gvp_vector_dim: int = 16,
        gvp_dropout: float = 0.1,
        use_tbdt_conditioning: bool = False,
        base_in_channels: int | None = None,
        region_vocab_size: int = 0,
        family_vocab_size: int = 0,
        state_vocab_size: int = 0,
        substrate_vocab_size: int = 0,
        condition_embedding_dim: int = 0,
        region_embedding_dim: int | None = None,
        family_embedding_dim: int | None = None,
        state_embedding_dim: int | None = None,
        substrate_embedding_dim: int | None = None,
        output_scale: float = 1.0,
        selection_disp_1to2_weight: float = 0.7,
        selection_disp_1to5_weight: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        coord_scale: float = 10.0,
        flexible_threshold: float = 1.0,
        barrel_core_loss_weight: float = 1.0,
        eval_region_loss_weight: float = 1.0,
        plug_loss_weight: float = 1.0,
        extracellular_loop_loss_weight: float = 1.0,
        tonb_box_loss_weight: float = 1.0,
        substrate_contact_loss_weight: float = 1.0,
        scaffold_anchor_weight: float = 0.0,
        scaffold_anchor_plddt_min: float = 70.0,
        lr_warmup_epochs: int = 10,
        coord_init_gain: float = 0.001,
        eps: float = 1e-8,
        disp_group_edges: list[float] | None = None,
        plddt_bin_edges: list[float] | None = None,
        test_disp_bin_edges: list[float] | None = None,
        default_total_epochs: int = 100,
        min_scheduler_epochs: int = 1,
        lr_start_factor: float = 1e-8,
    ):
        super().__init__()
        if disp_group_edges is None:
            disp_group_edges = [0.0, 1.0, 5.0]
        if plddt_bin_edges is None:
            plddt_bin_edges = [0.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        if test_disp_bin_edges is None:
            test_disp_bin_edges = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.use_tbdt_conditioning = bool(use_tbdt_conditioning)
        default_condition_dim = int(condition_embedding_dim)
        self.region_embedding_dim = int(
            default_condition_dim if region_embedding_dim is None else region_embedding_dim
        )
        self.family_embedding_dim = int(
            default_condition_dim if family_embedding_dim is None else family_embedding_dim
        )
        self.state_embedding_dim = int(default_condition_dim if state_embedding_dim is None else state_embedding_dim)
        self.substrate_embedding_dim = int(
            default_condition_dim if substrate_embedding_dim is None else substrate_embedding_dim
        )
        self.condition_embedding_total_dim = (
            self.region_embedding_dim
            + self.family_embedding_dim
            + self.state_embedding_dim
            + self.substrate_embedding_dim
            if self.use_tbdt_conditioning
            else 0
        )
        if self.use_tbdt_conditioning:
            base_in_channels = int(in_channels if base_in_channels is None else base_in_channels)
            in_channels = base_in_channels + self.condition_embedding_total_dim
        elif base_in_channels is None:
            base_in_channels = int(in_channels)
        self.save_hyperparameters()
        self.region_embedding = self._make_condition_embedding(
            region_vocab_size,
            self.region_embedding_dim if self.use_tbdt_conditioning else 0,
            "region",
        )
        self.family_embedding = self._make_condition_embedding(
            family_vocab_size,
            self.family_embedding_dim if self.use_tbdt_conditioning else 0,
            "family",
        )
        self.state_embedding = self._make_condition_embedding(
            state_vocab_size,
            self.state_embedding_dim if self.use_tbdt_conditioning else 0,
            "state",
        )
        self.substrate_embedding = self._make_condition_embedding(
            substrate_vocab_size,
            self.substrate_embedding_dim if self.use_tbdt_conditioning else 0,
            "substrate",
        )
        self.backbone = GVPBackbone(
            node_scalar_dim=in_channels,
            node_vector_dim=node_vector_dim,
            edge_scalar_dim=edge_scalar_dim,
            edge_vector_dim=edge_vector_dim,
            hidden_dim=hidden_dim,
            vector_dim=gvp_vector_dim,
            num_layers=num_layers,
            dropout=gvp_dropout,
            output_init_gain=coord_init_gain,
        )
        self.coord_scale = coord_scale
        self._test_disp_agg = {}

    def on_test_epoch_start(self):
        self._test_disp_agg = {}

    @staticmethod
    def _make_condition_embedding(vocab_size: int, embedding_dim: int, name: str) -> nn.Embedding | None:
        embedding_dim = int(embedding_dim)
        if embedding_dim <= 0:
            return None
        vocab_size = int(vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"{name}_vocab_size must be positive when {name}_embedding_dim > 0")
        return nn.Embedding(vocab_size, embedding_dim)

    def _node_condition_ids(self, batch, key: str) -> torch.Tensor:
        n_nodes = int(batch.x.size(0))
        device = batch.x.device
        values = getattr(batch, key, None)
        if values is None:
            return torch.zeros((n_nodes,), dtype=torch.long, device=device)
        values = values.to(device=device, dtype=torch.long).view(-1)
        if values.numel() == n_nodes:
            return values
        if values.numel() == 1:
            return values.expand(n_nodes)
        if values.numel() > n_nodes:
            return values[:n_nodes]
        padded = torch.zeros((n_nodes,), dtype=torch.long, device=device)
        if values.numel() > 0:
            padded[: values.numel()] = values
        return padded

    def _graph_condition_ids(self, batch, key: str) -> torch.Tensor:
        n_nodes = int(batch.x.size(0))
        device = batch.x.device
        values = getattr(batch, key, None)
        if values is None:
            return torch.zeros((n_nodes,), dtype=torch.long, device=device)

        values = values.to(device=device, dtype=torch.long).view(-1)
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            if values.numel() == n_nodes:
                return values
            if values.numel() == 1:
                return values.expand(n_nodes)
            return torch.zeros((n_nodes,), dtype=torch.long, device=device)

        batch_index = batch_index.to(device=device, dtype=torch.long).view(-1)
        if values.numel() == 0:
            values = torch.zeros((1,), dtype=torch.long, device=device)
        required = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
        if values.numel() < required:
            padded = torch.zeros((required,), dtype=torch.long, device=device)
            padded[: values.numel()] = values
            values = padded
        return values[batch_index]

    @staticmethod
    def _embedding_lookup(embedding: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        return embedding(ids.clamp(min=0, max=embedding.num_embeddings - 1))

    def _backbone_input_x(self, batch) -> torch.Tensor:
        x = batch.x
        if not self.use_tbdt_conditioning:
            return x

        base_in_channels = int(self.hparams.base_in_channels)
        if x.size(-1) != base_in_channels:
            raise ValueError(
                f"TBDT conditioning expects raw batch.x dim={base_in_channels}, got {x.size(-1)}"
            )

        condition_features = []
        if self.region_embedding is not None:
            condition_features.append(
                self._embedding_lookup(self.region_embedding, self._node_condition_ids(batch, "region_id"))
            )
        if self.family_embedding is not None:
            condition_features.append(
                self._embedding_lookup(self.family_embedding, self._graph_condition_ids(batch, "family_id"))
            )
        if self.state_embedding is not None:
            condition_features.append(
                self._embedding_lookup(self.state_embedding, self._graph_condition_ids(batch, "state_id"))
            )
        if self.substrate_embedding is not None:
            condition_features.append(
                self._embedding_lookup(self.substrate_embedding, self._graph_condition_ids(batch, "substrate_id"))
            )

        if not condition_features:
            return x
        return torch.cat([x, *condition_features], dim=-1)

    def _accumulate_disp_bin(
        self,
        suffix: str,
        sq_error: torch.Tensor,
        baseline_sq_error: torch.Tensor,
        mask: torch.Tensor,
    ):
        n_elem = int(mask.sum().item()) * sq_error.size(-1)
        sse_sum = sq_error[mask].sum() if n_elem > 0 else torch.zeros((), device=self.device, dtype=sq_error.dtype)
        baseline_sse_sum = (
            baseline_sq_error[mask].sum() if n_elem > 0 else torch.zeros((), device=self.device, dtype=sq_error.dtype)
        )

        self.log(
            f"test/disp_{suffix}_sse_sum",
            sse_sum,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=1,
        )

        if suffix not in self._test_disp_agg:
            self._test_disp_agg[suffix] = {
                "sse_sum": torch.zeros((), device=self.device, dtype=sq_error.dtype),
                "baseline_sse_sum": torch.zeros((), device=self.device, dtype=sq_error.dtype),
                "n_elem": 0,
            }

        self._test_disp_agg[suffix]["sse_sum"] = self._test_disp_agg[suffix]["sse_sum"] + sse_sum.detach()
        self._test_disp_agg[suffix]["baseline_sse_sum"] = (
            self._test_disp_agg[suffix]["baseline_sse_sum"] + baseline_sse_sum.detach()
        )
        self._test_disp_agg[suffix]["n_elem"] += n_elem

    def on_test_epoch_end(self):
        for suffix, agg in self._test_disp_agg.items():
            if agg["n_elem"] <= 0:
                continue
            mse = agg["sse_sum"] / agg["n_elem"]
            baseline_mse = agg["baseline_sse_sum"] / agg["n_elem"]
            self.log(f"test/disp_{suffix}_mse", mse)
            self.log(f"test/disp_{suffix}_rmsd", torch.sqrt(mse))
            self.log(f"test/baseline_disp_{suffix}_mse", baseline_mse)
            self.log(f"test/baseline_disp_{suffix}_rmsd", torch.sqrt(baseline_mse))

        # Aggregated displacement metric always uses [1.0, 5.0) for stable checkpoint comparison.
        agg_ranges = _build_bin_ranges(self.hparams.test_disp_bin_edges)
        agg_suffixes = [
            suffix
            for low, high, suffix in agg_ranges
            if high is not None and low >= 1.0 and high <= 5.0
        ]
        agg_sse_sum = None
        agg_baseline_sse_sum = None
        agg_n_elem = 0
        for suffix in agg_suffixes:
            if suffix not in self._test_disp_agg:
                continue
            if agg_sse_sum is None:
                agg_sse_sum = self._test_disp_agg[suffix]["sse_sum"]
                agg_baseline_sse_sum = self._test_disp_agg[suffix]["baseline_sse_sum"]
            else:
                agg_sse_sum = agg_sse_sum + self._test_disp_agg[suffix]["sse_sum"]
                agg_baseline_sse_sum = agg_baseline_sse_sum + self._test_disp_agg[suffix]["baseline_sse_sum"]
            agg_n_elem += self._test_disp_agg[suffix]["n_elem"]

        if agg_n_elem > 0:
            disp_1to5_mse = agg_sse_sum / agg_n_elem
            baseline_disp_1to5_mse = agg_baseline_sse_sum / agg_n_elem
            self.log("test/disp_1to5_mse", disp_1to5_mse)
            self.log("test/baseline_disp_1to5_mse", baseline_disp_1to5_mse)
            rel_improve = (baseline_disp_1to5_mse - disp_1to5_mse) / baseline_disp_1to5_mse.clamp_min(self.hparams.eps)
            self.log("test/summary/disp_1to5_rel_improve_vs_baseline", rel_improve)

    def forward(self, batch):
        x = self._backbone_input_x(batch)
        node_v = getattr(batch, "node_v", None)
        edge_s = getattr(batch, "edge_s", None)
        edge_v = getattr(batch, "edge_v", None)
        if node_v is None or edge_s is None or edge_v is None:
            raise KeyError("GVP-TBDT batches must contain precomputed node_v, edge_s, and edge_v features.")
        if node_v.size(0) != x.size(0) or edge_s.size(0) != batch.edge_index.size(1) or edge_v.size(0) != batch.edge_index.size(1):
            raise ValueError(
                "GVP feature shape mismatch: "
                f"node_v={tuple(node_v.shape)}, edge_s={tuple(edge_s.shape)}, "
                f"edge_v={tuple(edge_v.shape)}, nodes={x.size(0)}, edges={batch.edge_index.size(1)}"
            )
        return self.backbone(x, node_v, batch.edge_index, edge_s, edge_v) * self.hparams.output_scale

    def predict_displacement(self, batch):
        """Return displacement in the same real-coordinate scale used by validation."""
        delta_pred = self.forward(batch)
        return delta_pred * self.coord_scale

    def _log_disp_group_metrics(
        self,
        stage: str,
        delta_pred_real: torch.Tensor,
        y_true: torch.Tensor,
        gt_disp_mag: torch.Tensor,
        batch_size: int | None,
    ):
        groups = _build_bin_ranges(self.hparams.disp_group_edges)
        for low, high, suffix in groups:
            if high is None:
                mask = gt_disp_mag >= low
            else:
                mask = (gt_disp_mag >= low) & (gt_disp_mag < high)

            count = int(mask.sum().item())
            self.log(f"{stage}/disp_group/{suffix}_count", float(count), batch_size=batch_size)
            if count > 0:
                mse = F.mse_loss(delta_pred_real[mask], y_true[mask])
                mae = F.l1_loss(delta_pred_real[mask], y_true[mask])
                self.log(f"{stage}/disp_group/{suffix}_mse", mse, batch_size=batch_size)
                self.log(f"{stage}/disp_group/{suffix}_rmsd", torch.sqrt(mse), batch_size=batch_size)
                self.log(f"{stage}/disp_group/{suffix}_mae", mae, batch_size=batch_size)

    def _log_plddt_bin_metrics(
        self,
        stage: str,
        plddt: torch.Tensor,
        delta_pred_real: torch.Tensor,
        y_true: torch.Tensor,
        baseline_delta: torch.Tensor | None,
        batch_size: int | None,
    ):
        # pLDDT bins on raw pLDDT scale: [0,60), [60,70), [70,80), [80,90), [90,100]
        plddt_ranges = list(zip(self.hparams.plddt_bin_edges[:-1], self.hparams.plddt_bin_edges[1:]))
        for lower, upper in plddt_ranges:
            max_edge = self.hparams.plddt_bin_edges[-1]
            if upper < max_edge:
                plddt_mask = (plddt >= float(lower)) & (plddt < float(upper))
            else:
                plddt_mask = (plddt >= float(lower)) & (plddt <= float(upper))

            count = int(plddt_mask.sum().item())
            self.log(
                f"{stage}/plddt_bins/{lower}to{upper}/count",
                torch.tensor(float(count), device=self.device),
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum,
                batch_size=batch_size,
            )

            if count > 0:
                bin_mse = F.mse_loss(delta_pred_real[plddt_mask], y_true[plddt_mask])
                bin_mae = F.l1_loss(delta_pred_real[plddt_mask], y_true[plddt_mask])
                self.log(f"{stage}/plddt_bins/{lower}to{upper}/mse", bin_mse, batch_size=batch_size)
                self.log(f"{stage}/plddt_bins/{lower}to{upper}/rmsd", torch.sqrt(bin_mse), batch_size=batch_size)
                self.log(f"{stage}/plddt_bins/{lower}to{upper}/mae", bin_mae, batch_size=batch_size)

                if baseline_delta is not None:
                    baseline_bin_mse = F.mse_loss(baseline_delta[plddt_mask], y_true[plddt_mask])
                    baseline_bin_mae = F.l1_loss(baseline_delta[plddt_mask], y_true[plddt_mask])
                    self.log(
                        f"{stage}/baseline_plddt_bins/{lower}to{upper}/mse",
                        baseline_bin_mse,
                        batch_size=batch_size,
                    )
                    self.log(
                        f"{stage}/baseline_plddt_bins/{lower}to{upper}/rmsd",
                        torch.sqrt(baseline_bin_mse),
                        batch_size=batch_size,
                    )
                    self.log(
                        f"{stage}/baseline_plddt_bins/{lower}to{upper}/mae",
                        baseline_bin_mae,
                        batch_size=batch_size,
                    )

    @staticmethod
    def _aligned_bool_mask(batch, key: str, n_nodes: int, device: torch.device) -> torch.Tensor | None:
        value = getattr(batch, key, None)
        if value is None:
            return None
        value = value.to(device=device, dtype=torch.bool).view(-1)
        if value.numel() == n_nodes:
            return value
        if value.numel() == 1:
            return value.expand(n_nodes)
        if value.numel() > n_nodes:
            return value[:n_nodes]
        padded = torch.zeros((n_nodes,), dtype=torch.bool, device=device)
        if value.numel() > 0:
            padded[: value.numel()] = value
        return padded

    def _log_tbdt_mask_metrics(
        self,
        stage: str,
        batch,
        delta_pred_real: torch.Tensor,
        y_true: torch.Tensor,
        batch_size: int | None,
    ):
        n_nodes = int(y_true.size(0))
        eval_mask = self._aligned_bool_mask(batch, "eval_mask", n_nodes, y_true.device)
        use_eval_mask = eval_mask is not None and bool(eval_mask.any().item())
        for key, metric_name in TBDT_MASK_METRICS.items():
            mask = self._aligned_bool_mask(batch, key, n_nodes, y_true.device)
            if mask is None:
                continue
            if key != "eval_mask" and use_eval_mask:
                mask = mask & eval_mask

            count = int(mask.sum().item())
            self.log(
                f"{stage}/tbdt/{metric_name}_count",
                torch.tensor(float(count), device=self.device),
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum,
                batch_size=batch_size,
            )
            if count <= 0:
                continue

            mse = F.mse_loss(delta_pred_real[mask], y_true[mask])
            self.log(f"{stage}/tbdt/{metric_name}", torch.sqrt(mse), batch_size=batch_size)
            self.log(f"{stage}/tbdt/{metric_name}_mse", mse, batch_size=batch_size)

    def _shared_step(self, batch, stage: str):
        delta_pred = self.forward(batch)
        target_norm = batch.y / self.coord_scale
        target_mag_real = torch.norm(batch.y, dim=-1)
        mse_weights = torch.ones_like(target_mag_real)

        n_nodes = int(target_mag_real.numel())
        barrel_core_mask = self._aligned_bool_mask(batch, "barrel_core_mask", n_nodes, target_mag_real.device)
        scaffold_anchor_l2 = torch.zeros((), device=self.device, dtype=delta_pred.dtype)
        scaffold_anchor_count = 0

        def _apply_region_weight(mask_key: str, weight_value: float, *, mode: str = "max") -> None:
            nonlocal mse_weights
            weight = float(weight_value)
            if weight == 1.0:
                return
            mask = self._aligned_bool_mask(batch, mask_key, n_nodes, target_mag_real.device)
            if mask is None or not bool(mask.any().item()):
                return
            if mode == "min":
                updated = torch.minimum(mse_weights, torch.full_like(mse_weights, weight))
            else:
                updated = torch.maximum(mse_weights, torch.full_like(mse_weights, weight))
            mse_weights = torch.where(mask, updated, mse_weights)

        _apply_region_weight("barrel_core_mask", self.hparams.barrel_core_loss_weight, mode="min")
        _apply_region_weight("eval_mask", self.hparams.eval_region_loss_weight)
        _apply_region_weight("plug_mask", self.hparams.plug_loss_weight)
        _apply_region_weight("extracellular_loop_mask", self.hparams.extracellular_loop_loss_weight)
        _apply_region_weight("tonb_box_mask", self.hparams.tonb_box_loss_weight)
        _apply_region_weight("substrate_contact_mask", self.hparams.substrate_contact_loss_weight)

        if float(self.hparams.scaffold_anchor_weight) > 0.0 and barrel_core_mask is not None:
            scaffold_mask = barrel_core_mask.clone()
            if hasattr(batch, "plddt") and batch.plddt is not None and float(self.hparams.scaffold_anchor_plddt_min) > 0.0:
                plddt = _as_raw_plddt(batch.plddt).to(device=target_mag_real.device)
                if plddt.numel() == n_nodes:
                    scaffold_mask = scaffold_mask & (plddt >= float(self.hparams.scaffold_anchor_plddt_min))
            scaffold_anchor_count = int(scaffold_mask.sum().item())
            if scaffold_anchor_count > 0:
                scaffold_anchor_l2 = torch.sum(delta_pred[scaffold_mask].square(), dim=-1).mean()

        sample_loss_weights = torch.ones_like(target_mag_real)
        loss_weight = getattr(batch, "loss_weight", None)
        if loss_weight is not None:
            loss_weight = loss_weight.to(device=target_mag_real.device, dtype=target_mag_real.dtype).view(-1)
            if loss_weight.numel() == target_mag_real.numel():
                sample_loss_weights = loss_weight
            elif loss_weight.numel() == 1:
                sample_loss_weights = loss_weight.expand_as(target_mag_real)
            elif loss_weight.numel() > target_mag_real.numel():
                sample_loss_weights = loss_weight[: target_mag_real.numel()]
            elif loss_weight.numel() > 0:
                sample_loss_weights[: loss_weight.numel()] = loss_weight
            mse_weights = mse_weights * sample_loss_weights

        mse_weights = mse_weights / (mse_weights.mean().detach() + self.hparams.eps)

        loss_node_mse = F.smooth_l1_loss(delta_pred, target_norm, reduction="none").mean(dim=-1)
        loss_mse = (loss_node_mse * mse_weights).mean()
        delta_pred_real = delta_pred * self.coord_scale

        loss = loss_mse + float(self.hparams.scaffold_anchor_weight) * scaffold_anchor_l2

        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None and hasattr(batch, "ptr"):
            batch_size = batch.ptr.numel() - 1

        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), batch_size=batch_size)
        self.log(f"{stage}/loss_components/weighted_node", loss_mse, batch_size=batch_size)
        self.log(f"{stage}/loss_components/scaffold_anchor_l2", scaffold_anchor_l2, batch_size=batch_size)
        self.log(f"{stage}/weights/output_scale", float(self.hparams.output_scale), batch_size=batch_size)
        self.log(f"{stage}/weights/sample_loss_weight_mean", sample_loss_weights.mean(), batch_size=batch_size)
        self.log(
            f"{stage}/weights/sample_loss_weight_std",
            sample_loss_weights.std(unbiased=False),
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/weights/scaffold_anchor_weight",
            float(self.hparams.scaffold_anchor_weight),
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/weights/scaffold_anchor_count",
            torch.tensor(float(scaffold_anchor_count), device=self.device),
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/weights/barrel_core_loss_weight",
            float(self.hparams.barrel_core_loss_weight),
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/weights/eval_region_loss_weight",
            float(self.hparams.eval_region_loss_weight),
            batch_size=batch_size,
        )
        self.log(f"{stage}/weights/plug_loss_weight", float(self.hparams.plug_loss_weight), batch_size=batch_size)
        self.log(
            f"{stage}/weights/extracellular_loop_loss_weight",
            float(self.hparams.extracellular_loop_loss_weight),
            batch_size=batch_size,
        )
        self.log(f"{stage}/weights/tonb_box_loss_weight", float(self.hparams.tonb_box_loss_weight), batch_size=batch_size)
        self.log(
            f"{stage}/weights/substrate_contact_loss_weight",
            float(self.hparams.substrate_contact_loss_weight),
            batch_size=batch_size,
        )
        mse_real = F.mse_loss(delta_pred_real, batch.y)
        self.log(f"{stage}/loss_mse", mse_real, batch_size=batch_size)
        self.log(f"{stage}/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean(), batch_size=batch_size)
        self.log(f"{stage}/weights/mean", mse_weights.mean(), batch_size=batch_size)
        self.log(f"{stage}/weights/std", mse_weights.std(unbiased=False), batch_size=batch_size)

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        self._log_disp_group_metrics(stage, delta_pred_real, batch.y, gt_disp_mag, batch_size)
        self._log_tbdt_mask_metrics(stage, batch, delta_pred_real, batch.y, batch_size)
        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = _as_raw_plddt(batch.plddt)
            self._log_plddt_bin_metrics(
                stage=stage,
                plddt=plddt,
                delta_pred_real=delta_pred_real,
                y_true=batch.y,
                baseline_delta=None,
                batch_size=batch_size,
            )
        if stage == "val":
            disp_1to2_mse = None
            disp_1to5_mse = None
            disp_1to2_mask = _in_disp_range(gt_disp_mag, low=1.0, high=2.0)
            if disp_1to2_mask.any():
                disp_1to2_mse = F.mse_loss(delta_pred_real[disp_1to2_mask], batch.y[disp_1to2_mask])
                baseline_disp_1to2_mse = F.mse_loss(
                    torch.zeros_like(batch.y[disp_1to2_mask]),
                    batch.y[disp_1to2_mask],
                )
                disp_1to2_rel_improve = (baseline_disp_1to2_mse - disp_1to2_mse) / baseline_disp_1to2_mse.clamp_min(
                    self.hparams.eps
                )
                self.log(
                    "val/disp_1to2_mse",
                    disp_1to2_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=int(disp_1to2_mask.sum().item()),
                )
                self.log(
                    "val/disp_1to2_rel_improve_vs_baseline",
                    disp_1to2_rel_improve,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=int(disp_1to2_mask.sum().item()),
                )

            # Keep this metric semantically stable: always [1.0, 5.0).
            disp_1to5_mask = _in_disp_range(gt_disp_mag, low=1.0, high=5.0)
            if disp_1to5_mask.any():
                disp_1to5_mse = F.mse_loss(delta_pred_real[disp_1to5_mask], batch.y[disp_1to5_mask])
                self.log(
                    "val/disp_1to5_mse",
                    disp_1to5_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=int(disp_1to5_mask.sum().item()),
                )
            if disp_1to2_mse is not None and disp_1to5_mse is not None:
                selection_weight_sum = (
                    float(self.hparams.selection_disp_1to2_weight)
                    + float(self.hparams.selection_disp_1to5_weight)
                )
                if selection_weight_sum <= 0.0:
                    selection_weight_sum = 1.0
                disp_selection_mse = (
                    float(self.hparams.selection_disp_1to2_weight) * disp_1to2_mse
                    + float(self.hparams.selection_disp_1to5_weight) * disp_1to5_mse
                ) / selection_weight_sum
                self.log(
                    "val/disp_selection_mse",
                    disp_selection_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=int(disp_1to5_mask.sum().item()),
                )

        flexible_mask = gt_disp_mag > self.hparams.flexible_threshold
        flex_count = int(flexible_mask.sum().item())
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
        else:
            flex_mse = torch.zeros((), device=self.device, dtype=delta_pred_real.dtype)
        self.log(f"{stage}/flexible_mse", flex_mse, batch_size=batch_size)
        self.log(
            f"{stage}/flexible_count",
            torch.tensor(float(flex_count), device=self.device),
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_displacement(batch)

    def test_step(self, batch, batch_idx):
        delta_pred_real = self.predict_displacement(batch)
        sq_error = (delta_pred_real - batch.y) ** 2
        batch_size = getattr(batch, "num_graphs", None)
        if batch_size is None and hasattr(batch, "ptr"):
            batch_size = batch.ptr.numel() - 1

        loss_mse_real = F.mse_loss(delta_pred_real, batch.y)
        loss = loss_mse_real

        self.log("test/loss", loss)
        self.log("test/loss_mse", loss_mse_real)
        self.log("test/loss_mae", F.l1_loss(delta_pred_real, batch.y))
        self.log("test/loss_components/weighted_node", loss_mse_real)

        baseline_delta = torch.zeros_like(batch.y)
        baseline_sq_error = baseline_delta - batch.y
        baseline_sq_error = baseline_sq_error ** 2
        baseline_mse = F.mse_loss(baseline_delta, batch.y)
        self.log("test/baseline_mse", baseline_mse)
        self.log("test/baseline_mae", F.l1_loss(baseline_delta, batch.y))
        overall_rel_improve = (baseline_mse - loss_mse_real) / baseline_mse.clamp_min(self.hparams.eps)
        self.log("test/summary/overall_rel_improve_vs_baseline", overall_rel_improve)
        self.log("test/pred_magnitude", torch.norm(delta_pred_real, dim=-1).mean())

        gt_disp_mag = torch.norm(batch.y, dim=-1)
        self._log_disp_group_metrics("test", delta_pred_real, batch.y, gt_disp_mag, batch_size=batch_size)
        self._log_tbdt_mask_metrics("test", batch, delta_pred_real, batch.y, batch_size=batch_size)
        flexible_mask = gt_disp_mag > self.hparams.flexible_threshold
        if flexible_mask.any():
            flex_mse = F.mse_loss(delta_pred_real[flexible_mask], batch.y[flexible_mask])
            baseline_flex_mse = F.mse_loss(baseline_delta[flexible_mask], batch.y[flexible_mask])
            self.log("test/flexible_mse", flex_mse)
            self.log("test/flexible_rmsd", torch.sqrt(flex_mse))
            self.log("test/baseline_flexible_mse", baseline_flex_mse)
            self.log("test/baseline_flexible_rmsd", torch.sqrt(baseline_flex_mse))

        # Fine-grained displacement bins: [0,0.5), [0.5,1), [1,2), [2,3), [3,4), [4,5), [5,+inf)
        disp_bins = _build_bin_ranges(self.hparams.test_disp_bin_edges)
        for lower, upper, suffix in disp_bins:
            if upper is None:
                continue
            disp_mask = (gt_disp_mag >= lower) & (gt_disp_mag < upper)
            count = int(disp_mask.sum().item())
            self.log(
                f"test/disp_{suffix}_count",
                torch.tensor(float(count), device=self.device),
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum,
                batch_size=batch_size,
            )
            self._accumulate_disp_bin(suffix, sq_error, baseline_sq_error, disp_mask)

        gt_label = _build_bin_ranges(self.hparams.test_disp_bin_edges)[-1][2]
        gt_threshold = float(self.hparams.test_disp_bin_edges[-1])
        disp_mask_gt = gt_disp_mag >= gt_threshold
        count_gt5 = int(disp_mask_gt.sum().item())
        self.log(
            f"test/disp_{gt_label}_count",
            torch.tensor(float(count_gt5), device=self.device),
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
            batch_size=batch_size,
        )
        self._accumulate_disp_bin(gt_label, sq_error, baseline_sq_error, disp_mask_gt)

        # pLDDT-binned metrics
        if hasattr(batch, "plddt") and batch.plddt is not None:
            plddt = _as_raw_plddt(batch.plddt)
            self._log_plddt_bin_metrics(
                stage="test",
                plddt=plddt,
                delta_pred_real=delta_pred_real,
                y_true=batch.y,
                baseline_delta=baseline_delta,
                batch_size=batch_size,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        warmup_epochs = max(0, int(self.hparams.lr_warmup_epochs))
        total_epochs = int(getattr(self.trainer, "max_epochs", self.hparams.default_total_epochs))
        if total_epochs <= 0:
            total_epochs = self.hparams.default_total_epochs

        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.lr_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_epochs = max(int(self.hparams.min_scheduler_epochs), total_epochs - warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.hparams.min_scheduler_epochs), total_epochs),
            )

        return [optimizer], [scheduler]
