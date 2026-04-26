from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from evopoint_da.data.graph import (
    build_gvp_graph_features,
    build_knn_edges,
    gvp_edge_scalar_dim,
)
from evopoint_da.models.module import EvoPointLitModule, _as_raw_plddt


class ModelHelperTests(unittest.TestCase):
    def test_as_raw_plddt_accepts_raw_or_normalized_values(self) -> None:
        normalized = _as_raw_plddt(torch.tensor([[0.8], [0.4]]))
        raw = _as_raw_plddt(torch.tensor([80.0, 40.0]))
        torch.testing.assert_close(normalized, raw)

    def test_predict_displacement_does_not_apply_multiplier_by_default(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=3.0,
            inference_disp_multiplier=2.0,
        )
        module.forward = lambda _batch: torch.ones((2, 3), dtype=torch.float32)

        pred = module.predict_displacement(batch=object())
        legacy_pred = module.predict_displacement(batch=object(), apply_inference_multiplier=True)

        torch.testing.assert_close(pred, torch.full((2, 3), 3.0))
        torch.testing.assert_close(legacy_pred, torch.full((2, 3), 6.0))

    def test_output_scale_multiplies_forward_displacement(self) -> None:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        edge_index, edge_attr = build_knn_edges(pos, k=2)
        batch = Data(
            x=torch.randn((3, 4), dtype=torch.float32),
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        base = EvoPointLitModule(in_channels=4, hidden_dim=8, num_layers=1, output_scale=1.0)
        scaled = EvoPointLitModule(in_channels=4, hidden_dim=8, num_layers=1, output_scale=3.0)
        scaled.load_state_dict(base.state_dict())

        torch.testing.assert_close(scaled.forward(batch), base.forward(batch) * 3.0)

    def test_gvp_forward_predicts_node_displacements(self) -> None:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        edge_index, edge_attr = build_knn_edges(pos, k=2)
        node_v, edge_s, edge_v = build_gvp_graph_features(pos, edge_index, edge_attr)
        batch = Data(
            x=torch.randn((3, 4), dtype=torch.float32),
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
        )
        module = EvoPointLitModule(
            in_channels=4,
            hidden_dim=16,
            num_layers=2,
            backbone_type="gvp",
            edge_scalar_dim=gvp_edge_scalar_dim(),
            gvp_vector_dim=4,
        )
        module.eval()

        pred = module.forward(batch)

        self.assertEqual(tuple(pred.shape), (3, 3))
        self.assertTrue(bool(torch.isfinite(pred).all()))

    def test_loss_gates_can_disable_auxiliary_terms(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=1.0,
            loss_gates_enabled=False,
            lambda_cos=100.0,
            lambda_mag=100.0,
            lambda_clash=100.0,
            lambda_high_plddt_l2=100.0,
            lambda_low_plddt_l2=100.0,
        )
        pred = torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]], dtype=torch.float32)
        target = torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float32)
        batch = Data(
            x=torch.ones((2, 1), dtype=torch.float32),
            pos=torch.zeros((2, 3), dtype=torch.float32),
            y=target,
            plddt=torch.tensor([[10.0], [95.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros((2, 2), dtype=torch.float32),
        )
        module.forward = lambda _batch: pred
        module.log = lambda *args, **kwargs: None

        loss = module._shared_step(batch, "train")
        expected = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1).mean()

        torch.testing.assert_close(loss, expected)

    def test_target_weighting_changes_main_loss_weights(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=1.0,
            loss_gates_enabled=False,
            target_weight_beta=1.0,
            target_weight_ref=1.0,
            target_weight_max=4.0,
        )
        pred = torch.zeros((2, 3), dtype=torch.float32)
        target = torch.tensor([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
        batch = Data(
            x=torch.ones((2, 1), dtype=torch.float32),
            pos=torch.zeros((2, 3), dtype=torch.float32),
            y=target,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
        )
        module.forward = lambda _batch: pred
        module.log = lambda *args, **kwargs: None

        loss = module._shared_step(batch, "train")
        weights = torch.tensor([1.5, 3.0], dtype=torch.float32)
        weights = weights / weights.mean()
        node_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        expected = (node_loss * weights).mean()

        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
