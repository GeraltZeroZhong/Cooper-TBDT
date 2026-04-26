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

    def test_main_loss_can_focus_on_displacement_band(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=1.0,
            loss_gates_enabled=False,
            main_loss_min_disp=1.0,
            main_loss_max_disp=5.0,
            main_loss_outside_weight=0.0,
            main_loss_1to2_weight=2.0,
        )
        pred = torch.zeros((4, 3), dtype=torch.float32)
        target = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        batch = Data(
            x=torch.ones((4, 1), dtype=torch.float32),
            pos=torch.zeros((4, 3), dtype=torch.float32),
            y=target,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
        )
        module.forward = lambda _batch: pred
        module.log = lambda *args, **kwargs: None

        loss = module._shared_step(batch, "train")
        weights = torch.tensor([0.0, 2.0, 1.0, 0.0], dtype=torch.float32)
        weights = weights / weights.mean()
        node_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        expected = (node_loss * weights).mean()

        torch.testing.assert_close(loss, expected)

    def test_main_loss_can_use_mse(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=1.0,
            loss_gates_enabled=False,
            main_loss_type="mse",
        )
        pred = torch.zeros((1, 3), dtype=torch.float32)
        target = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32)
        batch = Data(
            x=torch.ones((1, 1), dtype=torch.float32),
            pos=torch.zeros((1, 3), dtype=torch.float32),
            y=target,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
        )
        module.forward = lambda _batch: pred
        module.log = lambda *args, **kwargs: None

        loss = module._shared_step(batch, "train")
        expected = F.mse_loss(pred, target, reduction="none").mean(dim=-1).mean()

        torch.testing.assert_close(loss, expected)

    def test_validation_logs_weighted_selection_metric(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=1.0,
            loss_gates_enabled=False,
            selection_disp_1to2_weight=0.7,
            selection_disp_1to5_weight=0.3,
        )
        pred = torch.zeros((3, 3), dtype=torch.float32)
        target = torch.tensor(
            [
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        batch = Data(
            x=torch.ones((3, 1), dtype=torch.float32),
            pos=torch.zeros((3, 3), dtype=torch.float32),
            y=target,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
        )
        logged = {}
        module.forward = lambda _batch: pred

        def _capture_log(name, value, *args, **kwargs):
            if hasattr(value, "detach"):
                value = value.detach().clone()
            logged.setdefault(name, value)

        module.log = _capture_log

        module._shared_step(batch, "val")

        expected_1to2 = F.mse_loss(pred[:1], target[:1])
        expected_1to5 = F.mse_loss(pred[:2], target[:2])
        expected_selection = 0.7 * expected_1to2 + 0.3 * expected_1to5
        torch.testing.assert_close(logged["val/disp_selection_mse"], expected_selection)


if __name__ == "__main__":
    unittest.main()
