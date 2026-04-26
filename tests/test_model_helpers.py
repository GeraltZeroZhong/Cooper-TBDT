from __future__ import annotations

import unittest

import torch
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


if __name__ == "__main__":
    unittest.main()
