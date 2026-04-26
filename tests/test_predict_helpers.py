from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_Predict import (
    _build_auto_features,
    _feature_edges_for_node_count,
    _load_prediction_features,
    _summarize_prediction_bins,
)


class PredictHelperTests(unittest.TestCase):
    def test_build_auto_features_pads_to_expected_dim(self) -> None:
        parsed = {
            "residue_names": ["ALA", "GLY"],
            "plddts": np.array([90.0, 45.0], dtype=np.float32),
            "coords": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        }
        features = _build_auto_features(parsed, expected_in=30)
        self.assertEqual(tuple(features.shape), (2, 30))
        self.assertAlmostEqual(float(features[0, 20]), 0.9, places=5)

    def test_load_prediction_features_requires_feature_pt_by_default(self) -> None:
        args = argparse.Namespace(feature_pt=None, allow_fallback_features=False, save_auto_feature_pt=None)
        parsed = {"coords": np.zeros((2, 3), dtype=np.float32)}
        pos = torch.zeros((2, 3), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "--feature_pt is required"):
            _load_prediction_features(args, parsed, pos, expected_in=30)

    def test_load_prediction_features_preserves_feature_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_path = Path(tmpdir) / "features.pt"
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0, 0.1], [1.0, 0.1]], dtype=torch.float32)
            torch.save(
                {
                    "x": torch.ones((2, 144), dtype=torch.float32),
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                },
                feature_path,
            )
            args = argparse.Namespace(
                feature_pt=str(feature_path),
                allow_fallback_features=False,
                save_auto_feature_pt=None,
            )

            x, loaded_edge_index, loaded_edge_attr = _load_prediction_features(
                args,
                parsed={},
                pos=torch.zeros((2, 3), dtype=torch.float32),
                expected_in=144,
            )

            self.assertEqual(tuple(x.shape), (2, 144))
            torch.testing.assert_close(loaded_edge_index, edge_index)
            torch.testing.assert_close(loaded_edge_attr, edge_attr)

    def test_feature_edges_filter_to_current_node_count(self) -> None:
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.1], [2.0, 0.2]], dtype=torch.float32)
        filtered_index, filtered_attr = _feature_edges_for_node_count(edge_index, edge_attr, n=3)

        torch.testing.assert_close(filtered_index, torch.tensor([[0], [1]], dtype=torch.long))
        torch.testing.assert_close(filtered_attr, torch.tensor([[1.0, 0.1]], dtype=torch.float32))

    def test_summarize_prediction_bins_handles_empty_bins(self) -> None:
        stats = _summarize_prediction_bins(np.array([0.25, 1.5], dtype=np.float32), [0.0, 1.0, 2.0])
        self.assertEqual(stats["0to1"]["count"], 1)
        self.assertEqual(stats["1to2"]["count"], 1)
        self.assertEqual(stats["gt2"]["count"], 0)
        self.assertIsNone(stats["gt2"]["mean_abs_dr"])


if __name__ == "__main__":
    unittest.main()
