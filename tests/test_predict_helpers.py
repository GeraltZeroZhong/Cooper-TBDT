from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_Predict import (
    _feature_edges_for_node_count,
    _load_prediction_features,
    _summarize_prediction_bins,
    build_arg_parser,
)


class PredictHelperTests(unittest.TestCase):
    def test_parser_allows_auto_feature_generation_without_feature_pt(self) -> None:
        args = build_arg_parser().parse_args(["--pdb_file", "input.pdb", "--ckpt_path", "model.ckpt"])

        self.assertIsNone(args.feature_pt)
        self.assertEqual(args.esm_weights, "esmc_weights/esmc_600m_2024_12_v0.pth")
        self.assertEqual(args.pca_path, "data/pca_esmc_128.pkl")

    def test_load_prediction_features_requires_feature_pt_by_default(self) -> None:
        args = argparse.Namespace(feature_pt=None)
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
                    "residue_ids": ["A_1", "A_2"],
                    "sequence": "AG",
                },
                feature_path,
            )
            args = argparse.Namespace(
                feature_pt=str(feature_path),
                feature_pos_tolerance=1e-3,
            )

            payload = _load_prediction_features(
                args,
                parsed={"residue_ids": ["A_1", "A_2"], "sequence": "AG"},
                pos=torch.zeros((2, 3), dtype=torch.float32),
                expected_in=144,
            )

            self.assertEqual(tuple(payload["x"].shape), (2, 144))
            torch.testing.assert_close(payload["edge_index"], edge_index)
            torch.testing.assert_close(payload["edge_attr"], edge_attr)

    def test_load_prediction_features_rejects_residue_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_path = Path(tmpdir) / "features.pt"
            torch.save(
                {
                    "x": torch.ones((2, 144), dtype=torch.float32),
                    "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    "edge_attr": torch.tensor([[1.0, 0.1], [1.0, 0.1]], dtype=torch.float32),
                    "residue_ids": ["A_1", "A_3"],
                    "sequence": "AG",
                },
                feature_path,
            )
            args = argparse.Namespace(feature_pt=str(feature_path), feature_pos_tolerance=1e-3)
            with self.assertRaisesRegex(ValueError, "residue_ids do not match"):
                _load_prediction_features(
                    args,
                    parsed={"residue_ids": ["A_1", "A_2"], "sequence": "AG"},
                    pos=torch.zeros((2, 3), dtype=torch.float32),
                    expected_in=144,
                )

    def test_feature_edges_reject_out_of_range_nodes(self) -> None:
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.1], [2.0, 0.2]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "outside the selected node range"):
            _feature_edges_for_node_count(edge_index, edge_attr, n=3)

    def test_summarize_prediction_bins_handles_empty_bins(self) -> None:
        stats = _summarize_prediction_bins(np.array([0.25, 1.5], dtype=np.float32), [0.0, 1.0, 2.0])
        self.assertEqual(stats["0to1"]["count"], 1)
        self.assertEqual(stats["1to2"]["count"], 1)
        self.assertEqual(stats["gt2"]["count"], 0)
        self.assertIsNone(stats["gt2"]["mean_abs_dr"])


if __name__ == "__main__":
    unittest.main()
