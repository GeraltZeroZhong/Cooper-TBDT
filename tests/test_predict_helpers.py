from __future__ import annotations

import unittest

import numpy as np

from run_Predict import _build_auto_features, _summarize_prediction_bins


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

    def test_summarize_prediction_bins_handles_empty_bins(self) -> None:
        stats = _summarize_prediction_bins(np.array([0.25, 1.5], dtype=np.float32), [0.0, 1.0, 2.0])
        self.assertEqual(stats["0to1"]["count"], 1)
        self.assertEqual(stats["1to2"]["count"], 1)
        self.assertEqual(stats["gt2"]["count"], 0)
        self.assertIsNone(stats["gt2"]["mean_abs_dr"])


if __name__ == "__main__":
    unittest.main()
