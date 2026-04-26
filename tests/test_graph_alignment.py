from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evopoint_da.data.alignment import apply_transform, kabsch_rotation
from evopoint_da.data.graph import build_knn_edges, parse_pae_matrix


class GraphAndAlignmentTests(unittest.TestCase):
    def test_kabsch_recovers_rigid_translation(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = points + np.array([3.0, -2.0, 0.5], dtype=np.float32)
        rotation, translation = kabsch_rotation(points, target)

        aligned = apply_transform(points, rotation, translation)
        np.testing.assert_allclose(aligned, target, atol=1e-5)

    def test_build_knn_edges_handles_empty_and_pae(self) -> None:
        empty_index, empty_attr = build_knn_edges(torch.zeros((0, 3)), k=16)
        self.assertEqual(tuple(empty_index.shape), (2, 0))
        self.assertEqual(tuple(empty_attr.shape), (0, 2))

        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        pae = np.arange(9, dtype=np.float32).reshape(3, 3)
        edge_index, edge_attr = build_knn_edges(pos, k=1, pae=pae)
        self.assertEqual(tuple(edge_index.shape), (2, 3))
        self.assertEqual(tuple(edge_attr.shape), (3, 2))
        for src, dst, attr in zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_attr.tolist()):
            self.assertEqual(attr[1], float(pae[src, dst]))

    def test_parse_pae_matrix_strict_and_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pae.json"
            path.write_text(json.dumps({"predicted_aligned_error": [[1.0, 2.0]]}), encoding="utf-8")
            parsed = parse_pae_matrix(str(path), 3)
            self.assertEqual(parsed.shape, (3, 3))
            self.assertEqual(float(parsed[0, 1]), 2.0)
            self.assertEqual(float(parsed[2, 2]), 0.0)

            bad = Path(tmpdir) / "bad.json"
            bad.write_text(json.dumps({"not_pae": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_pae_matrix(str(bad), 3, strict=True)


if __name__ == "__main__":
    unittest.main()
