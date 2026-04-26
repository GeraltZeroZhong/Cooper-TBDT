from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from evopoint_da.data.dataset import EvoPointDataset, build_split_file_lists


def _write_graph(path: Path, n: int = 3) -> None:
    torch.save(
        {
            "x": torch.ones((n, 4), dtype=torch.float32),
            "pos": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "y_delta": torch.zeros((n, 3), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 2), dtype=torch.float32),
            "plddt": torch.full((n, 1), 80.0),
        },
        path,
    )


class DatasetProcessingTests(unittest.TestCase):
    def test_split_file_lists_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx in range(10):
                _write_graph(root / f"{idx}.pt")
            ranges = {"train": (0.0, 0.6), "val": (0.6, 1.0)}

            first = build_split_file_lists(str(root), ranges, split_seed=7)
            second = build_split_file_lists(str(root), ranges, split_seed=7)
            self.assertEqual(first, second)
            self.assertFalse(set(first["train"]).intersection(second["val"]))

    def test_cache_name_depends_on_split_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_graph(root / "sample.pt")
            ds_a = EvoPointDataset(str(root), split="all", split_seed=1)
            ds_b = EvoPointDataset(str(root), split="all", split_seed=2)
            self.assertNotEqual(ds_a.processed_file_names, ds_b.processed_file_names)

    def test_empty_dataset_raises_unless_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                EvoPointDataset(str(tmpdir), split="all")

            ds = EvoPointDataset(str(tmpdir), split="all", allow_empty_fallback=True)
            self.assertEqual(len(ds), 1)

    def test_length_mismatch_requires_explicit_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torch.save(
                {
                    "x": torch.ones((2, 4), dtype=torch.float32),
                    "pos": torch.zeros((3, 3), dtype=torch.float32),
                    "y_delta": torch.zeros((3, 3), dtype=torch.float32),
                },
                root / "bad.pt",
            )

            with self.assertRaises(RuntimeError):
                EvoPointDataset(str(root), split="all")

            ds = EvoPointDataset(str(root), split="all", allow_length_truncation=True)
            self.assertEqual(ds[0].x.size(0), 2)

    def test_plddt_falls_back_to_feature_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            x = torch.zeros((2, 130), dtype=torch.float32)
            x[:, 128] = torch.tensor([0.8, 0.4])
            torch.save(
                {
                    "x": x,
                    "pos": torch.zeros((2, 3), dtype=torch.float32),
                    "y_delta": torch.zeros((2, 3), dtype=torch.float32),
                },
                root / "sample.pt",
            )

            ds = EvoPointDataset(str(root), split="all")
            self.assertAlmostEqual(float(ds[0].plddt[0, 0]), 0.8, places=6)
            self.assertAlmostEqual(float(ds[0].plddt[1, 0]), 0.4, places=6)

    def test_dataset_derives_gvp_features_for_legacy_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            torch.save(
                {
                    "x": torch.ones((3, 4), dtype=torch.float32),
                    "pos": torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "y_delta": torch.zeros((3, 3), dtype=torch.float32),
                    "edge_index": edge_index,
                    "edge_attr": torch.tensor([[1.0, 2.0], [1.0, 4.0]], dtype=torch.float32),
                },
                root / "sample.pt",
            )

            ds = EvoPointDataset(str(root), split="all")
            sample = ds[0]
            self.assertEqual(tuple(sample.node_v.shape), (3, 3, 3))
            self.assertEqual(sample.edge_s.size(0), edge_index.size(1))
            self.assertEqual(tuple(sample.edge_v.shape), (edge_index.size(1), 1, 3))
            batch = next(iter(DataLoader(ds, batch_size=1)))
            self.assertEqual(tuple(batch.node_v.shape), (3, 3, 3))
            self.assertEqual(tuple(batch.edge_v.shape), (edge_index.size(1), 1, 3))


if __name__ == "__main__":
    unittest.main()
