import glob
import os
import random
from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset


def build_split_file_lists(root: str, split_ranges: dict, split_seed: int) -> dict[str, list[str]]:
    raw_files = sorted(glob.glob(os.path.join(root, "*.pt")))
    rng = random.Random(split_seed)
    shuffled = raw_files[:]
    rng.shuffle(shuffled)

    split_files: dict[str, list[str]] = {}
    n = len(shuffled)
    for split, (lo, hi) in split_ranges.items():
        lo_i = int(n * float(lo))
        hi_i = int(n * float(hi))
        split_files[split] = shuffled[lo_i:hi_i]
    return split_files


class EvoPointDataset(InMemoryDataset):
    DEFAULT_SPLITS = {
        "train": (0.0, 0.7),
        "val": (0.7, 0.8),
        "calib": (0.8, 0.9),
        "test": (0.9, 1.0),
        "all": (0.0, 1.0),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_seed: int = 42,
        split_ranges: dict | None = None,
        fallback_num_features: int = 144,
        file_list: list[str] | None = None,
    ):
        self.split_ranges = split_ranges or self.DEFAULT_SPLITS
        if split not in self.split_ranges:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.split_seed = split_seed
        self.fallback_num_features = fallback_num_features
        self.file_list = file_list
        super().__init__(root)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"graph_cache_{self.split}.pt"]

    def process(self):
        if self.file_list is not None:
            files = self.file_list
        else:
            raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
            random.seed(self.split_seed)
            random.shuffle(raw_files)
            lo, hi = self.split_ranges[self.split]
            n = len(raw_files)
            files = raw_files[int(n * lo): int(n * hi)]

        if self.split == "val":
            print(f"[EvoPointDataset] val split selected {len(files)} files")

        data_list = []
        skipped_missing = 0
        skipped_error = 0
        total_nodes = 0
        total_edges = 0
        for f in files:
            try:
                d = torch.load(f, weights_only=False)
            except Exception as e:
                skipped_error += 1
                print(f"[EvoPointDataset] skip load error: {f} ({e})")
                continue
            if "x" not in d or "pos" not in d or "y_delta" not in d:
                skipped_missing += 1
                continue

            pos = d["pos"].float()
            y_delta = d["y_delta"].float()
            x = d["x"].float()

            if x.size(0) != pos.size(0) or y_delta.size(0) != pos.size(0):
                min_len = min(x.size(0), pos.size(0), y_delta.size(0))
                if min_len <= 0:
                    skipped_error += 1
                    continue
                x = x[:min_len]
                pos = pos[:min_len]
                y_delta = y_delta[:min_len]

            # Normalize coordinates to the origin to remove large-coordinate magnitude effects.
            pos = pos - pos.mean(dim=0, keepdim=True)
            total_nodes += int(pos.size(0))

            edge_index = d.get("edge_index", None)
            edge_attr = d.get("edge_attr", None)
            if edge_index is None or edge_attr is None:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float32)
            total_edges += int(edge_index.size(1))
            data_list.append(
                Data(
                    x=x,
                    pos=pos,
                    y=y_delta,
                    plddt=d.get("plddt", None),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pair_id=d.get("pair_id", os.path.splitext(os.path.basename(f))[0]),
                    residue_ids=d.get("residue_ids", None),
                )
            )

        if self.split == "val":
            print(
                "[EvoPointDataset] val split kept "
                f"{len(data_list)}/{len(files)} files "
                f"(missing={skipped_missing}, load_error={skipped_error}, "
                f"avg_nodes={(total_nodes / max(1, len(data_list))):.2f}, "
                f"avg_edges={(total_edges / max(1, len(data_list))):.2f})"
            )

        if not data_list:
            data_list = [
                Data(
                    x=torch.zeros((1, self.fallback_num_features), dtype=torch.float32),
                    pos=torch.zeros((1, 3), dtype=torch.float32),
                    y=torch.zeros((1, 3), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                )
            ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
