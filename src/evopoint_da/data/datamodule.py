import json
import os
from typing import Optional

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from .dataset import EvoPointDataset, build_split_file_lists


class EvoPointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/processed_graphs",
        batch_size: int = 4,
        num_workers: int = 0,
        calib_batch_size: int = 1,
        split_seed: int = 42,
        split_ranges: dict | None = None,
        fallback_num_features: int = 144,
        split_manifest_name: str = "split_manifest.json",
        allow_empty_fallback: bool = False,
        allow_length_truncation: bool = False,
        plddt_feature_index: int = 128,
    ):
        super().__init__()
        if split_ranges is None:
            split_ranges = {
                "train": [0.0, 0.7],
                "val": [0.7, 0.8],
                "calib": [0.8, 0.9],
                "test": [0.9, 1.0],
                "all": [0.0, 1.0],
            }
        self.save_hyperparameters()
        self._split_files: dict[str, list[str]] | None = None

    def _build_and_validate_splits(self):
        if self._split_files is not None:
            return
        self._split_files = build_split_file_lists(
            root=self.hparams.data_dir,
            split_ranges=self.hparams.split_ranges,
            split_seed=self.hparams.split_seed,
        )

        tracked = ["train", "val", "calib", "test"]
        pair_sets = {k: set(os.path.basename(p) for p in self._split_files.get(k, [])) for k in tracked}

        overlaps = {}
        for i, left in enumerate(tracked):
            for right in tracked[i + 1 :]:
                inter = sorted(pair_sets[left].intersection(pair_sets[right]))
                if inter:
                    overlaps[f"{left}__{right}"] = {"count": len(inter), "examples": inter[:10]}
        if overlaps:
            raise RuntimeError(f"Detected split leakage overlaps: {overlaps}")

        manifest = {
            "split_seed": int(self.hparams.split_seed),
            "split_ranges": OmegaConf.to_container(self.hparams.split_ranges, resolve=True),
            "counts": {k: len(self._split_files.get(k, [])) for k in self._split_files},
            "overlap_check": "passed",
        }
        manifest_path = os.path.join(self.hparams.data_dir, self.hparams.split_manifest_name)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _make_dataset(self, split: str) -> EvoPointDataset:
        if self._split_files is None:
            raise RuntimeError("Split files have not been initialized.")
        return EvoPointDataset(
            self.hparams.data_dir,
            split=split,
            split_seed=self.hparams.split_seed,
            split_ranges=self.hparams.split_ranges,
            fallback_num_features=self.hparams.fallback_num_features,
            file_list=self._split_files.get(split, []),
            allow_empty_fallback=self.hparams.allow_empty_fallback,
            allow_length_truncation=self.hparams.allow_length_truncation,
            plddt_feature_index=self.hparams.plddt_feature_index,
        )

    def setup(self, stage: Optional[str] = None):
        self._build_and_validate_splits()
        if stage in ("fit", None):
            self.train_set = self._make_dataset("train")
            self.val_set = self._make_dataset("val")
            self.calib_set = self._make_dataset("calib")
        if stage in ("test", None):
            self.test_set = self._make_dataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def calib_dataloader(self):
        return DataLoader(
            self.calib_set,
            batch_size=self.hparams.calib_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
