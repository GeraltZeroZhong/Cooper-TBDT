import glob
import hashlib
import json
import os
import random
from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset


GVP_FEATURE_SCHEMA_VERSION = 2
TBDT_NODE_ID_FIELDS = ("region_id",)
TBDT_GRAPH_ID_FIELDS = ("family_id", "state_id", "substrate_id")
TBDT_NODE_FLOAT_FIELDS = ("loss_weight",)
TBDT_NODE_MASK_FIELDS = (
    "barrel_core_mask",
    "plug_mask",
    "extracellular_loop_mask",
    "tonb_box_mask",
    "substrate_contact_mask",
    "eval_mask",
)


def _normalize_split_name(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    value = str(value).strip().lower()
    return value or None


def _metadata_split_for_file(path: str) -> str | None:
    try:
        sample = torch.load(path, weights_only=True)
    except Exception as exc:
        print(f"[EvoPointDataset] split metadata load error: {path} ({exc})")
        return None

    split = _normalize_split_name(sample.get("split"))
    if split:
        return split

    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    split = _normalize_split_name(metadata.get("split"))
    if split:
        return split

    manifest_row = metadata.get("manifest_row", {})
    if isinstance(manifest_row, dict):
        return _normalize_split_name(manifest_row.get("split"))
    return None


def build_split_file_lists(
    root: str,
    split_ranges: dict,
    split_seed: int,
    split_source: str = "range",
) -> dict[str, list[str]]:
    raw_files = sorted(glob.glob(os.path.join(root, "*.pt")))

    if str(split_source).lower() == "metadata":
        split_files: dict[str, list[str]] = {split: [] for split in split_ranges}
        for path in raw_files:
            split = _metadata_split_for_file(path)
            if split in split_files and split != "all":
                split_files[split].append(path)
        if "all" in split_files:
            split_files["all"] = raw_files
        return split_files

    if str(split_source).lower() != "range":
        raise ValueError(f"Unknown split_source: {split_source!r}")

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


def _as_graph_id(sample: dict, key: str) -> torch.Tensor:
    value = sample.get(key, None)
    if value is None:
        return torch.zeros((1,), dtype=torch.long)
    value = torch.as_tensor(value, dtype=torch.long).view(-1)
    if value.numel() == 0:
        return torch.zeros((1,), dtype=torch.long)
    return value[:1]


def _as_node_tensor(
    sample: dict,
    key: str,
    *,
    num_nodes: int,
    dtype: torch.dtype,
    default_value: bool | float | int,
) -> torch.Tensor:
    value = sample.get(key, None)
    default = torch.full((num_nodes,), default_value, dtype=dtype)
    if value is None:
        return default

    value = torch.as_tensor(value, dtype=dtype).view(-1)
    if value.numel() == num_nodes:
        return value
    if value.numel() == 1:
        return value.expand(num_nodes).clone()
    if value.numel() > num_nodes:
        return value[:num_nodes]
    if value.numel() > 0:
        default[: value.numel()] = value
    return default


class EvoPointDataset(InMemoryDataset):
    DEFAULT_SPLITS = {
        "train": (0.0, 0.7),
        "val": (0.7, 0.85),
        "test": (0.85, 1.0),
        "all": (0.0, 1.0),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_seed: int = 42,
        split_ranges: dict | None = None,
        file_list: list[str] | None = None,
    ):
        self.split_ranges = split_ranges or self.DEFAULT_SPLITS
        if split not in self.split_ranges:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.split_seed = split_seed
        self.file_list = file_list
        self._cache_key = self._build_cache_key()
        super().__init__(root)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _build_cache_key(self) -> str:
        payload = {
            "split": self.split,
            "split_seed": int(self.split_seed),
            "split_ranges": self.split_ranges,
            "file_list": [os.path.basename(path) for path in self.file_list] if self.file_list is not None else None,
            "gvp_feature_schema_version": GVP_FEATURE_SCHEMA_VERSION,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"graph_cache_{self.split}_{self._cache_key}.pt"]

    def process(self):
        if self.file_list is not None:
            files = self.file_list
        else:
            raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
            rng = random.Random(self.split_seed)
            rng.shuffle(raw_files)
            lo, hi = self.split_ranges[self.split]
            n = len(raw_files)
            files = raw_files[int(n * lo): int(n * hi)]

        if self.split == "val":
            print(f"[EvoPointDataset] val split selected {len(files)} files")

        data_list = []
        total_nodes = 0
        total_edges = 0
        for f in files:
            d = torch.load(f, weights_only=True)
            required = {"x", "pos", "y_delta", "plddt", "edge_index", "edge_attr", "node_v", "edge_s", "edge_v"}
            missing = sorted(required.difference(d))
            if missing:
                raise KeyError(f"{f} is missing required graph fields: {missing}")

            pos = d["pos"].float()
            y_delta = d["y_delta"].float()
            x = d["x"].float()

            if x.size(0) != pos.size(0) or y_delta.size(0) != pos.size(0):
                raise ValueError(
                    f"{f} has inconsistent node counts: x={x.size(0)}, pos={pos.size(0)}, y_delta={y_delta.size(0)}"
                )

            # Normalize coordinates to the origin to remove large-coordinate magnitude effects.
            pos = pos - pos.mean(dim=0, keepdim=True)
            total_nodes += int(pos.size(0))

            plddt = d["plddt"].float()
            edge_index = d["edge_index"].long()
            edge_attr = d["edge_attr"].float()
            node_v = d["node_v"].float()
            edge_s = d["edge_s"].float()
            edge_v = d["edge_v"].float()
            edge_index = edge_index.long()
            edge_attr = edge_attr.float()
            if edge_index.numel() > 0:
                valid_edges = (
                    (edge_index[0] >= 0)
                    & (edge_index[1] >= 0)
                    & (edge_index[0] < pos.size(0))
                    & (edge_index[1] < pos.size(0))
                )
                if not bool(valid_edges.all()):
                    raise ValueError(f"{f} contains edge indices outside node range.")
            if node_v.size(0) != pos.size(0) or edge_s.size(0) != edge_index.size(1) or edge_v.size(0) != edge_index.size(1):
                raise ValueError(
                    f"{f} has inconsistent GVP feature shapes: "
                    f"node_v={tuple(node_v.shape)}, edge_s={tuple(edge_s.shape)}, "
                    f"edge_v={tuple(edge_v.shape)}, nodes={pos.size(0)}, edges={edge_index.size(1)}"
                )

            total_edges += int(edge_index.size(1))
            tbdt_attrs = {
                key: _as_node_tensor(d, key, num_nodes=pos.size(0), dtype=torch.long, default_value=0)
                for key in TBDT_NODE_ID_FIELDS
            }
            tbdt_attrs.update(
                {
                    key: _as_graph_id(d, key)
                    for key in TBDT_GRAPH_ID_FIELDS
                }
            )
            tbdt_attrs.update(
                {
                    key: _as_node_tensor(d, key, num_nodes=pos.size(0), dtype=torch.float32, default_value=1.0)
                    for key in TBDT_NODE_FLOAT_FIELDS
                }
            )
            tbdt_attrs.update(
                {
                    key: _as_node_tensor(d, key, num_nodes=pos.size(0), dtype=torch.bool, default_value=False)
                    for key in TBDT_NODE_MASK_FIELDS
                }
            )
            data_list.append(
                Data(
                    x=x,
                    node_v=node_v,
                    pos=pos,
                    y=y_delta,
                    plddt=plddt,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_s=edge_s,
                    edge_v=edge_v,
                    pair_id=d.get("pair_id", os.path.splitext(os.path.basename(f))[0]),
                    residue_ids=d.get("residue_ids", None),
                    **tbdt_attrs,
                )
            )

        if self.split == "val":
            print(
                "[EvoPointDataset] val split kept "
                f"{len(data_list)}/{len(files)} files "
                f"avg_nodes={(total_nodes / max(1, len(data_list))):.2f}, "
                f"avg_edges={(total_edges / max(1, len(data_list))):.2f})"
            )

        if not data_list:
            raise RuntimeError(f"No graph samples found for split={self.split!r}, root={self.root!r}, files={len(files)}.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
