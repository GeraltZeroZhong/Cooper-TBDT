from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from evopoint_da.data.dataset import EvoPointDataset, build_split_file_lists
from evopoint_da.models.module import EvoPointLitModule


DEFAULT_SPLIT_RANGES = {
    "train": [0.0, 0.7],
    "val": [0.7, 0.85],
    "test": [0.85, 1.0],
    "all": [0.0, 1.0],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict displacement vectors for processed TBDT graph files.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/processed_tbdt_gold_graphs")
    p.add_argument("--output-dir", default="artifacts/tbdt_v1/predictions")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--split-source", default="metadata", choices=["metadata", "range"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-path", default="artifacts/tbdt_v1/predict_tbdt_graphs_report.json")
    return p.parse_args()


def _pair_id_from_batch(batch: Any, default_stem: str) -> str:
    value = getattr(batch, "pair_id", None)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return default_stem


def predict_graphs(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_files = build_split_file_lists(
        str(data_dir),
        DEFAULT_SPLIT_RANGES,
        int(args.split_seed),
        split_source=str(args.split_source),
    )
    files = split_files.get(args.split, [])
    if not files:
        raise RuntimeError(f"No files selected for split={args.split!r} in {data_dir}")

    dataset = EvoPointDataset(
        str(data_dir),
        split=args.split,
        split_seed=int(args.split_seed),
        split_ranges=DEFAULT_SPLIT_RANGES,
        file_list=files,
    )
    if int(args.batch_size) != 1:
        raise ValueError("predict_tbdt_graphs currently requires --batch-size 1 to preserve file-to-output mapping.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers))
    model = EvoPointLitModule.load_from_checkpoint(args.ckpt, map_location=args.device, weights_only=False)
    model.eval().to(args.device)

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch, input_path in zip(loader, files, strict=True):
            batch = batch.to(args.device)
            pred = model.predict_displacement(batch).detach().cpu()
            stem = Path(input_path).stem
            pair_id = _pair_id_from_batch(batch, stem)
            output_path = out_dir / f"{stem}.pt"
            torch.save({"pair_id": pair_id, "pred_delta": pred}, output_path)
            outputs.append({"input": str(input_path), "output": str(output_path), "pair_id": pair_id, "n": int(pred.size(0))})

    report = {
        "ckpt": str(args.ckpt),
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "report_path": str(args.report_path),
        "split": str(args.split),
        "split_source": str(args.split_source),
        "n_predictions": len(outputs),
        "outputs": outputs,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    report = predict_graphs(parse_args())
    print(json.dumps({"report_path": report.get("report_path"), "n_predictions": report["n_predictions"]}, indent=2))


if __name__ == "__main__":
    main()
