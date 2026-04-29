import os
import sys
import csv
import time
from typing import Any
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning import seed_everything
from torch.serialization import add_safe_globals

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


class ValidationMetricsCSVWriter(pl.Callback):
    """Persist validation epoch metrics into a dedicated CSV file."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "validation_metrics.csv"
        self._fieldnames: list[str] | None = None

    def _serialize(self, metric_value):
        if hasattr(metric_value, "detach"):
            metric_value = metric_value.detach()
        if hasattr(metric_value, "cpu"):
            metric_value = metric_value.cpu()
        if hasattr(metric_value, "item"):
            return metric_value.item()
        return metric_value

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        row = {"epoch": trainer.current_epoch}
        for key, value in metrics.items():
            key_str = str(key)
            if key_str.startswith("val/"):
                row[key_str] = self._serialize(value)

        if len(row) == 1:
            return

        if self._fieldnames is None:
            existing_keys = []
            if self.csv_path.exists():
                with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        existing_keys = [name for name in reader.fieldnames if name != "epoch"]

            merged_keys = sorted(set(existing_keys) | {k for k in row.keys() if k != "epoch"})
            self._fieldnames = ["epoch", *merged_keys]
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                    writer.writeheader()

        if self._fieldnames is not None:
            with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writerow(row)


class RuntimeCostCSVWriter(pl.Callback):
    """Persist epoch-level runtime and estimated compute-cost metrics."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "runtime_cost_metrics.csv"
        self._epoch_start_time: float | None = None
        self._fit_start_time: float | None = None
        self._header_written = False

    def _ensure_header(self):
        if self._header_written and self.csv_path.exists():
            return
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "epoch_seconds",
                        "fit_elapsed_seconds",
                        "estimated_gpu_hours",
                        "global_step",
                    ],
                )
                writer.writeheader()
        self._header_written = True

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        now = time.perf_counter()
        self._fit_start_time = now
        self._epoch_start_time = now
        self._ensure_header()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._epoch_start_time is None:
            return
        now = time.perf_counter()
        fit_elapsed = now - self._fit_start_time if self._fit_start_time is not None else float("nan")
        use_gpu = bool(getattr(trainer, "num_devices", 0)) and str(getattr(trainer, "accelerator", "")).lower() in {
            "cuda",
            "gpu",
        }
        row = {
            "epoch": trainer.current_epoch,
            "epoch_seconds": now - self._epoch_start_time,
            "fit_elapsed_seconds": fit_elapsed,
            "estimated_gpu_hours": (fit_elapsed / 3600.0) if use_gpu else 0.0,
            "global_step": trainer.global_step,
        }
        self._ensure_header()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "epoch_seconds",
                    "fit_elapsed_seconds",
                    "estimated_gpu_hours",
                    "global_step",
                ],
            )
            writer.writerow(row)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True). Lightning uses
    # this path for checkpoint restore in `trainer.test(..., ckpt_path=...)`.
    # Allowlist OmegaConf containers serialized in Lightning checkpoints so
    # safe weights-only loading can succeed.
    add_safe_globals([Any, ListConfig, OmegaDictConfig, ContainerMetadata])

    seed_everything(cfg.seed, workers=True)
    root = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(root, cfg.data.data_dir)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    init_checkpoint = str(cfg.get("init_checkpoint", "") or "").strip()
    if init_checkpoint:
        init_checkpoint_path = init_checkpoint
        if not os.path.isabs(init_checkpoint_path):
            init_checkpoint_path = os.path.join(root, init_checkpoint_path)
        checkpoint = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        exclude_prefixes = [str(item) for item in cfg.get("init_exclude_prefixes", [])]
        if exclude_prefixes:
            before_count = len(state_dict)
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not any(key == prefix or key.startswith(f"{prefix}.") for prefix in exclude_prefixes)
            }
            print(
                "Excluded "
                f"{before_count - len(state_dict)} checkpoint tensors by prefix: "
                f"{', '.join(exclude_prefixes)}"
            )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            "Initialized model weights from "
            f"{init_checkpoint_path} (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )

    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    study_name = str(cfg.get("study_name", "")).strip()
    if study_name:
        ckpt_dir = os.path.join(root, "checkpoints", study_name, run_time)
        val_metrics_dir = os.path.join(root, "val_metrics", study_name, run_time)
    else:
        ckpt_dir = os.path.join(root, "checkpoints", run_time)
        val_metrics_dir = os.path.join(root, "val_metrics", run_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(val_metrics_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-flex-{epoch:02d}-{val/flexible_mse:.4f}",
            monitor="val/flexible_mse",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-disp1to5-{epoch:02d}-{val/disp_1to5_mse:.4f}",
            monitor="val/disp_1to5_mse",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-disp1to2-{epoch:02d}-{val/disp_1to2_mse:.4f}",
            monitor="val/disp_1to2_mse",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-selection-{epoch:02d}-{val/disp_selection_mse:.4f}",
            monitor="val/disp_selection_mse",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        ),
        ValidationMetricsCSVWriter(output_dir=val_metrics_dir),
        RuntimeCostCSVWriter(output_dir=val_metrics_dir),
        RichProgressBar(),
    ]

    logger = hydra.utils.instantiate(cfg.logger)

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, enable_checkpointing=True)
    trainer.fit(model=model, datamodule=datamodule)

    best_flex_ckpt = callbacks[0].best_model_path
    best_disp_1to5_ckpt = callbacks[1].best_model_path
    best_disp_1to2_ckpt = callbacks[2].best_model_path
    best_selection_ckpt = callbacks[3].best_model_path

    if not bool(cfg.get("run_post_train_tests", True)):
        print("Skipping post-training Lightning tests because run_post_train_tests=false.")
        return

    if best_flex_ckpt:
        print(f"Running test with best-flex checkpoint: {best_flex_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_flex_ckpt, weights_only=False)
    else:
        print("Skipping best-flex checkpoint test: no best-flex checkpoint was saved.")

    if best_disp_1to5_ckpt:
        print(f"Running test with best-disp1to5 checkpoint: {best_disp_1to5_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_disp_1to5_ckpt, weights_only=False)
    else:
        print("Skipping best-disp1to5 checkpoint test: no best-disp1to5 checkpoint was saved.")

    if best_disp_1to2_ckpt:
        print(f"Running test with best-disp1to2 checkpoint: {best_disp_1to2_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_disp_1to2_ckpt, weights_only=False)
    else:
        print("Skipping best-disp1to2 checkpoint test: no best-disp1to2 checkpoint was saved.")

    if best_selection_ckpt:
        print(f"Running test with best-selection checkpoint: {best_selection_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_selection_ckpt, weights_only=False)
    else:
        print("Skipping best-selection checkpoint test: no best-selection checkpoint was saved.")


if __name__ == "__main__":
    main()
