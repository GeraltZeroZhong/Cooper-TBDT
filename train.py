import os
import sys
import csv
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
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


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True). Lightning uses
    # this path for checkpoint restore in `trainer.test(..., ckpt_path=...)`.
    # Allowlist OmegaConf containers serialized in Lightning checkpoints so
    # safe weights-only loading can succeed.
    add_safe_globals([ListConfig, OmegaDictConfig, ContainerMetadata])

    seed_everything(cfg.seed, workers=True)
    root = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(root, cfg.data.data_dir)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

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
        ValidationMetricsCSVWriter(output_dir=val_metrics_dir),
        RichProgressBar(),
    ]

    logger = hydra.utils.instantiate(cfg.logger)

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, enable_checkpointing=True)
    trainer.fit(model=model, datamodule=datamodule)

    best_flex_ckpt = callbacks[0].best_model_path
    best_disp_1to5_ckpt = callbacks[1].best_model_path
    best_disp_1to2_ckpt = callbacks[2].best_model_path

    if best_flex_ckpt:
        print(f"Running test with best-flex checkpoint: {best_flex_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_flex_ckpt)
    else:
        print("Skipping best-flex checkpoint test: no best-flex checkpoint was saved.")

    if best_disp_1to5_ckpt:
        print(f"Running test with best-disp1to5 checkpoint: {best_disp_1to5_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_disp_1to5_ckpt)
    else:
        print("Skipping best-disp1to5 checkpoint test: no best-disp1to5 checkpoint was saved.")

    if best_disp_1to2_ckpt:
        print(f"Running test with best-disp1to2 checkpoint: {best_disp_1to2_ckpt}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=best_disp_1to2_ckpt)
    else:
        print("Skipping best-disp1to2 checkpoint test: no best-disp1to2 checkpoint was saved.")


if __name__ == "__main__":
    main()
