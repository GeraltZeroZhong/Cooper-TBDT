import os
import sys

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Run evaluation using the exact same data pipeline setup as train.py."""
    seed_everything(cfg.seed, workers=True)
    root = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(root, cfg.data.data_dir)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        enable_checkpointing=False,
    )

    ckpt_path = cfg.get("ckpt_path", None)
    # NOTE:
    # PyTorch 2.6 changed `torch.load(..., weights_only=...)` default from
    # False to True. Lightning forwards this through `trainer.test`.
    # Older checkpoints that include OmegaConf objects in metadata can fail to
    # unpickle with the strict safe loader unless we explicitly disable
    # `weights_only`.
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        weights_only=False,
    )


if __name__ == "__main__":
    main()
