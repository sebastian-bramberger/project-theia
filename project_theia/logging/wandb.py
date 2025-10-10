from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
import os
import sys

from project_theia.logging import utils


def get_logger(dm, train_config, pl_config, data_config) -> WandbLogger:
    wandb_config, wandb_tags = get_wandb_config_and_tags(
        dm=dm,
        train_config=train_config,
        pl_config=pl_config,
        data_config=data_config
    )

    return WandbLogger(
        project=train_config.project_name,
        name=train_config.job_id,
        tags=wandb_tags,
        config=wandb_config
    )


def get_callbacks(train_config, log_dir):
    """Factory for Lightning callbacks compatible with W&B logging."""
    cbs = {}

    # --- 1. Checkpointing ---
    checkpoint_cb = ModelCheckpoint(
        monitor=train_config.ckpt_metric,
        mode=train_config.ckpt_mode,
        dirpath=log_dir.name,
        filename="{epoch}_{" + train_config.ckpt_metric + ":.4f}",
        save_top_k=3,
        save_last=True,
    )
    cbs["checkpoint"] = checkpoint_cb

    # --- 3. Learning rate monitor ---
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    cbs["lr_monitor"] = lr_monitor

    # --- 4. Early stopping ---
    if train_config.early_stopping:
        early_stopping = EarlyStopping(
            monitor=train_config.early_stopping_monitor,
            min_delta=train_config.early_stopping_min_delta,
            patience=train_config.early_stopping_patience,
            mode=train_config.early_stopping_mode,
            verbose=True,
        )
        cbs["early_stopping"] = early_stopping

    return cbs


def get_wandb_config_and_tags(dm, train_config, pl_config, data_config):
    # --- Parameters ---
    wandb_config = {
        "len_train_data": len(dm.train_dataset),
        "len_val_data": len(dm.val_dataset),
        "len_pred_data": len(dm.pred_dataset),
        "effective_train_batch_size": utils.get_effective_batch_size(pl_config, data_config),
        "train_samples": utils.get_train_samples(pl_config, data_config, dm),
    }

    # --- Tags ---
    wandb_tags = []
    wandb_tags.append(f"cmd:{' '.join(sys.argv)}")

    if data_config.common.manual_overfit_batches > 0:
        wandb_tags.append("overfit")
        # you can also log the overfit image names as an artifact or config field:
        wandb_config["overfit_imgs"] = dm.get_train_overfit_names()

    if train_config.description is not None:
        wandb_tags.append("described")
        wandb_config["description"] = train_config.description

    return wandb_config, wandb_tags