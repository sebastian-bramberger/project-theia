from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

import sys

from project_theia.logging import utils

from project_theia.logging.utils import get_tracking_uri_mlflow


def get_callbacks(dm, train_config, pl_config, log_dir, data_config):
    cbs = {}

    checkpoint_cb = ModelCheckpoint(
        monitor=train_config.ckpt_metric,
        mode=train_config.ckpt_mode,
        dirpath=log_dir.name,
        filename="{epoch}_{" + train_config.ckpt_metric + ":.4f}",
        save_top_k=3,
        save_last=True,
    )
    cbs["checkpoint"] = checkpoint_cb

    mlflow_params, mlflow_tags = get_mlflow_params_tags(pl_config, data_config, train_config, dm)
    mlf_logging_cb = callbacks.MLFlowLogging(
        train_config.job_id,
        mlflow_tags,
        mlflow_params,
        log_dir,
        mlflow_params["train_samples"],
    )
    cbs["mlf_logging"] = mlf_logging_cb

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    cbs["lr_monitor"] = lr_monitor

    early_stopping = EarlyStopping(
        monitor=train_config.early_stopping_monitor,
        min_delta=train_config.early_stopping_min_delta,
        patience=train_config.early_stopping_patience,
        mode=train_config.early_stopping_mode,
        verbose=True,
    )
    if train_config.early_stopping:
        cbs["early_stopping"] = early_stopping

    if train_config.log_gpu_stats and pl_config.gpus is not None and pl_config.gpus > 0:
        cbs["gpu_stats"] = callbacks.MLFlowGPUStatsMonitor()

    return cbs


def get_mlflow_params_tags(pl_config, data_config, train_config, dm):
    mlflow_params = {
        "len_train_data": len(dm.train_dataset),
        "len_val_data": len(dm.val_dataset),
        "len_pred_data": len(dm.pred_dataset),
        "effective_train_batch_size": utils.get_effective_batch_size(pl_config, data_config),
        "train_samples": utils.get_train_samples(pl_config, data_config, dm),
    }
    mlflow_tags = {"cmd": " ".join(sys.argv)}
    if data_config.common.manual_overfit_batches > 0:
        mlflow_tags["overfit_imgs"] = "\n".join(dm.get_train_overfit_names())

    if train_config.description is not None:
        mlflow_tags["mlflow.note.content"] = train_config.description

    return mlflow_params, mlflow_tags
