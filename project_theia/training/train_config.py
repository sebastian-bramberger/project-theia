from dataclasses import dataclass, field
from typing import Optional, Literal, Union, List, Dict
from datetime import timedelta
from pathlib import Path

from pytorch_lightning.accelerators import Accelerator

from project_theia.models_lightning.models_lightning import MODEL_CONFIGS_LITERAL
from project_theia.models_lightning.mnist.model_lightning_mnist import SimpleMNISTNetConfig

from project_theia.data.data_config import MnistConfig


@dataclass
class TrainConfig:
    logging: bool = True
    name: str = "train_config"
    job_id: str = "no_job_id"
    description: Optional[str] = None
    ckpt_metric: str = "val_iou_global_ignored"
    ckpt_mode: str = "max"
    eval_after_train: bool = True
    logging_project: str = "project_theia_mnist"  # project name for wandb
    logging_entity: Optional[str] = None  # team/org for wandb
    logging_tags: List[str] = field(default_factory=list) # for wandb
    log_gpu_stats: bool = True
    early_stopping: bool = False
    early_stopping_monitor: str = "val_iou_global_ignored"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    seed: Optional[int] = None
    load_checkpoint: Optional[str] = None
    logging_step_offset: int = 0


@dataclass
class SingleModelTrainRun:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: Literal[
        MnistConfig
    ] = field(default_factory=MnistConfig)
    model: MODEL_CONFIGS_LITERAL = field(default_factory=SimpleMNISTNetConfig)


@dataclass
class ResumeConfig:
    path: str  # MLFlow runid
    epoch: Optional[Literal["best", "last", "number"]] = "last"
    epoch_number: Optional[str] = None  # Optional epoch number to resume from if epoch == "number"
    train_run_config: SingleModelTrainRun = field(default_factory=SingleModelTrainRun)


@dataclass
class PLConfig:
    # Paths & logging
    default_root_dir: Optional[str] = None
    log_every_n_steps: int = 50

    # Device/precision/strategy (set these explicitly in code when building the Trainer)
    accelerator: Optional[str] = None        # e.g., "gpu", "cpu", "auto"
    devices: Optional[Union[int, List[int], str]] = None  # e.g., 1, 4, "auto"
    num_nodes: int = 1
    strategy: Optional[Union[str, object]] = None   # e.g., "ddp" or DDPStrategy(...)

    # Training loop
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"
    check_val_every_n_epoch: int = 1
    val_check_interval: Union[int, float] = 1.0
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None

    # Batching / limits
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    limit_predict_batches: Union[int, float] = 1.0

    # Dev / debug
    fast_dev_run: Union[int, bool] = False
    deterministic: bool = False
    benchmark: bool = False
    reload_dataloaders_every_n_epochs: int = 0  # replaces reload_dataloaders_every_epoch
    overfit_batches: Union[int, float] = 0.0    # you’re guarding this upstream anyway

    # Mixed precision
    precision: Union[int, str] = "32-true"  # prefer "16-mixed", "bf16-mixed", "32-true", "64-true"

    # Distributed
    sync_batchnorm: bool = False
    move_metrics_to_cpu: bool = False

    # Auto-tune
    auto_lr_find: Union[bool, str] = False
    auto_scale_batch_size: Union[str, bool] = False
