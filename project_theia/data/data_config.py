from dataclasses import dataclass, field


@dataclass
class DataCommonConfig:
    train_worker: int = 2
    val_worker: int = 2
    shuffle: bool = True
    batch_size: int = 32
    val_batch_size: int = 32
    pred_batch_size: int = 4
    manual_overfit_batches: int = 0

    training_data_fraction: float = 1.0  # fraction of data to use for training
    data_fraction_seed: int = (
        42  # enables fixing the subset taken as fraction for comparing multiple runs
    )

    def __post_init__(self):
        assert 0.0 < self.training_data_fraction <= 1.0, "training_data_fraction not in (0.0, 1.0]"



@dataclass
class MnistConfig:
    common: DataCommonConfig = field(default_factory=DataCommonConfig)