from dataclasses import dataclass, field


@dataclass
class DataCommonConfig:
    batch_size: int = 32


@dataclass
class MnistConfig:
    common: DataCommonConfig = field(default_factory=DataCommonConfig)
    val_split: int = 5000
    num_workers: int = 4
    normalize: bool = False
    seed: int = 42