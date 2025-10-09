from typing import Tuple
from pytorch_lightning.demos import mnist_datamodule
from project_theia.data.data_spec import create_dataspec_from_data_module
from data_config import MnistConfig


def get_mnist_data_module(config):
    dm = mnist_datamodule.MNISTDataModule(
        val_split=config.val_split,
        num_workers=config.num_workers,
        normalize=config.normalize,
        seed=config.seed,
        batch_size=config.batch_size,
        ** config.common.__dict__
    )
    data_spec = create_dataspec_from_data_module(dm)
    return dm, data_spec



def get_data_module(data_config):
    data_dispatch = {
        MnistConfig.__name__: get_mnist_data_module
    }
    return data_dispatch[data_config.__class__.__name__](data_config)
