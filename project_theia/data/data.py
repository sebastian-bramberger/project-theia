from typing import Tuple
from project_theia.data.mnist import mnist_dataset
from project_theia.data.data_spec import create_dataspec_from_data_module
from project_theia.data.data_config import MnistConfig



def get_mnist_data_module(config: MnistConfig):
    dm = mnist_dataset.MNISTDataModule(
        val_split=config.val_split,
        num_workers=config.num_workers,
        normalize=config.normalize,
        seed=config.seed,
        ** config.common.__dict__
    )
    data_spec = create_dataspec_from_data_module(dm)
    return dm, data_spec



def get_data_module(data_config):
    data_dispatch = {
        MnistConfig.__name__: get_mnist_data_module
    }
    return data_dispatch[data_config.__class__.__name__](data_config)
