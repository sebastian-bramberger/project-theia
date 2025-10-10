import json
import os

from project_theia.utils import get_paths
from compute_environment.compute_environment import LOGGING


def get_effective_batch_size(pl_config, config):
    batch_size = config.common.batch_size
    batch_size *= pl_config.num_nodes
    batch_size *= 1 if pl_config.gpus == 0 or pl_config.gpus is None else pl_config.gpus
    batch_size *= (
        1 if pl_config.accumulate_grad_batches is None else pl_config.accumulate_grad_batches
    )
    return batch_size


def get_train_samples(pl_config, config, dm):
    batch_size = get_effective_batch_size(pl_config, config)
    if pl_config.fast_dev_run is True:
        return batch_size
    elif type(pl_config.fast_dev_run) is int:
        return pl_config.fast_dev_run * batch_size
    elif type(pl_config.limit_train_batches) is int:
        return pl_config.limit_train_batches * batch_size
    elif type(pl_config.limit_train_batches) is float and pl_config.limit_train_batches < 1.0:
        return int(pl_config.limit_train_batches * len(dm.train_dataset))
    elif type(pl_config.overfit_batches) is int:
        return pl_config.overfit_batches * batch_size
    elif type(pl_config.overfit_batches) is float and pl_config.overfit_batches != 0.0:
        return int(pl_config.overfit_batches * len(dm.train_dataset))
    else:
        return len(dm.train_dataset)


def get_tracking_uri_mlflow():
    backend = LOGGING.mlflow_backend
    if backend == "filesystem":
        return "file://" + get_paths.get_mlruns_path()
    elif backend == "sqlite":
        server_file = get_paths.get_tracking_server_file_path()
        assert os.path.isfile(server_file), "Tracking server file not found, is the server running?"
        with open(server_file, "r") as f:
            server_data = json.load(f)
        return f"http://{server_data['host']}:{server_data['port']}"
    else:
        assert False, "Unknown mlflow backend {backend} specified in compute environment"
