import tempfile
from pathlib import Path
import subprocess


from project_theia.testing.validate_mlflow import ValidateMlflowTrainRun


class ValidateMnistMlflowTrainRun(ValidateMlflowTrainRun):
    def __init__(self, slurm_path):
        super().__init__(slurm_path)
        self.params += [
            "dim_in",
            "optimizer_config.scheduler_factor",
            "optimizer_config.scheduler_min_lr",
            "optimizer_config.scheduler_mode",
            "optimizer_config.scheduler_monitor",
            "optimizer_config.scheduler_patience",
            "optimizer_config.scheduler_threshold",
            "optimizer_config.scheduler",
            "swin_hp_transformer_config.ape",
            "swin_hp_transformer_config.attn_drop_rate",
            "swin_hp_transformer_config.depths",
            "swin_hp_transformer_config.drop_path_rate",
            "swin_hp_transformer_config.drop_rate",
            "swin_hp_transformer_config.embed_dim",
            "swin_hp_transformer_config.mlp_ratio",
            "swin_hp_transformer_config.num_heads",
            "swin_hp_transformer_config.patch_embed_norm_layer",
            "swin_hp_transformer_config.patch_norm",
            "swin_hp_transformer_config.patch_size",
            "swin_hp_transformer_config.qk_scale",
            "swin_hp_transformer_config.qkv_bias",
            "swin_hp_transformer_config.shift_size",
            "swin_hp_transformer_config.use_checkpoint",
            "swin_hp_transformer_config.window_size",
        ]
        self.files += ["swin_hp_test_run_config.py"]


def test_mnist_training():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"

        command = (
            "python3 -m project_theia.train --config_path project_theia/testing/mnist_test_run_config.py"
        )
        command += f" > {tmpfile}"

        assert subprocess.run(command, shell=True).returncode == 0

        mlflow_validator = ValidateMnistMlflowTrainRun(tmpfile)
        mlflow_validator.validate_mlflow_run()

    return True
