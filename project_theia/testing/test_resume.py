import os
import tempfile
from pathlib import Path
import subprocess

from project_theia.testing.test_mnist import ValidateMnistMlflowTrainRun


def test_mnist_resume():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"

        train_command = (
            "python3 -m project_theia.train --config_path project_theia/testing/mnist_test_run_config.py"
        )
        command = f"{train_command} > {tmpfile}"

        assert subprocess.run(command, shell=True).returncode == 0

        mlflow_validator = ValidateMnistMlflowTrainRun(tmpfile)
        mlflow_validator.validate_mlflow_run()

    resume_env = os.environ.copy()
    resume_env["RESUME_RUN_ID"] = mlflow_validator.run_id

    resume_command = (
        "python3 -m project_theia.resume --config_path project_theia/testing/resume_test_run_config.py"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"
        command = f"{resume_command} > {tmpfile}"

        assert subprocess.run(command, shell=True, env=resume_env).returncode == 0

        mlflow_validator = ValidateMnistMlflowTrainRun(tmpfile)
        mlflow_validator.files.remove("mnist_test_run_config.py")
        mlflow_validator.files.append("resume_test_run_config.py")
        mlflow_validator.validate_mlflow_run()

    return True
