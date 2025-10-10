#!usr/bin/env python3
import argparse
import tempfile
import os
import sys
import shutil
from pathlib import Path
from dataclasses import asdict

import matplotlib.pyplot as plt


from lightning import seed_everything
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    ModelSummary
)

from project_theia.utils import get_paths, serialize, utils
from project_theia.training import logging_callbacks as callbacks
from project_theia.models_lightning import models_lightning

from project_theia import evaluate
from project_theia.evaluation import evaluate_config

from project_theia.data.data import get_data_module
from project_theia.logging import wandb


def save_config(
    output_path,
    config_path,
    train_config,
    pl_config,
    model_config,
    data_spec,
    data_config,
    run_config,
):
    output_path = Path(output_path)
    serialize.save(train_config, output_path / "train_config")
    serialize.save(pl_config, output_path / "pl_config")
    serialize.save(model_config, output_path / "model_config")
    serialize.save(data_spec, output_path / "data_spec")
    serialize.save(data_config, output_path / "data_config")
    serialize.save(run_config, output_path / "run_config")

    abs_config_path = get_paths.get_abs_path_from_config_path(config_path)
    config_name = os.path.basename(abs_config_path)

    if config_name == "slurm_script":
        config_name = f"train_config_{train_config.job_id}.py"

    shutil.copyfile(abs_config_path, output_path / config_name)


def train_model(
    config_path, train_config, pl_config, model_class, model_config, data_config, run_config
):
    if pl_config.overfit_batches > 0:
        print("overfit_batches is set. Please use manual_overfit_batches instead")
        sys.exit(1)

    train_config.seed = seed_everything(train_config.seed, workers=True)

    dm, data_spec = get_data_module(data_config)

    logger = wandb.get_logger(dm=dm, train_config=train_config, pl_config=pl_config, data_config=data_config) if getattr(train_config, "logging", True) else False

    with tempfile.TemporaryDirectory() as log_dir:
        save_config(
            log_dir,
            config_path=config_path,
            train_config=train_config,
            pl_config=pl_config,
            model_config=model_config,
            data_config=data_config,
            data_spec=data_spec,
            run_config=run_config,
        )

        profiler = SimpleProfiler(dirpath=log_dir, filename="profiling-results")
        cbs = wandb.get_callbacks(train_config=train_config, log_dir=log_dir)


        trainer = pl.Trainer(
            **asdict(pl_config),
            logger=logger,
            profiler=profiler,
            callbacks=[*cbs.values(), ModelSummary(max_depth=-1)],
            strategy="ddp"
        )

        try:
            if train_config.load_checkpoint is not None:
                assert os.path.isfile(train_config.load_checkpoint)
                model = model_class.load_from_checkpoint(
                    train_config.load_checkpoint,
                    config=model_config,
                    data_spec=data_spec,
                    data_config=data_config,
                )
            else:
                model = model_class(config=model_config, data_spec=data_spec, data_config=data_config)

            if pl_config.auto_lr_find:
                tuner_results = trainer.tune(model, datamodule=dm)
                lr_finder = tuner_results.get("lr_find") if tuner_results else None

                # Plot scan
                fig = lr_finder.plot(suggest=True)  # noqa: F841
                plt.savefig(str(Path(log_dir) / "lr_plot.png"))

                # Pick point based on plot, or get suggestion
                model.config.optimizer_config.learning_rate = lr_finder.suggestion()
                model.learning_rate = model.config.optimizer_config.learning_rate

            trainer.fit(model, dm)

            if trainer.is_global_zero:
                best_ckpt_src_path = cbs["checkpoint"].best_model_path
                cbs["logging"].log_param("best_checkpoint", os.path.basename(best_ckpt_src_path))
                best_ckpt_dst_path = Path(log_dir) / "best.ckpt"
                shutil.copy(best_ckpt_src_path, best_ckpt_dst_path)

            if not cbs["logging"].deactivate:
                artifacts_path = cbs["logging"].copy_log_dir_to_artifacts()

                print(f"\nSaved artifacts to {artifacts_path}")

            model.logger.finalize(status="FINISHED")
        except Exception as e:
            cbs["logging"].kill_run(f"exception: {e}")
            raise


    do_eval = trainer.is_global_zero and train_config.eval_after_train

    del trainer
    del model
    del dm

    if do_eval:
        eval_config = evaluate_config.EvaluateConfig(
            path=logger.experiment.id,
            eval_config_name="end_of_train_eval_config",
            epoch="best",
            metric_prefix="best",
            pred_writer=None,
            validate=True,
            predict=True,
            train_config=train_config,
            data_config=data_config,
        )

        evaluate.evaluate(eval_config, pl_config, config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_configs/default_train_run_config.py",
        help="path to config python file, absolute or relative to dir path",
    )

    args = parser.parse_args()

    train_config = utils.get_config_from_config_path(args.config_path, "get_train_run_config")
    pl_config = utils.get_config_from_config_path(args.config_path, "get_pl_config")

    train_model(
        config_path=args.config_path,
        train_config=train_config.train,
        pl_config=pl_config,
        model_class=models_lightning.MODEL_FROM_CONFIG_NAME[train_config.model.__class__.__name__],
        model_config=train_config.model,
        data_config=train_config.data,
        run_config=train_config,
    )


if __name__ == "__main__":
    main()
