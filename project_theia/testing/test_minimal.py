from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from project_theia.models_lightning.mnist.model_lightning_mnist import LitMNIST, LitMNISTConfig
from project_theia.data.data_spec import DataSpec
from project_theia.data.data import get_mnist_data_module
from project_theia.data.data_config import MnistConfig

# Define the model and the training and test steps
# The model uses convolutional neural network layers
import os, certifi

os.environ["SSL_CERT_FILE"] = certifi.where()


def main():


    import os, time
    print({
      "pid": os.getpid(),
      "RANK": os.environ.get("RANK"),
      "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
      "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
      "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    wandb_logger = WandbLogger(
        entity="us-guidance",
        project="sandbox",
        tags=["test", "local"],  # optional
        log_model=True           # log model checkpoints to W&B
    )
    datamodule, data_spec = get_mnist_data_module(config=MnistConfig())

    # --- Train ---
    trainer = pl.Trainer(max_epochs=3, accelerator="auto", logger=wandb_logger)
    model = LitMNIST(config=LitMNISTConfig(), data_spec=data_spec)
    trainer.fit(model, datamodule=datamodule)

    # --- Validate ---
    val_results = trainer.validate(model, datamodule=datamodule, verbose=False)
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)

    print(f"\nFinal validation accuracy: {val_results[0]['val_acc']:.4f}")
    print(f"Final test accuracy:        {test_results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    # (Optional) makes behavior consistent across platforms
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    main()