from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from project_theia.models_lightning.mnist.model_lightning_mnist import LitMNIST, LitMNISTConfig
from project_theia.data.data_spec import DataSpec

# Define the model and the training and test steps
# The model uses convolutional neural network layers
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# --- Prepare Data ---
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root=".", download=True, transform=transform)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])
mnist_test = MNIST(root=".", train=False, transform=transform)


train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=32)
test_loader = DataLoader(mnist_test, batch_size=32)

wandb_logger = WandbLogger(
    entity="us-guidance",
    project="sandbox",
    tags=["test", "local"],  # optional
    log_model=True           # log model checkpoints to W&B
)

# --- Train ---
trainer = pl.Trainer(max_epochs=3, accelerator="auto", logger=wandb_logger)
model = LitMNIST(config=LitMNISTConfig(),data_spec=DataSpec())
trainer.fit(model, train_loader, val_loader)

# --- Validate ---
val_results = trainer.validate(model, val_loader, verbose=False)
test_results = trainer.test(model, test_loader, verbose=False)

print(f"\nFinal validation accuracy: {val_results[0]['val_acc']:.4f}")
print(f"Final test accuracy:        {test_results[0]['test_acc']:.4f}")
