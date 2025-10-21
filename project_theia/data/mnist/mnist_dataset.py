import lightning as L
import torch
from typing import Callable, Optional, Tuple
from collections.abc import Sized


from static import _DATASETS_PATH

from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.datasets import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE


class MNISTDataModule(L.LightningDataModule):
    """Standard MNIST, train, val, test splits and transforms.
doctest: +ELLIPSIS
    <...mnist_datamodule.MNISTDataModule object at ...>

    """

    name = "mnist"

    def __init__(
        self,
        data_dir: str = _DATASETS_PATH,
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__()

        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Split the train and valid dataset."""
        extra = {"transform": self.default_transforms} if self.default_transforms else {}
        dataset: Dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        assert isinstance(dataset, Sized)
        train_length = len(dataset)
        self.dataset_train, self.dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        """MNIST train set removes a subset to use for validation."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """MNIST val set uses a subset of the training set for validation."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """MNIST test set uses the test split."""
        extra = {"transform": self.default_transforms} if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=False, download=False, **extra)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )


    def get_img_features(self) -> int:
        """Number of channels in pictures"""
        return 3


    def get_img_dims(self) -> Tuple[int, int]:
        dataset = MNIST(self.data_dir, train=True, download=True)
        img, _ = dataset[0]
        return tuple(img.size[::-1]) if hasattr(img, "size") else (1, 28, 28)


    def get_class_names(self):
        return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


    def get_original_img_dims(self) -> Tuple[int, int, int]:
        """Return the original (C, H, W) dimensions of MNIST images before any transforms."""
        return (1, 28, 28)


    def get_classes(self):
        """Number of classes in segmentation (including background)"""
        return len(self.get_class_names())

    @property
    def default_transforms(self) -> Optional[Callable]:
        if not _TORCHVISION_AVAILABLE:
            return None

        from torchvision import transforms

        if self.normalize:
            mnist_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        else:
            mnist_transforms = transforms.ToTensor()

        return mnist_transforms