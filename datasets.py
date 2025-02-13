import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os


class ToyDataset(Dataset):
    def __init__(self, n_samples=256, image_channel=3, image_size=32):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        self.n_samples = n_samples
        self.images = torch.rand(n_samples, image_channel, image_size, image_size)
        self.labels = torch.randint(low=0, high=4, size=(n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ImageGenerationDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        args:
            root: path to image (str)
            transform: torchvision transforms
        rerturns: None
        """
        dirs = os.listdir(root)
        assert dirs, f"nothing found in {root}"
        # check if root includes directories for files
        if os.path.isdir(os.path.join(root, dirs[0])):
            self.class_names = dirs
            self.is_dir = True
        else:
            self.class_names = [root.split("/")[-1]]
            self.is_dir = False
        self.dirs = dirs  # dirs are directories or files
        self.root = root
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class LitImageGenerationDataset(L.LightningDataModule):
    def __init__(
        self,
        train_data: str | Dataset,
        val_data: str | Dataset = None,
        test_data: str | Dataset = None,
        image_size: int = 256,
        image_channel: int = 3,
        batch_size: int = 16,
    ) -> None:
        """
        args:
            train_data: path/to/train_data or torch Dataset
            val_data: path/to/val_data or torch Dataset
            test_data: path/to/test_data or torch Dataset
            image_size: image size
            image_channel: image channel
            batch_size: batch size
        """
        assert isinstance(
            train_data, (str, Dataset)
        ), "train_data should be a str or a torch Dataset"
        if val_data:
            assert isinstance(
                val_data, (str, Dataset)
            ), "val_data should be a str or a torch Dataset"
        if test_data:
            assert isinstance(
                test_data, (str, Dataset)
            ), "test_data should be a str or a torch Dataset"
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.batch_size = batch_size
        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        if isinstance(train_data, str):
            train_data = ImageFolder(root=train_data, transform=train_transform)

        if val_data:
            if isinstance(val_data, str):
                val_data = ImageFolder(root=val_data, transform=val_transform)

        if test_data:
            if isinstance(test_data, str):
                test_data = ImageFolder(root=test_data, transform=val_transform)

        if val_data and not test_data:
            train_data, test_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        if not val_data and test_data:
            train_data, val_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        if not val_data and not test_data:
            train_data, val_data, test_data = torch.utils.data.random_split(
                dataset=train_data,
                lengths=[0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(42),
            )
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
