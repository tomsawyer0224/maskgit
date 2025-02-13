import sys

if not "." in sys.path:
    sys.path.append(".")
import unittest
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import datasets


class Test_datasets(unittest.TestCase):
    def setUp(self):
        image_size = 256
        image_channel = 3
        batch_size = 32
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        train_data = ImageFolder(
            root="../datasets/CelebrityFacesDataset", transform=train_transform
        )
        val_data = ImageFolder(
            root="../datasets/CelebrityFacesDataset", transform=val_transform
        )
        test_data = ImageFolder(
            root="../datasets/CelebrityFacesDataset", transform=val_transform
        )
        self.dataset = datasets.LitImageGenerationDataset(
            train_data="../datasets/CelebrityFacesDataset",
            val_data=None,  #'../datasets/CelebrityFacesDataset',
            test_data=None,  #'../datasets/CelebrityFacesDataset',
            image_size=256,
            image_channel=3,
            batch_size=32,
        )

    def test_train_dataloader(self):
        train_loader = self.dataset.train_dataloader()
        batch = next(iter(train_loader))
        imgs, lbls = batch
        print("___test train_dataloader:")
        print(f"images: {imgs.shape}, range: {imgs.min().item(), imgs.max().item()}")
        print(f"labels: {lbls.shape}, range: {lbls.min().item(), lbls.max().item()}")

    def test_val_dataloader(self):
        val_loader = self.dataset.val_dataloader()
        batch = next(iter(val_loader))
        imgs, lbls = batch
        print("___test val_dataloader:")
        print(f"images: {imgs.shape}, range: {imgs.min().item(), imgs.max().item()}")
        print(f"labels: {lbls.shape}, range: {lbls.min().item(), lbls.max().item()}")

    def test_test_dataloader(self):
        test_loader = self.dataset.test_dataloader()
        batch = next(iter(test_loader))
        imgs, lbls = batch
        print("___test test_dataloader:")
        print(f"images: {imgs.shape}, range: {imgs.min().item(), imgs.max().item()}")
        print(f"labels: {lbls.shape}, range: {lbls.min().item(), lbls.max().item()}")


"""
toyds = datasets.ToyDataset()
train, val, test = torch.utils.data.random_split(
    dataset = toyds,
    lengths = [0.8,0.1,0.1],
    generator = torch.Generator().manual_seed(42)
)
"""
if __name__ == "__main__":
    unittest.main()
