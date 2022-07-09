import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms


class DeepDPMDataModule(pl.LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        # This code runs a single time to download the dataset, etc.
        pass


    def setup(self, stage):
        # Setup is run by every process across all nodes - set state here.

        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    # Other dataloaders basically unwrap datasets defined in setup.
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.mnist_val, batch_size=64)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.mnist_test, batch_size=64)

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass