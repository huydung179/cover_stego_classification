import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

# Define transformations for training data
TRAIN_TF = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
])

# Define transformations for validation/test data
VALID_TF = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
])


def prepare_train_valid(data_dir: str, batch_size: int, seed: int = 42) -> (DataLoader, DataLoader):
    """
    Prepare DataLoaders for training and validation datasets.
    
    :param data_dir: Directory path containing the data.
    :param batch_size: Size of each batch.
    :param seed: Random seed for reproducibility.
    :return: Training and validation DataLoader objects.
    """
    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TF)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))  # 90% for training
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))  # Seed set for reproducibility

    # Apply different transforms to training and validation sets (post splitting)
    train_dataset.dataset.transform = TRAIN_TF
    valid_dataset.dataset.transform = VALID_TF

    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


class PairDataset(Dataset):
    def __init__(self, image1_paths, image2_paths, transform=None):
        self.image1_paths = image1_paths
        self.image2_paths = image2_paths
        self.transform = transform

    def __getitem__(self, index):
        img1 = Image.open(self.image1_paths[index])
        img2 = Image.open(self.image2_paths[index])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.image1_paths)


def prepare_train_valid_autoencoder(data_dir: str, batch_size: int, seed: int = 42) -> (DataLoader, DataLoader):
    cover_dir = data_dir + '/cover'
    stego_dir = data_dir + '/stego'

    cover_paths = sorted([cover_dir + '/' + f for f in os.listdir(cover_dir)])
    stego_paths = sorted([stego_dir + '/' + f for f in os.listdir(stego_dir)])

    # Load the dataset using ImageFolder
    dataset = PairDataset(stego_paths, cover_paths, transform=TRAIN_TF)
    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))  # 90% for training
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))  # Seed set for reproducibility

    # Apply different transforms to training and validation sets (post splitting)
    train_dataset.dataset.transform = TRAIN_TF
    valid_dataset.dataset.transform = VALID_TF

    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def prepare_test(data_dir: str, batch_size: int) -> DataLoader:
    """
    Prepare DataLoader for the test dataset.
    
    :param data_dir: Directory path containing the test data.
    :param batch_size: Size of each batch.
    :return: Test DataLoader object.
    """
    test_dataset = datasets.ImageFolder(root=data_dir, transform=VALID_TF)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def prepare_test_autoencoder(data_dir: str, batch_size: int) -> DataLoader:
    """
    Prepare DataLoader for the test dataset.
    
    :param data_dir: Directory path containing the test data.
    :param batch_size: Size of each batch.
    :return: Test DataLoader object.
    """
    cover_dir = data_dir + '/cover'
    stego_dir = data_dir + '/stego'

    cover_paths = [cover_dir + '/' + f for f in os.listdir(cover_dir)]
    stego_paths = [stego_dir + '/' + f for f in os.listdir(stego_dir)]

    # Load the dataset using ImageFolder
    dataset = PairDataset(stego_paths, cover_paths, transform=VALID_TF)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader
