"""
Data loading utilities for Fashion-MNIST dataset
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class UnlabeledWrapper(Dataset):
    """
    Wrapper to hide labels during self-supervised training
    Returns only images without labels
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Return image and label, but label won't be used in training
        # We keep it for evaluation purposes
        return image, label


def get_fashion_mnist_loaders(data_dir='./dataset', batch_size=128, use_labels=False, num_workers=4):
    """
    Create Fashion-MNIST DataLoaders for training and testing

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for DataLoader
        use_labels: If False, treats data as unlabeled (self-supervised mode)
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    # Data augmentation for training (helps learn shape features)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Fashion-MNIST datasets
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=False  # Dataset already exists
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=False
    )

    # Wrap if treating as unlabeled (self-supervised mode)
    if not use_labels:
        train_dataset = UnlabeledWrapper(train_dataset)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, test_loader


def get_transform_for_features():
    """
    Get transform for feature extraction (no augmentation)
    Used during clustering phase
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# Fashion-MNIST class names
FASHION_MNIST_CLASSES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


if __name__ == '__main__':
    # Test data loading
    print("Testing Fashion-MNIST data loading...")
    train_loader, test_loader = get_fashion_mnist_loaders(
        data_dir='./dataset',
        batch_size=128,
        use_labels=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check first batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # Should be [128, 1, 28, 28]
    print(f"Labels shape: {labels.shape}")  # Should be [128]
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")

    print("\nClass names:")
    for i, name in enumerate(FASHION_MNIST_CLASSES):
        print(f"  {i}: {name}")

    print("\nData loading test passed!")
