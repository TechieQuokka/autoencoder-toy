"""
Data loading utilities for Fashion-MNIST dataset
Enhanced with contrastive learning support
Updated to use Fashion-MNIST optimized augmentations (v2)
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from .augmentation_v2 import (
    FashionMNISTAugmentation, MinimalAugmentation,
    get_contrastive_transforms, get_train_transform, get_test_transform
)


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


class ContrastiveDataset(Dataset):
    """
    대조 학습을 위한 데이터셋 래퍼

    같은 이미지에서 두 개의 다른 augmented view를 생성
    """
    def __init__(self, dataset, transform1=None, transform2=None):
        """
        Args:
            dataset: 원본 데이터셋 (PIL 이미지 반환 필요)
            transform1: 첫 번째 view용 transform
            transform2: 두 번째 view용 transform
        """
        self.dataset = dataset
        self.transform1 = transform1 or FashionMNISTAugmentation('medium')
        self.transform2 = transform2 or FashionMNISTAugmentation('medium')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 원본 PIL 이미지 가져오기
        image, label = self.dataset[idx]

        # 두 개의 다른 augmented view 생성
        view1 = self.transform1(image)
        view2 = self.transform2(image)

        return view1, view2, label


class RawFashionMNIST(Dataset):
    """
    Transform 없이 PIL 이미지를 반환하는 래퍼

    ContrastiveDataset에서 사용
    """
    def __init__(self, data_dir='./dataset', train=True, download=False):
        self.dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            transform=None,  # PIL 이미지 그대로 반환
            download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_fashion_mnist_loaders(data_dir='./dataset', batch_size=128, use_labels=False,
                              num_workers=4, augmentation='strong'):
    """
    Create Fashion-MNIST DataLoaders for training and testing

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for DataLoader
        use_labels: If False, treats data as unlabeled (self-supervised mode)
        num_workers: Number of worker processes for data loading
        augmentation: 'strong', 'medium', 'weak' - augmentation 강도

    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    # 강화된 augmentation 사용
    train_transform = get_train_transform(augmentation)
    test_transform = get_test_transform()

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


def get_contrastive_loader(data_dir='./dataset', batch_size=256, num_workers=4):
    """
    대조 학습을 위한 DataLoader 생성

    Args:
        data_dir: 데이터셋 경로
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수

    Returns:
        contrastive_loader: 두 개의 view를 반환하는 DataLoader
    """
    # Raw 데이터셋 (PIL 이미지)
    raw_dataset = RawFashionMNIST(data_dir=data_dir, train=True, download=False)

    # 대조 학습용 transforms
    transform1, transform2 = get_contrastive_transforms()

    # ContrastiveDataset으로 래핑
    contrastive_dataset = ContrastiveDataset(
        raw_dataset,
        transform1=transform1,
        transform2=transform2
    )

    # DataLoader 생성
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 대조 학습은 배치 크기 일정해야 함
    )

    return contrastive_loader


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
