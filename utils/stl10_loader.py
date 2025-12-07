"""
STL-10 Dataset Loader for Self-Supervised Learning
Handles 96x96 RGB images with contrastive learning support
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class STL10Dataset(Dataset):
    """
    STL-10 Dataset wrapper

    Loads images from directory structure and optionally applies transforms
    """
    def __init__(self, root_dir, split='train', transform=None, return_labels=False):
        """
        Args:
            root_dir: Root directory containing train_images, test_images, unlabeled_images
            split: 'train', 'test', or 'unlabeled'
            transform: Optional transform to be applied
            return_labels: Whether to return labels (for supervised evaluation)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_labels = return_labels
        self.split = split

        # Set image directory based on split
        if split == 'train':
            self.image_dir = self.root_dir / 'train_images'
        elif split == 'test':
            self.image_dir = self.root_dir / 'test_images'
        elif split == 'unlabeled':
            self.image_dir = self.root_dir / 'unlabeled_images'
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'unlabeled'")

        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.glob('*.png')))

        # Load labels if needed (STL-10 labels are encoded in filenames)
        if return_labels and split != 'unlabeled':
            self.labels = self._load_labels()
        else:
            self.labels = None

    def _load_labels(self):
        """
        Load labels from separate label files or parse from metadata
        For this Kaggle version, we'll need to load from the binary files
        """
        # For simplicity, we'll return dummy labels for now
        # In practice, you'd load from the .bin label files
        return torch.zeros(len(self.image_paths), dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Return with or without labels
        if self.return_labels and self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, -1  # Dummy label for unsupervised


class ContrastiveSTL10Dataset(Dataset):
    """
    STL-10 dataset wrapper for contrastive learning
    Returns two augmented views of the same image
    """
    def __init__(self, root_dir, split='unlabeled', transform1=None, transform2=None):
        """
        Args:
            root_dir: Root directory containing STL-10 data
            split: 'train', 'test', or 'unlabeled'
            transform1: First view transform
            transform2: Second view transform
        """
        self.root_dir = Path(root_dir)
        self.split = split

        # Set image directory
        if split == 'train':
            self.image_dir = self.root_dir / 'train_images'
        elif split == 'test':
            self.image_dir = self.root_dir / 'test_images'
        elif split == 'unlabeled':
            self.image_dir = self.root_dir / 'unlabeled_images'
        else:
            raise ValueError(f"Invalid split: {split}")

        self.image_paths = sorted(list(self.image_dir.glob('*.png')))

        # Import augmentation
        from .stl10_augmentation import STL10Augmentation
        self.transform1 = transform1 or STL10Augmentation('medium', image_size=96)
        self.transform2 = transform2 or STL10Augmentation('medium', image_size=96)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Create two augmented views
        view1 = self.transform1(image)
        view2 = self.transform2(image)

        return view1, view2


def get_stl10_loaders(data_dir, batch_size=128, num_workers=4,
                      augmentation='medium', use_unlabeled=True):
    """
    Get STL-10 train and test loaders

    Args:
        data_dir: Root directory containing STL10 folder
        batch_size: Batch size
        num_workers: Number of worker processes
        augmentation: Augmentation strength ('weak', 'medium', 'strong')
        use_unlabeled: Whether to use unlabeled data for training

    Returns:
        train_loader, test_loader
    """
    from .stl10_augmentation import get_train_transform, get_test_transform

    stl10_dir = Path(data_dir) / 'STL10'

    # Transforms
    train_transform = get_train_transform(strength=augmentation)
    test_transform = get_test_transform()

    # Datasets
    if use_unlabeled:
        train_dataset = STL10Dataset(
            stl10_dir,
            split='unlabeled',
            transform=train_transform,
            return_labels=False
        )
    else:
        train_dataset = STL10Dataset(
            stl10_dir,
            split='train',
            transform=train_transform,
            return_labels=False
        )

    test_dataset = STL10Dataset(
        stl10_dir,
        split='test',
        transform=test_transform,
        return_labels=True
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_stl10_contrastive_loader(data_dir, batch_size=128, num_workers=4):
    """
    Get STL-10 contrastive learning loader (unlabeled data)

    Returns:
        contrastive_loader: DataLoader for contrastive pretraining
    """
    from .stl10_augmentation import get_contrastive_transforms

    stl10_dir = Path(data_dir) / 'STL10'

    # Get contrastive transforms
    transform1, transform2 = get_contrastive_transforms(strength='medium')

    # Create contrastive dataset using unlabeled data
    contrastive_dataset = ContrastiveSTL10Dataset(
        stl10_dir,
        split='unlabeled',  # Use 100k unlabeled images
        transform1=transform1,
        transform2=transform2
    )

    # Create loader
    contrastive_loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning
    )

    return contrastive_loader


if __name__ == '__main__':
    # Test the data loader
    print("Testing STL-10 Data Loader...")

    data_dir = './dataset'

    # Test regular loaders
    print("\n1. Testing regular loaders...")
    train_loader, test_loader = get_stl10_loaders(
        data_dir,
        batch_size=32,
        num_workers=2,
        use_unlabeled=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}")
    print(f"Train batch range: [{images.min():.2f}, {images.max():.2f}]")

    images, labels = next(iter(test_loader))
    print(f"Test batch shape: {images.shape}")

    # Test contrastive loader
    print("\n2. Testing contrastive loader...")
    contrastive_loader = get_stl10_contrastive_loader(
        data_dir,
        batch_size=32,
        num_workers=2
    )

    print(f"Contrastive batches: {len(contrastive_loader)}")

    view1, view2 = next(iter(contrastive_loader))
    print(f"View1 shape: {view1.shape}")
    print(f"View2 shape: {view2.shape}")
    print(f"Views are different: {not torch.equal(view1, view2)}")

    print("\n✓ STL-10 data loader test passed!")
