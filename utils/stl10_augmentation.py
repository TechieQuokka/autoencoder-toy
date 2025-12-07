"""
STL-10 Specific Augmentation Strategy

For 96x96 RGB natural images (airplane, bird, car, cat, etc.)
Different from Fashion-MNIST - these are natural scenes that can handle
stronger geometric augmentations
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class GaussianBlur:
    """Gaussian blur with probability"""
    def __init__(self, kernel_size=9, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return TF.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img


class RandomNoise:
    """Add random Gaussian noise"""
    def __init__(self, std=0.02, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, -1, 1)
        return img


class STL10Augmentation:
    """
    STL-10 optimized augmentation for 96×96 RGB images

    Natural images can handle stronger augmentations than Fashion-MNIST:
    - Random crops and resizing
    - Color jittering (RGB advantage!)
    - Random horizontal flips (OK for natural scenes)
    - Rotations and affine transforms
    - Gaussian blur
    - Random erasing
    """
    def __init__(self, strength='medium', image_size=96):
        """
        Args:
            strength: 'weak', 'medium', 'strong'
            image_size: Output image size (default: 96 for STL-10)
        """
        self.image_size = image_size

        if strength == 'strong':
            # Strong augmentation for contrastive learning
            self.transform = T.Compose([
                # Random resized crop (SimCLR style)
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                # Horizontal flip is OK for natural images
                T.RandomHorizontalFlip(p=0.5),
                # Color jittering - critical for RGB!
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # Random grayscale
                T.RandomGrayscale(p=0.2),
                # Geometric transforms
                T.RandomRotation(degrees=15),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Blur and erasing
                GaussianBlur(kernel_size=9, p=0.5),
                T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
                RandomNoise(std=0.02, p=0.2),
            ])
        elif strength == 'medium':
            # Balanced augmentation
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                GaussianBlur(kernel_size=5, p=0.3),
                T.RandomErasing(p=0.15),
            ])
        else:  # weak
            # Minimal augmentation
            self.transform = T.Compose([
                T.RandomCrop(image_size, padding=8),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, x):
        return self.transform(x)


class MinimalAugmentation:
    """Minimal augmentation for testing (no data augmentation)"""
    def __init__(self, image_size=96):
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x)


class TwoViewTransform:
    """
    Generate two augmented views for contrastive learning
    Both views use STL-10 appropriate augmentations
    """
    def __init__(self, strength='strong', image_size=96):
        self.transform = STL10Augmentation(strength=strength, image_size=image_size)

    def __call__(self, x):
        """Return two different augmented views"""
        return self.transform(x), self.transform(x)


def get_contrastive_transforms(strength='strong', image_size=96):
    """
    Get transforms for contrastive learning

    Args:
        strength: 'weak', 'medium', 'strong'
        image_size: Image size (default: 96 for STL-10)

    Returns:
        transform1, transform2: Two augmentation pipelines
    """
    transform1 = STL10Augmentation(strength=strength, image_size=image_size)
    transform2 = STL10Augmentation(strength=strength, image_size=image_size)
    return transform1, transform2


def get_train_transform(strength='medium', image_size=96):
    """
    Get training transform

    Args:
        strength: 'weak', 'medium', 'strong'
        image_size: Image size

    Returns:
        transform: Augmentation pipeline
    """
    return STL10Augmentation(strength=strength, image_size=image_size)


def get_test_transform(image_size=96):
    """Get test transform (no augmentation)"""
    return MinimalAugmentation(image_size=image_size)


if __name__ == '__main__':
    # Test augmentations
    print("Testing STL-10 Augmentation...")

    from PIL import Image
    import numpy as np

    # Create fake STL-10 image (96x96 RGB)
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
        mode='RGB'
    )

    # Test different strengths
    for strength in ['weak', 'medium', 'strong']:
        aug = STL10Augmentation(strength=strength)
        augmented = aug(fake_img)
        print(f"{strength.capitalize():6s} augmentation: {augmented.shape}, "
              f"range=[{augmented.min():.2f}, {augmented.max():.2f}]")

    # Test two-view transform
    two_view = TwoViewTransform(strength='strong')
    view1, view2 = two_view(fake_img)
    print(f"\nTwo-view transform:")
    print(f"  View1: {view1.shape}")
    print(f"  View2: {view2.shape}")
    print(f"  Different views: {not torch.equal(view1, view2)}")

    # Test contrastive transforms
    t1, t2 = get_contrastive_transforms(strength='strong')
    aug1 = t1(fake_img)
    aug2 = t2(fake_img)
    print(f"\nContrastive transforms:")
    print(f"  Aug1: {aug1.shape}")
    print(f"  Aug2: {aug2.shape}")

    print("\n✓ STL-10 augmentation test passed!")
