"""
Feature Extractor for STL-10: CNN-based feature extraction
Designed for 96×96 RGB images

Input: (B, 3, 96, 96) RGB images
Output: (B, 512) L2-normalized feature vectors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STL10Encoder(nn.Module):
    """
    CNN-based feature extractor for STL-10

    Architecture optimized for 96×96 RGB images:
        - 4 convolutional blocks with progressive depth (64 → 128 → 256 → 512)
        - BatchNorm for training stability
        - MaxPooling for spatial dimension reduction
        - Global average pooling before final features
        - L2 normalization for stable clustering

    Input: (B, 3, 96, 96) RGB images
    Output: (B, feature_dim) L2-normalized feature vectors
    """

    def __init__(self, feature_dim=512):
        super(STL10Encoder, self).__init__()

        self.feature_dim = feature_dim

        # Block 1: 96×96 → 48×48, channels: 3 → 64
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 96×96 → 48×48
        )

        # Block 2: 48×48 → 24×24, channels: 64 → 128
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 48×48 → 24×24
        )

        # Block 3: 24×24 → 12×12, channels: 128 → 256
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 24×24 → 12×12
        )

        # Block 4: 12×12 → 6×6, channels: 256 → 512
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12×12 → 6×6
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection head (if feature_dim != 512)
        if feature_dim != 512:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, feature_dim)
            )
        else:
            self.fc = nn.Flatten()

    def forward(self, x):
        """
        Forward pass through feature extractor

        Args:
            x: Input images (B, 3, 96, 96)

        Returns:
            features: L2-normalized features (B, feature_dim)
        """
        # Convolutional feature extraction
        x = self.conv_block1(x)  # (B, 64, 48, 48)
        x = self.conv_block2(x)  # (B, 128, 24, 24)
        x = self.conv_block3(x)  # (B, 256, 12, 12)
        x = self.conv_block4(x)  # (B, 512, 6, 6)

        # Global average pooling
        x = self.gap(x)  # (B, 512, 1, 1)

        # Feature projection
        features = self.fc(x)  # (B, feature_dim)

        # L2 normalization - CRITICAL for stable clustering
        features = F.normalize(features, p=2, dim=1)

        return features


if __name__ == '__main__':
    # Test feature extractor
    print("Testing STL-10 Feature Extractor...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = STL10Encoder(feature_dim=512).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 3, 96, 96).to(device)
    print(f"\nInput shape: {test_input.shape}")

    with torch.no_grad():
        features = model(test_input)

    print(f"Output shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Feature norm (should be ~1.0): {features.norm(dim=1).mean():.4f}")

    # Verify L2 normalization
    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Features are not L2-normalized!"

    print("\n✓ STL-10 Feature Extractor test passed!")
