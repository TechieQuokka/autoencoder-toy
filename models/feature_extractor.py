"""
Feature Extractor: CNN-based feature extraction for Fashion-MNIST
Outputs L2-normalized 256-dimensional feature vectors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for Fashion-MNIST

    Architecture:
        - 3 convolutional blocks with progressive depth (32 → 64 → 128)
        - BatchNorm for training stability
        - MaxPooling for spatial dimension reduction
        - Fully connected layers for feature projection
        - L2 normalization for stable clustering

    Input: (B, 1, 28, 28) grayscale images
    Output: (B, 256) L2-normalized feature vectors
    """

    def __init__(self, feature_dim=256):
        super(FeatureExtractor, self).__init__()

        self.feature_dim = feature_dim

        # Convolutional backbone
        # Block 1: 28×28 → 14×14, channels: 1 → 32
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28×28 → 14×14
        )

        # Block 2: 14×14 → 7×7, channels: 32 → 64
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14×14 → 7×7
        )

        # Block 3: 7×7 → 3×3, channels: 64 → 128
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7×7 → 3×3
        )

        # Feature projection head
        # 128 channels × 3×3 spatial = 1152 dimensional
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        """
        Forward pass through feature extractor

        Args:
            x: Input images (B, 1, 28, 28)

        Returns:
            features: L2-normalized features (B, feature_dim)
        """
        # Convolutional feature extraction
        x = self.conv_block1(x)  # (B, 32, 14, 14)
        x = self.conv_block2(x)  # (B, 64, 7, 7)
        x = self.conv_block3(x)  # (B, 128, 3, 3)

        # Feature projection
        features = self.fc_layers(x)  # (B, feature_dim)

        # L2 normalization - CRITICAL for stable clustering
        # Prevents feature magnitude from dominating distance calculations
        features = F.normalize(features, p=2, dim=1)

        return features


if __name__ == '__main__':
    # Test feature extractor
    print("Testing Feature Extractor...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = FeatureExtractor(feature_dim=256).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    print(f"\nInput shape: {test_input.shape}")

    with torch.no_grad():
        features = model(test_input)

    print(f"Output shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Feature norm (should be ~1.0): {features.norm(dim=1).mean():.4f}")

    # Verify L2 normalization
    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Features are not L2-normalized!"

    print("\nFeature Extractor test passed!")
