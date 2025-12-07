"""
Self-Supervised Classifier: Complete classification model
Combines Feature Extractor with Classification Head
"""
import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor


class SelfSupervisedClassifier(nn.Module):
    """
    Complete self-supervised classification model

    Architecture:
        - Feature Extractor (CNN) → 256-dim features
        - Classification Head (FC layers) → 10-class predictions

    The model can operate in two modes:
        1. Training: Extract features + classify with pseudo-labels
        2. Inference: Extract features + classify new images
    """

    def __init__(self, num_classes=10, feature_dim=256, encoder=None):
        super(SelfSupervisedClassifier, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Feature extraction backbone (use provided encoder or default)
        if encoder is not None:
            self.feature_extractor = encoder
        else:
            self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)

        # Classification head
        # Maps features to class logits
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        Forward pass through the model

        Args:
            x: Input images (B, 1, 28, 28)
            return_features: If True, also return feature vectors

        Returns:
            logits: Class predictions (B, num_classes)
            features (optional): Feature vectors (B, feature_dim)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Classify
        logits = self.classifier_head(features)

        if return_features:
            return logits, features

        return logits

    def extract_features(self, x):
        """
        Extract only features (for clustering)

        Args:
            x: Input images (B, 1, 28, 28)

        Returns:
            features: Feature vectors (B, feature_dim)
        """
        return self.feature_extractor(x)

    def predict(self, x):
        """
        Make predictions (argmax of logits)

        Args:
            x: Input images (B, 1, 28, 28)

        Returns:
            predictions: Predicted class indices (B,)
            probabilities: Class probabilities (B, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        return predictions, probabilities


if __name__ == '__main__':
    # Test classifier
    print("Testing Self-Supervised Classifier...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = SelfSupervisedClassifier(num_classes=10, feature_dim=256).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    print(f"\nInput shape: {test_input.shape}")

    # Test standard forward
    with torch.no_grad():
        logits = model(test_input)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 10), "Incorrect logits shape!"

    # Test forward with features
    with torch.no_grad():
        logits, features = model(test_input, return_features=True)
    print(f"Features shape: {features.shape}")
    assert features.shape == (batch_size, 256), "Incorrect features shape!"

    # Test feature extraction only
    with torch.no_grad():
        features_only = model.extract_features(test_input)
    print(f"Features-only shape: {features_only.shape}")
    assert torch.allclose(features, features_only), "Feature mismatch!"

    # Test prediction
    predictions, probabilities = model.predict(test_input)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Probability sum (should be ~1.0): {probabilities.sum(dim=1)[:5]}")

    # Verify probabilities sum to 1
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-5), \
        "Probabilities don't sum to 1!"

    print("\nSelf-Supervised Classifier test passed!")
