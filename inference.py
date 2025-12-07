"""
Inference script for single image prediction
Loads trained model and predicts class for a single Fashion-MNIST image
"""
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.classifier import SelfSupervisedClassifier
from utils.data_loader import FASHION_MNIST_CLASSES


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config['model']['num_classes']
        feature_dim = config['model']['feature_dim']
    else:
        num_classes = 10
        feature_dim = 256

    # Create and load model
    model = SelfSupervisedClassifier(
        num_classes=num_classes,
        feature_dim=feature_dim
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get cluster-to-class mapping if available
    mapping = None
    if 'final_results' in checkpoint and 'mapping' in checkpoint['final_results']:
        mapping = checkpoint['final_results']['mapping']

    return model, mapping


def preprocess_image(image_path):
    """
    Load and preprocess image for inference

    Args:
        image_path: Path to image file (28x28 grayscale or will be converted)

    Returns:
        image_tensor: (1, 1, 28, 28) preprocessed tensor
    """
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to 28x28 if needed
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.BILINEAR)

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict(model, image_tensor, device, mapping=None):
    """
    Make prediction for single image

    Args:
        model: SelfSupervisedClassifier
        image_tensor: (1, 1, 28, 28) preprocessed image
        device: cuda or cpu
        mapping: Optional cluster-to-class mapping

    Returns:
        predicted_class: Predicted class index
        predicted_name: Predicted class name
        probabilities: Class probabilities (10,)
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions, probabilities = model.predict(image_tensor)

    predicted_cluster = predictions.item()

    # Apply mapping if available
    if mapping is not None:
        predicted_class = mapping[predicted_cluster]
    else:
        predicted_class = predicted_cluster

    predicted_name = FASHION_MNIST_CLASSES[predicted_class]
    probabilities = probabilities.cpu().numpy()[0]

    return predicted_class, predicted_name, probabilities


def main():
    parser = argparse.ArgumentParser(description='Predict class for Fashion-MNIST image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top K predictions')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, mapping = load_model(args.checkpoint, device)
    print("✓ Model loaded\n")

    # Load and preprocess image
    print(f"Loading image from {args.image}...")
    image_tensor = preprocess_image(args.image)
    print(f"✓ Image loaded: shape {image_tensor.shape}\n")

    # Predict
    print("Making prediction...")
    predicted_class, predicted_name, probabilities = predict(
        model, image_tensor, device, mapping
    )

    # Display results
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Predicted Name:  {predicted_name}")
    print(f"Confidence:      {probabilities[predicted_class]:.2%}")

    # Top-K predictions
    print(f"\nTop {args.top_k} Predictions:")
    top_k_indices = np.argsort(probabilities)[-args.top_k:][::-1]
    for rank, idx in enumerate(top_k_indices, 1):
        print(f"  {rank}. {FASHION_MNIST_CLASSES[idx]:15s} - {probabilities[idx]:.2%}")

    print("="*60 + "\n")


if __name__ == '__main__':
    main()
