"""
Evaluation script for trained self-supervised classifier
Loads saved model and performs comprehensive evaluation on test set
"""
import os
import argparse
import torch
import numpy as np

from models.classifier import SelfSupervisedClassifier
from utils.data_loader import get_fashion_mnist_loaders, FASHION_MNIST_CLASSES
from utils.metrics import evaluate_clustering


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: cuda or cpu

    Returns:
        model: Loaded SelfSupervisedClassifier
        checkpoint: Full checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config['model']['num_classes']
        feature_dim = config['model']['feature_dim']
    else:
        # Default values
        num_classes = 10
        feature_dim = 256

    # Create model
    model = SelfSupervisedClassifier(
        num_classes=num_classes,
        feature_dim=feature_dim
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def evaluate_model(model, test_loader, device, n_classes=10):
    """
    Comprehensive evaluation on test set

    Args:
        model: SelfSupervisedClassifier
        test_loader: Test DataLoader
        device: cuda or cpu
        n_classes: Number of classes

    Returns:
        results: dict with all metrics
        all_predictions: numpy array of predictions
        all_true_labels: numpy array of ground-truth labels
    """
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Get predictions
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu())
            all_true_labels.append(labels)

    all_predictions = torch.cat(all_predictions).numpy()
    all_true_labels = torch.cat(all_true_labels).numpy()

    # Comprehensive evaluation
    results = evaluate_clustering(
        all_predictions, all_true_labels,
        n_clusters=n_classes, n_classes=n_classes
    )

    return results, all_predictions, all_true_labels


def print_results(results, verbose=True):
    """
    Pretty print evaluation results

    Args:
        results: dict from evaluate_clustering
        verbose: If True, print detailed per-class metrics
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print(f"\nOverall Performance:")
    print(f"  Matched Accuracy:  {results['matched_accuracy']:.4f} ({results['matched_accuracy']:.2%})")
    print(f"  Precision:         {results['precision']:.4f}")
    print(f"  Recall:            {results['recall']:.4f}")
    print(f"  F1 Score:          {results['f1']:.4f}")

    print(f"\nClustering Quality:")
    print(f"  NMI:               {results['nmi']:.4f}")
    print(f"  Purity:            {results['purity']:.4f}")

    if verbose:
        print(f"\nCluster-to-Class Mapping:")
        for cluster, cls in sorted(results['mapping'].items()):
            print(f"  Cluster {cluster} → Class {cls}: {FASHION_MNIST_CLASSES[cls]}")

        print(f"\nPer-Class F1 Scores:")
        for i, (f1, class_name) in enumerate(zip(results['per_class_f1'], FASHION_MNIST_CLASSES)):
            status = "✓" if f1 > 0.60 else "✗"
            print(f"  {status} Class {i}: {class_name:15s} - F1: {f1:.4f}")

    print(f"\nSuccess Criteria:")
    print(f"  Overall Accuracy > 70%:     {'✓ PASS' if results['matched_accuracy'] > 0.70 else '✗ FAIL'} ({results['matched_accuracy']:.2%})")
    print(f"  NMI > 0.65:                 {'✓ PASS' if results['nmi'] > 0.65 else '✗ FAIL'} ({results['nmi']:.4f})")
    print(f"  Purity > 0.75:              {'✓ PASS' if results['purity'] > 0.75 else '✗ FAIL'} ({results['purity']:.4f})")

    all_classes_good = all(f1 > 0.60 for f1 in results['per_class_f1'])
    print(f"  All Per-Class F1 > 0.60:    {'✓ PASS' if all_classes_good else '✗ FAIL'}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained self-supervised classifier')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to Fashion-MNIST dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-class metrics')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    print(f"Evaluation Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}\n")

    # Load model
    print("Loading model...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model, checkpoint = load_model(args.checkpoint, device)
    print(f"✓ Model loaded from {args.checkpoint}")

    # Load test data
    print("Loading test data...")
    _, test_loader = get_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_labels=False,
        num_workers=4
    )
    print(f"✓ Test batches: {len(test_loader)}")

    # Evaluate
    print("\nEvaluating...")
    results, predictions, true_labels = evaluate_model(model, test_loader, device)

    # Print results
    print_results(results, verbose=args.verbose)

    # Save predictions if requested
    if args.save_predictions:
        pred_path = 'results/predictions.npz'
        os.makedirs('results', exist_ok=True)
        np.savez(
            pred_path,
            predictions=predictions,
            true_labels=true_labels,
            mapping=results['mapping'],
            confusion_matrix=results['confusion_matrix']
        )
        print(f"✓ Predictions saved to {pred_path}\n")


if __name__ == '__main__':
    main()
