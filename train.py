"""
Main training script for Self-Supervised Image Classification
Implements iterative pseudo-labeling with feature refinement
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from models.classifier import SelfSupervisedClassifier
from utils.data_loader import get_fashion_mnist_loaders, get_transform_for_features, FASHION_MNIST_CLASSES
from utils.clustering import KMeansClustering
from utils.metrics import evaluate_clustering


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def extract_all_features(model, dataloader, device):
    """
    Extract features for entire dataset

    Args:
        model: SelfSupervisedClassifier
        dataloader: PyTorch DataLoader
        device: cuda or cpu

    Returns:
        all_features: (N, feature_dim) numpy array
        all_true_labels: (N,) numpy array of ground-truth labels
        all_images: (N, 1, 28, 28) numpy array of images
    """
    model.eval()
    all_features = []
    all_true_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)

            # Extract features
            features = model.extract_features(images)

            all_features.append(features.cpu())
            all_true_labels.append(labels)
            all_images.append(images.cpu())

    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0).numpy()
    all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
    all_images = torch.cat(all_images, dim=0)

    return all_features, all_true_labels, all_images


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train classifier for one epoch with pseudo-labels

    Args:
        model: SelfSupervisedClassifier
        dataloader: DataLoader with (images, pseudo_labels)
        criterion: Loss function (CrossEntropyLoss)
        optimizer: PyTorch optimizer
        device: cuda or cpu

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, pseudo_labels in dataloader:
        images = images.to(device)
        pseudo_labels = pseudo_labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, pseudo_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_on_test(model, test_loader, device, n_classes=10):
    """
    Evaluate model on test set with optimal cluster-to-class matching

    Args:
        model: SelfSupervisedClassifier
        test_loader: Test DataLoader
        device: cuda or cpu
        n_classes: Number of classes

    Returns:
        results: dict with evaluation metrics
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

    # Comprehensive evaluation with Hungarian matching
    results = evaluate_clustering(all_predictions, all_true_labels, n_clusters=n_classes, n_classes=n_classes)

    return results


def main():
    """Main training loop"""
    # Load configuration
    config = load_config('config/config.yaml')

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Setup device
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}\n")

    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)

    # Load data
    print("Loading Fashion-MNIST dataset...")
    train_loader, test_loader = get_fashion_mnist_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        use_labels=False,  # Self-supervised mode
        num_workers=config['data']['num_workers']
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}\n")

    # Initialize model
    print("Initializing model...")
    model = SelfSupervisedClassifier(
        num_classes=config['model']['num_classes'],
        feature_dim=config['model']['feature_dim']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Initialize optimizer and loss
    criterion = nn.CrossEntropyLoss()

    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )

    # Initialize clustering
    clustering = KMeansClustering(
        n_clusters=config['clustering']['n_clusters'],
        device=device,
        use_gpu=config['clustering']['use_gpu_clustering'],
        max_iter=config['clustering']['max_iter'],
        n_init=config['clustering']['n_init']
    )

    # Tracking metrics across iterations
    history = {
        'iteration': [],
        'nmi': [],
        'purity': [],
        'matched_accuracy': [],
        'test_f1': [],
        'label_stability': [],
        'train_loss': []
    }

    prev_pseudo_labels = None
    best_accuracy = 0.0

    print("="*80)
    print("Starting Iterative Self-Supervised Training")
    print("="*80)

    # Main Iterative Loop
    for iteration in range(config['training']['num_iterations']):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config['training']['num_iterations']}")
        print(f"{'='*80}\n")

        # ===== STEP 1: Feature Extraction =====
        print("Step 1: Extracting features from training set...")
        all_features, all_true_labels, all_images = extract_all_features(
            model, train_loader, device
        )
        print(f"  Extracted features shape: {all_features.shape}")

        # ===== STEP 2: K-means Clustering =====
        print("\nStep 2: Performing K-means clustering...")
        pseudo_labels, centroids = clustering.fit_predict(all_features)
        print(f"  Pseudo-labels shape: {pseudo_labels.shape}")
        print(f"  Unique clusters: {np.unique(pseudo_labels)}")
        print(f"  Cluster sizes: {np.bincount(pseudo_labels)}")

        # Evaluate clustering quality
        clustering_results = evaluate_clustering(
            pseudo_labels, all_true_labels,
            n_clusters=config['model']['num_classes'],
            n_classes=config['model']['num_classes']
        )

        print(f"\n  Clustering Quality:")
        print(f"    NMI: {clustering_results['nmi']:.4f}")
        print(f"    Purity: {clustering_results['purity']:.4f}")
        print(f"    Matched Accuracy: {clustering_results['matched_accuracy']:.4f}")

        # Check label stability (convergence criterion)
        if prev_pseudo_labels is not None:
            stability = clustering.compute_stability(prev_pseudo_labels, pseudo_labels)
            print(f"    Label Stability: {stability:.4f}")
            history['label_stability'].append(stability)

            # Check for convergence
            if config['convergence']['early_stopping'] and stability > config['convergence']['label_stability_threshold']:
                print(f"\n  ✓ Converged! Stability {stability:.4f} > {config['convergence']['label_stability_threshold']}")
                print("  Stopping early as pseudo-labels have stabilized.")
                break
        else:
            history['label_stability'].append(0.0)

        prev_pseudo_labels = pseudo_labels.copy()

        # ===== STEP 3: Train Classifier with Pseudo-labels =====
        print(f"\nStep 3: Training classifier for {config['training']['epochs_per_iteration']} epochs...")

        # Create DataLoader with pseudo-labels
        # Use original images, not features, for better learning
        train_dataset_labeled = TensorDataset(
            all_images,
            torch.from_numpy(pseudo_labels).long()
        )

        labeled_loader = DataLoader(
            train_dataset_labeled,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=0,  # Already in memory
            pin_memory=True
        )

        epoch_losses = []
        for epoch in range(config['training']['epochs_per_iteration']):
            avg_loss = train_one_epoch(model, labeled_loader, criterion, optimizer, device)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{config['training']['epochs_per_iteration']}], Loss: {avg_loss:.4f}")

        # ===== STEP 4: Evaluate on Test Set =====
        if config['evaluation']['eval_every_iteration']:
            print("\nStep 4: Evaluating on test set...")
            test_results = evaluate_on_test(model, test_loader, device, config['model']['num_classes'])

            print(f"  Test Performance:")
            print(f"    Matched Accuracy: {test_results['matched_accuracy']:.4f}")
            print(f"    NMI: {test_results['nmi']:.4f}")
            print(f"    Purity: {test_results['purity']:.4f}")
            print(f"    F1 Score: {test_results['f1']:.4f}")

            # Save best model
            if config['evaluation']['save_best_model'] and test_results['matched_accuracy'] > best_accuracy:
                best_accuracy = test_results['matched_accuracy']
                best_model_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'mapping': test_results['mapping']
                }, best_model_path)
                print(f"  ✓ New best model saved! Accuracy: {best_accuracy:.4f}")

            # Record history
            history['iteration'].append(iteration + 1)
            history['nmi'].append(clustering_results['nmi'])
            history['purity'].append(clustering_results['purity'])
            history['matched_accuracy'].append(test_results['matched_accuracy'])
            history['test_f1'].append(test_results['f1'])
            history['train_loss'].append(np.mean(epoch_losses))

        # Save checkpoint
        checkpoint_path = os.path.join(
            config['paths']['checkpoint_dir'],
            f'model_iter_{iteration + 1}.pth'
        )
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'pseudo_labels': pseudo_labels,
            'centroids': centroids,
            'history': history
        }, checkpoint_path)

    # ===== Final Evaluation =====
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)

    final_results = evaluate_on_test(model, test_loader, device, config['model']['num_classes'])

    print(f"\nFinal Performance:")
    print(f"  Matched Accuracy: {final_results['matched_accuracy']:.4f}")
    print(f"  NMI: {final_results['nmi']:.4f}")
    print(f"  Purity: {final_results['purity']:.4f}")
    print(f"  Precision: {final_results['precision']:.4f}")
    print(f"  Recall: {final_results['recall']:.4f}")
    print(f"  F1 Score: {final_results['f1']:.4f}")

    print(f"\nPer-class F1 scores:")
    for i, (f1, class_name) in enumerate(zip(final_results['per_class_f1'], FASHION_MNIST_CLASSES)):
        print(f"  {i}: {class_name:15s} - F1: {f1:.4f}")

    # Save final model
    final_model_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_results': final_results,
        'history': history,
        'config': config
    }, final_model_path)

    # Save history
    history_path = os.path.join(config['paths']['results_dir'], 'training_history.npy')
    np.save(history_path, history)

    print(f"\n✓ Training completed!")
    print(f"  Best accuracy achieved: {best_accuracy:.4f}")
    print(f"  Final model saved to: {final_model_path}")
    print(f"  Training history saved to: {history_path}")

    # Print success criteria check
    print(f"\n{'='*80}")
    print("Success Criteria Check:")
    print(f"{'='*80}")
    print(f"  Overall Accuracy > 70%:     {'✓' if final_results['matched_accuracy'] > 0.70 else '✗'} ({final_results['matched_accuracy']:.2%})")
    print(f"  NMI > 0.65:                 {'✓' if final_results['nmi'] > 0.65 else '✗'} ({final_results['nmi']:.4f})")
    print(f"  Purity > 0.75:              {'✓' if final_results['purity'] > 0.75 else '✗'} ({final_results['purity']:.4f})")

    all_classes_good = all(f1 > 0.60 for f1 in final_results['per_class_f1'])
    print(f"  All Per-Class F1 > 0.60:    {'✓' if all_classes_good else '✗'}")

    return model, history, final_results


if __name__ == '__main__':
    model, history, results = main()
