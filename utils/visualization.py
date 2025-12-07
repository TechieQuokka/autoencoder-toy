"""
Visualization utilities for training analysis
Plots loss curves, confusion matrices, clustering quality, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_loss_curves(train_losses, save_path='results/loss_curve.png', title='Training Loss'):
    """
    Plot training loss over epochs

    Args:
        train_losses: List or array of loss values
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss curve saved to {save_path}")


def plot_confusion_matrix(conf_matrix, class_names, save_path='results/confusion_matrix.png',
                          title='Confusion Matrix'):
    """
    Visualize confusion matrix with heatmap

    Args:
        conf_matrix: (n_classes, n_classes) confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_clustering_quality(history, save_path='results/clustering_quality.png'):
    """
    Plot clustering quality metrics over iterations

    Args:
        history: dict with 'nmi', 'purity', 'matched_accuracy', 'iteration' keys
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # NMI and Purity
    ax1.plot(history['iteration'], history['nmi'], marker='o', linewidth=2,
             markersize=8, label='NMI', color='#A23B72')
    ax1.plot(history['iteration'], history['purity'], marker='s', linewidth=2,
             markersize=8, label='Purity', color='#F18F01')
    ax1.axhline(y=0.65, color='#A23B72', linestyle='--', alpha=0.5, label='NMI Target (0.65)')
    ax1.axhline(y=0.75, color='#F18F01', linestyle='--', alpha=0.5, label='Purity Target (0.75)')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Clustering Quality Metrics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Accuracy
    ax2.plot(history['iteration'], history['matched_accuracy'], marker='D', linewidth=2,
             markersize=8, label='Test Accuracy', color='#2E86AB')
    ax2.axhline(y=0.70, color='#2E86AB', linestyle='--', alpha=0.5, label='Target (70%)')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Clustering quality plot saved to {save_path}")


def plot_training_progress(history, save_path='results/training_progress.png'):
    """
    Comprehensive training progress visualization

    Args:
        history: dict with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curve
    ax = axes[0, 0]
    ax.plot(history['iteration'], history['train_loss'], marker='o', linewidth=2,
            markersize=6, color='#2E86AB')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training Loss per Iteration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(history['iteration'], history['matched_accuracy'], marker='s', linewidth=2,
            markersize=6, color='#A23B72')
    ax.axhline(y=0.70, color='gray', linestyle='--', alpha=0.5, label='Target (70%)')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Test Accuracy per Iteration', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NMI and Purity
    ax = axes[1, 0]
    ax.plot(history['iteration'], history['nmi'], marker='o', linewidth=2,
            markersize=6, label='NMI', color='#F18F01')
    ax.plot(history['iteration'], history['purity'], marker='^', linewidth=2,
            markersize=6, label='Purity', color='#C73E1D')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Clustering Quality (NMI & Purity)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Label Stability
    ax = axes[1, 1]
    if len(history['label_stability']) > 0:
        # Skip first iteration (always 0)
        stability_iters = history['iteration'][1:]
        stability_values = history['label_stability'][1:]
        ax.plot(stability_iters, stability_values, marker='D', linewidth=2,
                markersize=6, color='#06A77D')
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Convergence (0.95)')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Stability', fontsize=11, fontweight='bold')
    ax.set_title('Pseudo-Label Stability', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training progress plot saved to {save_path}")


def plot_tsne_features(features, labels, save_path='results/tsne_features.png',
                       title='t-SNE Feature Visualization', n_samples=5000):
    """
    t-SNE visualization of feature space

    Args:
        features: (N, feature_dim) feature vectors
        labels: (N,) labels for coloring
        save_path: Path to save figure
        title: Plot title
        n_samples: Number of samples to use (for speed)
    """
    # Subsample if too many samples
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(14, 11))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels, cmap='tab10',
        alpha=0.6, s=10, edgecolors='none'
    )
    plt.colorbar(scatter, label='Class', ticks=range(10))
    plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE visualization saved to {save_path}")


def plot_per_class_metrics(per_class_f1, class_names, save_path='results/per_class_f1.png'):
    """
    Bar plot of per-class F1 scores

    Args:
        per_class_f1: Array of F1 scores per class
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    # Color bars based on threshold
    colors = ['#06A77D' if f1 > 0.60 else '#C73E1D' for f1 in per_class_f1]

    bars = plt.bar(range(len(per_class_f1)), per_class_f1, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(y=0.60, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.60)')

    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class F1 plot saved to {save_path}")


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization functions...")

    # Create dummy data
    np.random.seed(42)

    # Test loss curve
    losses = np.exp(-np.linspace(0, 3, 50)) + np.random.rand(50) * 0.1
    plot_loss_curves(losses, save_path='test_loss.png')

    # Test confusion matrix
    conf_matrix = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(conf_matrix, np.random.randint(200, 300, 10))
    class_names = [f'Class_{i}' for i in range(10)]
    plot_confusion_matrix(conf_matrix, class_names, save_path='test_confusion.png')

    # Test per-class F1
    per_class_f1 = np.random.rand(10) * 0.3 + 0.5
    plot_per_class_metrics(per_class_f1, class_names, save_path='test_per_class_f1.png')

    print("\n✓ All visualization tests passed!")
    print("  Test plots created: test_loss.png, test_confusion.png, test_per_class_f1.png")
