"""
Evaluation metrics for self-supervised classification
Includes Hungarian matching, NMI, Purity, Accuracy, F1, etc.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def compute_cluster_to_class_mapping(pseudo_labels, true_labels, n_clusters=10, n_classes=10):
    """
    Find optimal cluster-to-class mapping using Hungarian algorithm

    This is CRITICAL for fair evaluation in self-supervised learning:
    - Cluster IDs are arbitrary (0-9) and don't match class IDs
    - We need to find the best permutation that maximizes agreement
    - Uses Hungarian algorithm to solve optimal assignment problem

    Args:
        pseudo_labels: (N,) cluster assignments from model
        true_labels: (N,) ground-truth class labels
        n_clusters: Number of clusters (K)
        n_classes: Number of true classes

    Returns:
        mapping: dict {cluster_id: class_id} optimal mapping
        matched_accuracy: Accuracy after applying optimal mapping
    """
    # Convert to numpy if needed
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    # Build confusion/cost matrix
    # confusion[i, j] = number of samples in cluster i that belong to class j
    confusion = np.zeros((n_clusters, n_classes), dtype=np.int64)

    for cluster_id in range(n_clusters):
        cluster_mask = (pseudo_labels == cluster_id)
        if cluster_mask.sum() == 0:
            continue
        cluster_true_labels = true_labels[cluster_mask]

        for class_id in range(n_classes):
            confusion[cluster_id, class_id] = (cluster_true_labels == class_id).sum()

    # Hungarian algorithm finds optimal assignment
    # We maximize agreement, so minimize negative of confusion matrix
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Create mapping dictionary
    mapping = {cluster: cls for cluster, cls in zip(row_ind, col_ind)}

    # Handle unmapped clusters (if n_clusters != n_classes)
    for cluster_id in range(n_clusters):
        if cluster_id not in mapping:
            mapping[cluster_id] = -1  # Unmapped cluster

    # Compute matched accuracy
    matched_labels = np.array([mapping[c] for c in pseudo_labels])
    matched_accuracy = (matched_labels == true_labels).mean()

    return mapping, matched_accuracy


def compute_nmi(pseudo_labels, true_labels):
    """
    Compute Normalized Mutual Information (NMI)

    NMI measures how much information the clustering provides about true classes
    Range: [0, 1], where 1 = perfect agreement

    Args:
        pseudo_labels: (N,) cluster assignments
        true_labels: (N,) ground-truth labels

    Returns:
        nmi: Normalized Mutual Information score
    """
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    nmi = normalized_mutual_info_score(true_labels, pseudo_labels)
    return nmi


def compute_purity(pseudo_labels, true_labels, n_clusters=10):
    """
    Compute cluster purity

    Purity = average fraction of dominant class in each cluster
    Range: [0, 1], where 1 = each cluster contains only one class

    Args:
        pseudo_labels: (N,) cluster assignments
        true_labels: (N,) ground-truth labels
        n_clusters: Number of clusters

    Returns:
        purity: Cluster purity score
    """
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    total_correct = 0

    for cluster_id in range(n_clusters):
        cluster_mask = (pseudo_labels == cluster_id)
        if cluster_mask.sum() == 0:
            continue

        cluster_true_labels = true_labels[cluster_mask]
        # Most common class in this cluster
        most_common_count = np.bincount(cluster_true_labels).max()
        total_correct += most_common_count

    purity = total_correct / len(true_labels)
    return purity


def compute_classification_metrics(predictions, true_labels, n_classes=10):
    """
    Compute comprehensive classification metrics

    Args:
        predictions: (N,) predicted class labels
        true_labels: (N,) ground-truth labels
        n_classes: Number of classes

    Returns:
        metrics: dict containing:
            - accuracy: Overall accuracy
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - f1: Macro-averaged F1 score
            - per_class_f1: Per-class F1 scores
            - confusion_matrix: (n_classes, n_classes) confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)

    # Precision, Recall, F1 (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions,
        average='macro',
        zero_division=0
    )

    # Per-class F1
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        true_labels, predictions,
        average=None,
        zero_division=0,
        labels=list(range(n_classes))
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(n_classes)))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': conf_matrix
    }

    return metrics


def evaluate_clustering(pseudo_labels, true_labels, n_clusters=10, n_classes=10):
    """
    Comprehensive clustering evaluation

    Combines all metrics for clustering quality assessment

    Args:
        pseudo_labels: (N,) cluster assignments
        true_labels: (N,) ground-truth labels
        n_clusters: Number of clusters
        n_classes: Number of true classes

    Returns:
        results: dict containing all metrics
    """
    # Cluster-to-class mapping and matched accuracy
    mapping, matched_accuracy = compute_cluster_to_class_mapping(
        pseudo_labels, true_labels, n_clusters, n_classes
    )

    # NMI
    nmi = compute_nmi(pseudo_labels, true_labels)

    # Purity
    purity = compute_purity(pseudo_labels, true_labels, n_clusters)

    # Apply mapping to get predictions
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels_np = pseudo_labels.cpu().numpy()
    else:
        pseudo_labels_np = pseudo_labels

    matched_predictions = np.array([mapping[c] for c in pseudo_labels_np])

    # Classification metrics
    clf_metrics = compute_classification_metrics(matched_predictions, true_labels, n_classes)

    results = {
        'mapping': mapping,
        'matched_accuracy': matched_accuracy,
        'nmi': nmi,
        'purity': purity,
        **clf_metrics
    }

    return results


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_clusters = 10
    n_classes = 10

    # Simulate pseudo-labels (shifted by 3 to test Hungarian matching)
    true_labels = np.random.randint(0, n_classes, n_samples)
    pseudo_labels = (true_labels + 3) % n_clusters  # Shifted mapping

    # Add some noise
    noise_mask = np.random.rand(n_samples) < 0.2  # 20% error rate
    pseudo_labels[noise_mask] = np.random.randint(0, n_clusters, noise_mask.sum())

    print(f"\nDataset: {n_samples} samples, {n_classes} classes, {n_clusters} clusters")
    print(f"Simulated error rate: ~20%")

    # Test individual metrics
    print("\n--- Individual Metrics ---")

    # Hungarian matching
    mapping, matched_accuracy = compute_cluster_to_class_mapping(
        pseudo_labels, true_labels, n_clusters, n_classes
    )
    print(f"\nOptimal cluster-to-class mapping:")
    for cluster, cls in sorted(mapping.items()):
        print(f"  Cluster {cluster} → Class {cls}")
    print(f"Matched Accuracy: {matched_accuracy:.4f}")

    # NMI
    nmi = compute_nmi(pseudo_labels, true_labels)
    print(f"\nNMI: {nmi:.4f}")

    # Purity
    purity = compute_purity(pseudo_labels, true_labels, n_clusters)
    print(f"Purity: {purity:.4f}")

    # Full evaluation
    print("\n--- Comprehensive Evaluation ---")
    results = evaluate_clustering(pseudo_labels, true_labels, n_clusters, n_classes)

    print(f"\nMatched Accuracy: {results['matched_accuracy']:.4f}")
    print(f"NMI: {results['nmi']:.4f}")
    print(f"Purity: {results['purity']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")

    print(f"\nPer-class F1 scores:")
    for i, f1 in enumerate(results['per_class_f1']):
        print(f"  Class {i}: {f1:.4f}")

    print(f"\nConfusion matrix shape: {results['confusion_matrix'].shape}")
    print(f"Diagonal sum (correct predictions): {np.diag(results['confusion_matrix']).sum()}")

    # Verify matched accuracy equals sklearn accuracy on mapped labels
    matched_preds = np.array([mapping[c] for c in pseudo_labels])
    sklearn_acc = accuracy_score(true_labels, matched_preds)
    assert np.isclose(matched_accuracy, sklearn_acc), "Accuracy mismatch!"

    print("\nMetrics test passed!")
