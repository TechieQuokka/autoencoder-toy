"""
Clustering utilities for pseudo-label generation
Supports both CPU (scikit-learn) and GPU (FAISS) K-means
"""
import numpy as np
import torch
from sklearn.cluster import KMeans


class KMeansClustering:
    """
    K-means clustering for self-supervised pseudo-label generation

    Supports two backends:
        - sklearn: CPU-based, always available
        - faiss: GPU-accelerated, optional (faster for large datasets)
    """

    def __init__(self, n_clusters=10, device='cuda', use_gpu=True, max_iter=300, n_init=20):
        """
        Args:
            n_clusters: Number of clusters (K)
            device: 'cuda' or 'cpu'
            use_gpu: Attempt to use GPU clustering (FAISS) if available
            max_iter: Maximum K-means iterations
            n_init: Number of K-means initializations (sklearn only)
        """
        self.n_clusters = n_clusters
        self.device = device
        self.max_iter = max_iter
        self.n_init = n_init
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.backend = None
        self.centroids = None

        # Try to initialize GPU clustering
        if self.use_gpu:
            try:
                import faiss
                self.backend = 'faiss'
                print(f"Using FAISS GPU K-means clustering")
            except ImportError:
                print("FAISS not available, falling back to sklearn")
                self.use_gpu = False
                self.backend = 'sklearn'
        else:
            self.backend = 'sklearn'

        # Initialize sklearn K-means
        if self.backend == 'sklearn':
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=42
            )
            print(f"Using scikit-learn CPU K-means clustering")

    def fit_predict(self, features):
        """
        Perform K-means clustering and return pseudo-labels

        Args:
            features: (N, feature_dim) numpy array or torch tensor

        Returns:
            pseudo_labels: (N,) cluster assignments [0, K-1]
            centroids: (K, feature_dim) cluster centers
        """
        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.backend == 'faiss':
            pseudo_labels, centroids = self._fit_predict_faiss(features)
        else:
            pseudo_labels, centroids = self._fit_predict_sklearn(features)

        self.centroids = centroids
        return pseudo_labels, centroids

    def _fit_predict_sklearn(self, features):
        """CPU K-means using scikit-learn"""
        pseudo_labels = self.kmeans.fit_predict(features)
        centroids = self.kmeans.cluster_centers_
        return pseudo_labels, centroids

    def _fit_predict_faiss(self, features):
        """GPU K-means using FAISS"""
        import faiss

        # Ensure float32
        features = features.astype('float32')
        feature_dim = features.shape[1]

        # Create FAISS K-means
        kmeans = faiss.Kmeans(
            d=feature_dim,
            k=self.n_clusters,
            niter=self.max_iter,
            verbose=False,
            gpu=True
        )

        # Train K-means
        kmeans.train(features)

        # Get cluster assignments
        _, pseudo_labels = kmeans.index.search(features, 1)
        pseudo_labels = pseudo_labels.squeeze()

        # Get centroids
        centroids = kmeans.centroids

        return pseudo_labels, centroids

    def compute_stability(self, old_labels, new_labels):
        """
        Compute stability between two label assignments

        Stability = fraction of labels that didn't change

        Args:
            old_labels: Previous cluster assignments
            new_labels: Current cluster assignments

        Returns:
            stability: Float in [0, 1], 1.0 = all labels unchanged
        """
        if old_labels is None:
            return 0.0

        # Convert to numpy if needed
        if isinstance(old_labels, torch.Tensor):
            old_labels = old_labels.cpu().numpy()
        if isinstance(new_labels, torch.Tensor):
            new_labels = new_labels.cpu().numpy()

        # Count unchanged labels
        unchanged = (old_labels == new_labels).sum()
        total = len(old_labels)

        stability = unchanged / total
        return stability

    def predict(self, features):
        """
        Assign new features to existing clusters

        Args:
            features: (N, feature_dim) features to cluster

        Returns:
            labels: (N,) cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Must call fit_predict first to establish centroids")

        # Convert to numpy if needed
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.backend == 'faiss':
            # Use FAISS index search
            import faiss
            features = features.astype('float32')
            index = faiss.IndexFlatL2(self.centroids.shape[1])
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            index.add(self.centroids.astype('float32'))
            _, labels = index.search(features, 1)
            labels = labels.squeeze()
        else:
            # Use sklearn predict
            labels = self.kmeans.predict(features)

        return labels


if __name__ == '__main__':
    # Test clustering
    print("Testing K-means clustering...")

    # Create random features
    np.random.seed(42)
    n_samples = 1000
    feature_dim = 256
    features = np.random.randn(n_samples, feature_dim).astype('float32')

    # Normalize features (like in real scenario)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Test clustering
    clustering = KMeansClustering(n_clusters=10, use_gpu=True)

    print(f"\nBackend: {clustering.backend}")
    print(f"Clustering {n_samples} samples with {feature_dim} features...")

    pseudo_labels, centroids = clustering.fit_predict(features)

    print(f"\nPseudo-labels shape: {pseudo_labels.shape}")
    print(f"Centroids shape: {centroids.shape}")
    print(f"Unique labels: {np.unique(pseudo_labels)}")
    print(f"Label distribution: {np.bincount(pseudo_labels)}")

    # Test stability
    print("\nTesting stability...")
    new_features = features + np.random.randn(*features.shape) * 0.01
    new_labels, _ = clustering.fit_predict(new_features)

    stability = clustering.compute_stability(pseudo_labels, new_labels)
    print(f"Stability (should be high for small noise): {stability:.4f}")

    # Test predict
    print("\nTesting predict on new samples...")
    test_features = np.random.randn(100, feature_dim).astype('float32')
    test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
    test_labels = clustering.predict(test_features)
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Test unique labels: {np.unique(test_labels)}")

    print("\nClustering test passed!")
