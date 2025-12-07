"""
Clustering utilities for pseudo-label generation
Supports both CPU (scikit-learn) and GPU (FAISS) K-means
Enhanced with confidence-based filtering
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ConfidenceFilter:
    """
    신뢰도 기반 샘플 필터링

    클러스터 중심과의 거리를 기반으로 신뢰도 계산
    낮은 신뢰도의 샘플은 학습에서 제외하거나 가중치 감소
    """

    def __init__(self, threshold=0.7, soft_filtering=True):
        """
        Args:
            threshold: 신뢰도 임계값 (이하는 필터링/가중치 감소)
            soft_filtering: True면 가중치 사용, False면 하드 필터링
        """
        self.threshold = threshold
        self.soft_filtering = soft_filtering

    def compute_confidence(self, features, centroids, pseudo_labels):
        """
        각 샘플의 클러스터 할당 신뢰도 계산

        Args:
            features: (N, feature_dim) 특징 벡터
            centroids: (K, feature_dim) 클러스터 중심
            pseudo_labels: (N,) 클러스터 할당

        Returns:
            confidence: (N,) 각 샘플의 신뢰도 [0, 1]
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(centroids, torch.Tensor):
            centroids = centroids.cpu().numpy()

        # 모든 클러스터와의 거리 계산
        # distances[i, k] = ||features[i] - centroids[k]||^2
        distances = np.zeros((len(features), len(centroids)))
        for k in range(len(centroids)):
            diff = features - centroids[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)

        # 소프트맥스로 확률 변환 (음의 거리 사용)
        # temperature를 사용해 분포 조절
        temperature = 1.0
        neg_distances = -distances / temperature
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)

        # 할당된 클러스터에 대한 확률이 신뢰도
        confidence = probabilities[np.arange(len(pseudo_labels)), pseudo_labels]

        return confidence

    def get_sample_weights(self, confidence):
        """
        신뢰도 기반 샘플 가중치 계산

        Args:
            confidence: (N,) 신뢰도 배열

        Returns:
            weights: (N,) 샘플 가중치
        """
        if self.soft_filtering:
            # 소프트 필터링: 신뢰도를 가중치로 직접 사용
            # 낮은 신뢰도 샘플도 약간의 가중치 부여
            weights = np.clip(confidence, 0.1, 1.0)
        else:
            # 하드 필터링: 임계값 이상만 사용
            weights = (confidence >= self.threshold).astype(np.float32)

        return weights

    def filter_samples(self, features, pseudo_labels, confidence):
        """
        낮은 신뢰도 샘플 필터링

        Args:
            features: (N, feature_dim) 특징 벡터
            pseudo_labels: (N,) 클러스터 할당
            confidence: (N,) 신뢰도 배열

        Returns:
            filtered_indices: 필터링 통과한 샘플 인덱스
            filtered_features: 필터링된 특징
            filtered_labels: 필터링된 라벨
        """
        mask = confidence >= self.threshold
        filtered_indices = np.where(mask)[0]

        if isinstance(features, torch.Tensor):
            filtered_features = features[mask]
        else:
            filtered_features = features[mask]

        filtered_labels = pseudo_labels[mask]

        return filtered_indices, filtered_features, filtered_labels


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

    def rebalance_clusters(self, features, pseudo_labels, min_ratio=0.5, max_ratio=2.0):
        """
        클러스터 재균형화

        너무 크거나 작은 클러스터의 경계 샘플을 재할당

        Args:
            features: (N, feature_dim) 특징 벡터
            pseudo_labels: (N,) 클러스터 할당
            min_ratio: 평균 대비 최소 크기 비율
            max_ratio: 평균 대비 최대 크기 비율

        Returns:
            rebalanced_labels: (N,) 재균형화된 클러스터 할당
        """
        if self.centroids is None:
            return pseudo_labels

        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        n_samples = len(pseudo_labels)
        avg_size = n_samples / self.n_clusters
        min_size = int(avg_size * min_ratio)
        max_size = int(avg_size * max_ratio)

        # 클러스터별 크기 계산
        cluster_sizes = np.bincount(pseudo_labels, minlength=self.n_clusters)

        # 모든 클러스터와의 거리 계산
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            diff = features - self.centroids[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)

        rebalanced_labels = pseudo_labels.copy()

        # 너무 큰 클러스터에서 샘플 재할당
        for k in range(self.n_clusters):
            if cluster_sizes[k] > max_size:
                # 현재 클러스터의 샘플들
                cluster_mask = pseudo_labels == k
                cluster_indices = np.where(cluster_mask)[0]

                # 클러스터 내 거리 순으로 정렬 (멀수록 경계에 가까움)
                cluster_distances = distances[cluster_mask, k]
                sorted_idx = np.argsort(cluster_distances)[::-1]  # 거리가 먼 것부터

                # 초과분 재할당
                n_to_move = cluster_sizes[k] - max_size
                for i in range(min(n_to_move, len(sorted_idx))):
                    sample_idx = cluster_indices[sorted_idx[i]]

                    # 두 번째로 가까운 클러스터 찾기
                    sample_distances = distances[sample_idx]
                    second_best = np.argsort(sample_distances)[1]

                    # 그 클러스터가 너무 크지 않으면 재할당
                    if cluster_sizes[second_best] < max_size:
                        rebalanced_labels[sample_idx] = second_best
                        cluster_sizes[k] -= 1
                        cluster_sizes[second_best] += 1

        return rebalanced_labels


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
