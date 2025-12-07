"""
Contrastive Learning Module for Self-Supervised Classification
Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
Similar to SimCLR approach for learning discriminative features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)

    같은 이미지의 두 augmentation view는 가깝게,
    다른 이미지들은 멀게 학습하는 대조 학습 손실 함수

    Args:
        temperature: 소프트맥스 온도 파라미터 (낮을수록 더 엄격)
        batch_size: 배치 크기
    """

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        대조 손실 계산

        Args:
            z_i: 첫 번째 augmentation view의 특징 (B, feature_dim)
            z_j: 두 번째 augmentation view의 특징 (B, feature_dim)

        Returns:
            loss: NT-Xent 손실값
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # L2 정규화 (이미 정규화되어 있어도 안전하게)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 모든 특징 벡터 결합: [z_i; z_j] = (2B, feature_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # 코사인 유사도 행렬 계산: (2B, 2B)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # 자기 자신과의 유사도는 제외 (대각선)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, j) 쌍과 (j, i) 쌍
        # i번째 샘플의 positive는 batch_size + i 위치에 있음
        pos_i = torch.diag(sim_matrix, batch_size)  # z_i와 z_j의 유사도
        pos_j = torch.diag(sim_matrix, -batch_size)  # z_j와 z_i의 유사도
        positives = torch.cat([pos_i, pos_j], dim=0)  # (2B,)

        # Negative pairs: 자신과 positive pair 제외한 모든 샘플
        # Log-sum-exp 계산을 위해 모든 유사도 사용
        numerator = positives
        denominator = torch.logsumexp(sim_matrix, dim=1)

        # NT-Xent Loss: -log(exp(pos) / sum(exp(all)))
        loss = -numerator + denominator
        loss = loss.mean()

        return loss


class ContrastiveProjector(nn.Module):
    """
    대조 학습을 위한 프로젝션 헤드

    특징 공간에서 대조 학습 공간으로 투영
    학습 후에는 이 헤드를 제거하고 특징만 사용
    """

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super(ContrastiveProjector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: 입력 특징 (B, input_dim)

        Returns:
            projected: 투영된 특징 (B, output_dim)
        """
        return self.projector(x)


class ClusterContrastiveLoss(nn.Module):
    """
    클러스터 기반 대조 손실

    같은 클러스터(pseudo-label)에 속한 샘플들은 가깝게,
    다른 클러스터의 샘플들은 멀게 학습
    """

    def __init__(self, temperature=0.5, num_classes=10):
        super(ClusterContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, features, pseudo_labels):
        """
        클러스터 대조 손실 계산

        Args:
            features: 특징 벡터 (B, feature_dim)
            pseudo_labels: 클러스터 할당 (B,)

        Returns:
            loss: 클러스터 대조 손실
        """
        device = features.device
        batch_size = features.shape[0]

        # L2 정규화
        features = F.normalize(features, dim=1)

        # 유사도 행렬
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # 자기 자신 제외
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask_self, float('-inf'))

        # Positive mask: 같은 pseudo-label을 가진 샘플들
        labels_equal = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        positive_mask = labels_equal & ~mask_self

        # 각 샘플에 대해 positive가 있는 경우만 손실 계산
        has_positive = positive_mask.sum(dim=1) > 0

        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Log-softmax over all negatives and positives
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Positive pairs에 대한 평균 log probability
        positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        positive_count = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(positive_log_prob / positive_count)

        # 유효한 샘플에 대해서만 평균
        loss = loss[has_positive].mean()

        return loss


class CombinedContrastiveLoss(nn.Module):
    """
    결합된 대조 손실

    1. Instance-level: 같은 이미지의 다른 augmentation은 가깝게
    2. Cluster-level: 같은 클러스터의 샘플들은 가깝게
    """

    def __init__(self, temperature=0.5, instance_weight=1.0, cluster_weight=0.5):
        super(CombinedContrastiveLoss, self).__init__()
        self.instance_loss = NTXentLoss(temperature)
        self.cluster_loss = ClusterContrastiveLoss(temperature)
        self.instance_weight = instance_weight
        self.cluster_weight = cluster_weight

    def forward(self, z_i, z_j, features, pseudo_labels):
        """
        결합 손실 계산

        Args:
            z_i: 첫 번째 view의 프로젝션 (B, proj_dim)
            z_j: 두 번째 view의 프로젝션 (B, proj_dim)
            features: 원본 특징 (B, feature_dim)
            pseudo_labels: 클러스터 할당 (B,)

        Returns:
            total_loss: 총 대조 손실
            loss_dict: 개별 손실값 딕셔너리
        """
        inst_loss = self.instance_loss(z_i, z_j)
        clust_loss = self.cluster_loss(features, pseudo_labels)

        total_loss = self.instance_weight * inst_loss + self.cluster_weight * clust_loss

        return total_loss, {
            'instance_loss': inst_loss.item(),
            'cluster_loss': clust_loss.item()
        }


if __name__ == '__main__':
    # 테스트
    print("Testing Contrastive Learning Module...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 32
    feature_dim = 256
    proj_dim = 128

    # 가짜 특징 생성
    z_i = torch.randn(batch_size, feature_dim).to(device)
    z_j = torch.randn(batch_size, feature_dim).to(device)
    pseudo_labels = torch.randint(0, 10, (batch_size,)).to(device)

    # NT-Xent Loss 테스트
    nt_xent = NTXentLoss(temperature=0.5).to(device)
    loss = nt_xent(z_i, z_j)
    print(f"NT-Xent Loss: {loss.item():.4f}")

    # Projector 테스트
    projector = ContrastiveProjector(feature_dim, feature_dim, proj_dim).to(device)
    proj_i = projector(z_i)
    print(f"Projected shape: {proj_i.shape}")

    # Cluster Contrastive Loss 테스트
    cluster_loss = ClusterContrastiveLoss(temperature=0.5).to(device)
    c_loss = cluster_loss(z_i, pseudo_labels)
    print(f"Cluster Contrastive Loss: {c_loss.item():.4f}")

    # Combined Loss 테스트
    combined = CombinedContrastiveLoss(temperature=0.5).to(device)
    total_loss, loss_dict = combined(proj_i, projector(z_j), z_i, pseudo_labels)
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"  Instance: {loss_dict['instance_loss']:.4f}")
    print(f"  Cluster: {loss_dict['cluster_loss']:.4f}")

    print("\nContrastive module test passed!")
