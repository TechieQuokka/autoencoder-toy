"""
Fashion-MNIST Specific Augmentation Strategy

핵심 인사이트:
- Fashion 아이템은 자연 이미지와 다르게 명확한 방향성이 있음
- 신발, 가방, 부츠는 위아래 방향이 고정
- 옷(셔츠, 드레스)은 자연스러운 세로 방향
- 좌우 반전은 의미론적 의미를 변경
- 큰 회전은 비현실적

Fashion-MNIST에 효과적인 증강:
✅ 작은 랜덤 크롭 (위치 변화)
✅ 아주 작은 회전 (±3-5도, 자연스러운 자세 변화)
✅ 약간의 스케일/줌 (크기 변화)
✅ 가우시안 블러 (초점 변화)
✅ 랜덤 노이즈 (이미징 변화)
✅ 밝기/대비 (조명 변화)
❌ NO 좌우 반전 (의미론적 의미 파괴)
❌ NO 큰 회전 (비현실적 방향)
❌ NO 원근 변환 (실루엣 파괴)
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class GaussianBlur:
    """확률적 가우시안 블러"""
    def __init__(self, kernel_size=3, sigma=(0.1, 1.0), p=0.3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return TF.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img


class RandomNoise:
    """랜덤 가우시안 노이즈 추가"""
    def __init__(self, std=0.02, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, -1, 1)
        return img


class FashionMNISTAugmentation:
    """
    Fashion-MNIST 최적화 증강

    의미론적 방향성을 보존하면서 현실적인 변화 추가:
    - 작은 위치 이동 (random crop)
    - 아주 작은 회전 (최대 ±5도)
    - 약간의 스케일 변화
    - 블러와 노이즈로 견고성
    - NO 좌우 반전, NO 큰 회전
    """
    def __init__(self, strength='medium'):
        """
        Args:
            strength: 'weak', 'medium', 'strong'
        """
        if strength == 'strong':
            # 강하지만 의미론적으로 유효한 증강
            self.transform = T.Compose([
                # 작은 랜덤 크롭과 리사이즈 (위치 변화)
                T.RandomResizedCrop(28, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                # 아주 작은 회전만 (±5도)
                T.RandomRotation(degrees=5, interpolation=T.InterpolationMode.BILINEAR),
                # 미묘한 어파인 변환
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # 작은 이동
                    scale=(0.95, 1.05),      # 아주 작은 스케일 변화
                ),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
                # 견고성을 위한 블러와 노이즈
                GaussianBlur(kernel_size=3, sigma=(0.1, 0.8), p=0.3),
                T.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.5, 2.0)),
                RandomNoise(std=0.02, p=0.2),
            ])
        elif strength == 'medium':
            # 균형 잡힌 증강
            self.transform = T.Compose([
                T.RandomResizedCrop(28, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                T.RandomRotation(degrees=3),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
                GaussianBlur(kernel_size=3, p=0.2),
                RandomNoise(std=0.015, p=0.15),
            ])
        else:  # weak
            # 최소한의 증강
            self.transform = T.Compose([
                T.RandomCrop(28, padding=2),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])

    def __call__(self, x):
        return self.transform(x)


class TwoViewTransform:
    """
    대조 학습을 위한 두 개의 증강된 뷰 생성
    둘 다 Fashion-MNIST에 적합한 증강 사용
    """
    def __init__(self, strength='medium'):
        self.transform = FashionMNISTAugmentation(strength=strength)

    def __call__(self, x):
        """두 개의 다른 증강된 뷰 반환"""
        return self.transform(x), self.transform(x)


class MinimalAugmentation:
    """테스트용 최소 증강"""
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __call__(self, x):
        return self.transform(x)


def get_contrastive_transforms(strength='medium'):
    """
    대조 학습을 위한 변환 쌍 반환

    Args:
        strength: 'weak', 'medium', 'strong'

    Returns:
        transform1, transform2: 두 개의 증강 파이프라인
    """
    # 둘 다 같은 증강 전략을 사용하지만 다른 뷰를 생성
    transform1 = FashionMNISTAugmentation(strength=strength)
    transform2 = FashionMNISTAugmentation(strength=strength)
    return transform1, transform2


def get_train_transform(strength='medium'):
    """
    학습용 변환 반환

    Args:
        strength: 'weak', 'medium', 'strong'

    Returns:
        transform: 증강 파이프라인
    """
    return FashionMNISTAugmentation(strength=strength)


def get_test_transform():
    """테스트용 변환 반환 (증강 없음)"""
    return MinimalAugmentation()


if __name__ == '__main__':
    # 테스트
    print("Fashion-MNIST 증강 테스트...")

    from PIL import Image
    import numpy as np

    # 가짜 Fashion-MNIST 이미지 생성 (28x28 그레이스케일)
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (28, 28), dtype=np.uint8),
        mode='L'
    )

    # 다양한 강도 테스트
    for strength in ['weak', 'medium', 'strong']:
        aug = FashionMNISTAugmentation(strength=strength)
        augmented = aug(fake_img)
        print(f"{strength.capitalize():6s} 증강: {augmented.shape}, "
              f"범위=[{augmented.min():.2f}, {augmented.max():.2f}]")

    # Two-view 변환 테스트
    two_view = TwoViewTransform(strength='medium')
    view1, view2 = two_view(fake_img)
    print(f"\nTwo-view 변환:")
    print(f"  View1: {view1.shape}")
    print(f"  View2: {view2.shape}")
    print(f"  서로 다른 뷰: {not torch.equal(view1, view2)}")

    # 대조 변환 테스트
    t1, t2 = get_contrastive_transforms(strength='strong')
    aug1 = t1(fake_img)
    aug2 = t2(fake_img)
    print(f"\n대조 변환:")
    print(f"  Aug1: {aug1.shape}")
    print(f"  Aug2: {aug2.shape}")

    print("\n✓ Fashion-MNIST 증강 테스트 통과!")
