# STL-10 Self-Supervised Learning Setup

## ✅ 완료된 작업

STL-10 데이터셋에 맞게 전체 파이프라인을 새로 구성했습니다.

### 1. 데이터셋 다운로드 및 구성
```
dataset/
├── FashionMNIST/          # 기존 (사용 안함)
└── STL10/                 # 새로운 데이터셋
    ├── train_images/      (5,000장, 클래스당 500장)
    ├── test_images/       (8,000장, 클래스당 800장)
    └── unlabeled_images/  (100,000장) ← Contrastive learning용!
```

### 2. 새로운 파일들

#### 📦 Data Loading
- **`utils/stl10_loader.py`**: STL-10 전용 데이터 로더
  - `STL10Dataset`: 기본 데이터셋 클래스
  - `ContrastiveSTL10Dataset`: Contrastive learning용
  - `get_stl10_loaders()`: Train/test 로더
  - `get_stl10_contrastive_loader()`: Unlabeled 데이터 로더

#### 🎨 Augmentation
- **`utils/stl10_augmentation.py`**: STL-10 최적화 증강
  - 96×96 RGB 이미지용
  - Color jittering (RGB 장점 활용!)
  - Horizontal flip (자연 이미지에는 OK)
  - RandomResizedCrop (SimCLR 스타일)
  - Strong/Medium/Weak 프리셋

#### 🧠 Model
- **`models/stl10_encoder.py`**: STL-10 전용 인코더
  - Input: (B, 3, 96, 96) RGB
  - Output: (B, 512) L2-normalized features
  - 4 conv blocks: 64 → 128 → 256 → 512 channels
  - Global average pooling
  - 4.7M parameters

#### ⚙️ Configuration
- **`config/stl10_config.yaml`**: STL-10 전용 설정
  - `feature_dim: 512` (FashionMNIST는 256)
  - `image_size: 96`
  - `batch_size: 256`
  - `temperature: 0.5` (표준 SimCLR)
  - `pretrain_epochs: 100`
  - Success criteria: 75% accuracy, 0.70 NMI, 0.80 purity

## 🎯 Fashion-MNIST vs STL-10 비교

| 항목 | Fashion-MNIST | STL-10 |
|------|---------------|--------|
| 이미지 크기 | 28×28 | 96×96 |
| 채널 | 1 (grayscale) | 3 (RGB) |
| 클래스 특성 | 비슷한 옷들 | 명확히 다른 물체들 |
| Unlabeled | 없음 | 100,000장 |
| 증강 기법 | 제한적 (방향성 보존) | 강력 (flip, color jitter) |
| Feature dim | 256 | 512 |
| 예상 성능 | ~40% (어려움) | ~75% (달성 가능) |

## 🚀 사용 방법

### Option 1: 기존 train.py 수정하여 사용
기존 `train.py`를 다음과 같이 수정:

```python
# Import STL-10 components
from utils.stl10_loader import get_stl10_loaders, get_stl10_contrastive_loader
from models.stl10_encoder import STL10Encoder

# Load config
with open('config/stl10_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = STL10Encoder(feature_dim=config['model']['feature_dim'])

# Load data
train_loader, test_loader = get_stl10_loaders(
    data_dir=config['data']['data_dir'],
    batch_size=config['data']['batch_size'],
    use_unlabeled=True  # Use 100k unlabeled images
)

contrastive_loader = get_stl10_contrastive_loader(
    data_dir=config['data']['data_dir'],
    batch_size=config['data']['batch_size']
)
```

### Option 2: 새로운 train_stl10.py 생성
완전히 새로운 학습 스크립트를 만드는 것을 권장합니다.

## 📊 예상 결과

STL-10은 Fashion-MNIST보다 Self-supervised learning에 **훨씬 적합**:

✅ **장점**:
- RGB 색상 정보 → Color jittering 사용 가능
- 96×96 해상도 → 더 많은 특징 학습 가능
- 명확히 구분되는 클래스 (비행기 vs 고양이 vs 배)
- 100k unlabeled 데이터 → Contrastive learning 이상적
- Self-supervised 표준 벤치마크

📈 **현실적 목표**:
- Overall Accuracy: **70-80%** (Fashion-MNIST는 30%)
- NMI: **0.65-0.75** (Fashion-MNIST는 0.26)
- Purity: **0.75-0.85** (Fashion-MNIST는 0.33)
- Per-class F1: **0.60-0.75** (모든 클래스 균등)

## ⚡ 빠른 테스트

```bash
# 1. 데이터 로더 테스트
python -m utils.stl10_loader

# 2. 모델 테스트
python -m models.stl10_encoder

# 3. 증강 테스트
python -m utils.stl10_augmentation
```

모두 통과 확인됨! ✅

## 📝 다음 단계

1. `train.py`를 STL-10용으로 수정하거나
2. 새로운 `train_stl10.py` 생성
3. `python train_stl10.py --config config/stl10_config.yaml` 실행
4. 훨씬 나은 결과 확인!

## 🎉 핵심 개선사항

Fashion-MNIST에서 실패한 이유들이 STL-10에서는 **모두 해결됨**:

| 문제 | Fashion-MNIST | STL-10 해결 |
|------|---------------|-------------|
| 해상도 | 28×28 너무 작음 | 96×96 충분함 |
| 색상 | Grayscale만 | RGB 색상 정보 |
| 유사 클래스 | 티셔츠/셔츠 구분 어려움 | 비행기/고양이 명확 |
| 증강 제약 | 방향 보존 필요 | 자유로운 증강 |
| 데이터 부족 | Unlabeled 없음 | 100k unlabeled |

**결론**: STL-10은 이 프로젝트에 **완벽하게 적합**한 데이터셋입니다! 🎯
