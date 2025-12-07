# Self-Supervised Image Classification Architecture

## 목적
Fashion-MNIST 데이터에서 ground-truth label 없이 학습하여, **이미지를 분류하는 모델** 구축

## 최종 목표
**입력**: 패션 아이템 이미지 (28×28)
**출력**: 클래스 예측 (0~9 중 하나)
**평가**: Classification Accuracy (올바른 클래스로 분류했는가)

## 핵심 아이디어

### Self-Supervised Classification 전략
- **Label 없이 학습**: Ground-truth label을 학습 과정에서 사용하지 않음
- **자동 그룹화**: 이미지의 시각적 특성으로 자동으로 카테고리 발견
- **분류기 생성**: 최종 산출물은 새로운 이미지를 분류할 수 있는 classifier
- **평가는 supervised**: 학습 후 ground-truth와 비교하여 분류 성능 측정

### 학습 파이프라인

```
Phase 1: Feature Learning (특징 학습)
Unlabeled Images
    ↓
Feature Extractor (CNN)
    ↓
Feature Vectors

Phase 2: Pseudo-Label Generation (자동 레이블링)
Feature Vectors
    ↓
Clustering (K-means)
    ↓
Pseudo-Labels (자동 생성된 클래스 ID)

Phase 3: Classifier Training (분류기 학습)
Images + Pseudo-Labels
    ↓
Classification Network 학습
    ↓
개선된 Classifier

Phase 4: Iterative Refinement (반복 개선)
개선된 Features로 Re-clustering
    ↓
더 정확한 Pseudo-Labels
    ↓
더 좋은 Classifier

최종 산출물: Image → Class Prediction
```

## 모델 구조

### 1. Feature Extractor (특징 추출기)
**목적**: 이미지를 분류에 유용한 feature vector로 변환

**구조**:
- Convolutional layers: 이미지 패턴 인식
- Pooling layers: 공간 차원 축소
- Fully connected layers: Feature 통합
- 출력: Fixed-size feature vector (예: 256-dim)

**학습 방식**:
- 초기: Random initialization 또는 Autoencoder pre-training
- 반복 학습: Classification task로 지속적 개선

### 2. Clustering Module (자동 레이블 생성)
**목적**: Feature 유사도 기반으로 이미지를 K개 그룹으로 분할

**알고리즘**: K-means Clustering
- K=10 (Fashion-MNIST의 실제 클래스 개수)
- 유사한 feature를 가진 이미지들을 같은 그룹으로 할당
- 각 그룹에 pseudo-label (0~9) 부여

**역할**:
- Supervised learning의 "label" 역할을 대신
- 학습 과정에서 주기적으로 갱신

### 3. Classifier (분류기)
**목적**: 이미지를 입력받아 클래스를 예측

**구조**:
```
Input Image (28×28)
    ↓
Feature Extractor
    ↓
Feature Vector (256-dim)
    ↓
Classification Head (FC + Softmax)
    ↓
Class Probabilities (10 classes)
    ↓
Predicted Class (argmax)
```

**학습**:
- Loss: Cross-Entropy Loss (pseudo-label 기준)
- Optimizer: Adam 또는 SGD
- Feature Extractor와 Classification Head를 함께 학습

**추론 (Inference)**:
- 새로운 이미지 입력
- Feature 추출 → Classification Head 통과
- 가장 높은 확률의 클래스 출력

## 학습 알고리즘

### Iterative Training Loop

```
Initialization:
  - Feature Extractor 초기화 (random 또는 pre-trained)
  - 모든 이미지를 unlabeled dataset으로 구성

Main Loop (여러 iteration 반복):

  Iteration i:

  Step 1: Feature 추출
    - 전체 dataset의 feature 계산
    - features = feature_extractor(all_images)

  Step 2: Pseudo-Label 생성
    - K-means clustering으로 K개 그룹 생성
    - pseudo_labels = kmeans(features, K=10)

  Step 3: Classification 학습
    - 여러 epoch 동안 학습:
      for epoch in range(epochs_per_iteration):
        for batch in dataloader:
          predictions = classifier(images)
          loss = cross_entropy(predictions, pseudo_labels)
          optimize(loss)

  Step 4: Feature Extractor 개선
    - Classification 학습으로 feature가 자동 개선됨
    - 다음 iteration의 clustering이 더 정확해짐

수렴 조건:
  - Pseudo-label이 안정화 (변화율 < threshold)
  - Classification loss가 plateau
  - 최대 iteration 도달
```

### Warm-up Strategy (초기화 전략)

**Option 1: Random Features**
- Feature Extractor를 랜덤 초기화
- 첫 clustering 품질은 낮지만 점진적 개선
- 장점: 단순, 빠름
- 단점: 초기 수렴 느림

**Option 2: Autoencoder Pre-training**
- Reconstruction task로 먼저 feature 학습
- 의미있는 초기 feature 확보
- 장점: 초기 clustering 품질 높음
- 단점: 추가 학습 시간 필요

## 평가 전략

### 최종 목표: Classification Performance

**Primary Metrics (분류 성능)**:
- **Accuracy**: 전체 이미지 중 올바르게 분류한 비율
- **Precision/Recall/F1**: 클래스별 분류 정확도
- **Confusion Matrix**: 어떤 클래스가 어떤 클래스로 잘못 분류되는지

**평가 방법**:
1. Clustering으로 생성된 pseudo-label과 ground-truth 매칭
   - Hungarian algorithm으로 optimal matching 찾기
   - 예: cluster 0 → class 3, cluster 1 → class 7, ...
2. 매칭 후 classification accuracy 계산

### Intermediate Metrics (학습 과정 모니터링)

**Clustering Quality**:
- NMI (Normalized Mutual Information): Clustering이 실제 클래스와 얼마나 일치하는가
- Purity: 각 클러스터가 주로 하나의 클래스로 구성되는가

**Training Stability**:
- Loss curve: 안정적으로 감소하는가
- Pseudo-label stability: Re-clustering 시 얼마나 변하는가

### Test Protocol

```
Training Phase (unsupervised):
  - Ground-truth label 사용 안 함
  - Pseudo-label로만 학습

Evaluation Phase (supervised):
  - Test set의 ground-truth label 사용
  - Cluster-to-class optimal matching 수행
  - Classification accuracy 측정

Final Test:
  - 학습에 없던 새로운 이미지
  - Classifier가 올바른 클래스 예측하는지 평가
```

## 실험 설계

### 데이터셋 구성
```
Fashion-MNIST:
  - Train: 60,000 images (unlabeled로 취급)
  - Test: 10,000 images (평가용)
  - Classes: 10 (T-shirt, Trouser, Pullover, Dress, Coat,
              Sandal, Shirt, Sneaker, Bag, Ankle boot)

학습 시:
  - 모든 이미지를 하나의 directory에 섞음
  - Label 정보는 완전히 숨김

평가 시:
  - Ground-truth label로 분류 정확도 측정
```

### 하이퍼파라미터

**Model Architecture**:
- Feature dimension: 256
- CNN layers: 2~4 layers
- Classification head: 1~2 FC layers

**Clustering**:
- Algorithm: K-means
- K: 10 (클래스 개수와 동일)
- Re-clustering interval: 10 epochs

**Training**:
- Batch size: 128~256
- Learning rate: 0.001
- Optimizer: Adam
- Epochs per iteration: 50
- Total iterations: 5~10

**GPU Optimization**:
- Batch processing으로 feature 추출 가속
- GPU K-means 라이브러리 활용 (faiss, cuML)

## 주요 도전과제

### Challenge 1: Cluster-Class Mismatch
**문제**: Cluster가 실제 클래스와 다르게 형성될 수 있음
- 예: "검은색 옷"끼리 묶임 (티셔츠+드레스)

**해결책**:
- Strong feature learning: 색상보다 형태에 집중하도록 유도
- Data augmentation: 색상 변화 등으로 형태 특징 강조
- Multiple clustering: 여러 번 clustering 후 가장 좋은 결과 선택

### Challenge 2: Degenerate Solutions
**문제**: 모든 이미지가 하나의 클러스터로 몰림

**해결책**:
- Balanced assignment: 각 클러스터 크기 제약
- Good initialization: K-means++ 사용
- Feature normalization: L2 normalize features

### Challenge 3: Label Noise
**문제**: Pseudo-label의 오류가 학습을 방해

**해결책**:
- Confident samples first: 확신도 높은 샘플 먼저 학습
- Label smoothing: Hard label 대신 soft label
- Gradual refinement: 점진적 pseudo-label 개선

## 성공 기준

### 주요 목표: 분류 정확도 (Classification Accuracy)
**Target**: Self-supervised 학습으로 Test set Accuracy > 70%

**비교 기준**:
- Random Classifier: 10% (10개 클래스 uniform random)
- K-means Only (baseline): ~40-50% (feature 개선 없이 한 번만 clustering)
- Supervised Learning (upper bound): ~90% (ground-truth label 사용)
- **Self-supervised Target**: 70-80%

### 세부 분류 성능 지표

**1. Overall Accuracy**
- Test set 전체에 대한 분류 정확도
- Metric: `correct_predictions / total_samples`
- Target: > 70%

**2. Per-Class Performance**
- **Precision**: 예측한 클래스 중 실제 정답 비율
- **Recall**: 실제 클래스 중 올바르게 예측한 비율
- **F1-score**: Precision과 Recall의 조화평균
- Target: 모든 클래스 F1-score > 0.6 (클래스 불균형 없이 골고루 학습)

**3. Confusion Matrix 분석**
- 10×10 confusion matrix 생성
- 어떤 클래스가 어떤 클래스로 혼동되는지 패턴 분석
- 유사 클래스 간 혼동은 허용 (예: T-shirt ↔ Shirt)
- 형태가 다른 클래스 간 혼동은 최소화 (예: Bag ↔ Trouser)

### 학습 안정성 지표

**1. Loss Convergence**
- Classification loss가 안정적으로 감소
- Plateau 도달 후 fluctuation < 5%
- Overfitting 징후 모니터링 (train vs validation loss gap)

**2. Pseudo-Label Stability**
- Re-clustering 시 pseudo-label 변화율 추적
- Iteration이 진행될수록 변화율 감소 (수렴)
- 최종 iteration에서 변화율 < 10%

**3. Cluster Quality**
- **NMI (Normalized Mutual Information)**: > 0.65
  - Clustering과 ground-truth 간 상호정보량
- **Purity**: > 0.75
  - 각 클러스터의 dominant class 비율
- **Silhouette Score**: > 0.25
  - Feature space에서 클러스터 분리도

### 실패 케이스 분석

**허용 가능한 실패**:
- 유사 형태 클래스 간 혼동: T-shirt ↔ Shirt ↔ Pullover
- 유사 카테고리: Sneaker ↔ Ankle boot

**허용 불가능한 실패**:
- 형태가 완전히 다른 클래스 혼동: Bag ↔ Sandal
- 특정 클래스가 완전히 학습 실패 (F1-score < 0.3)
- Degenerate solution (모든 이미지를 하나의 클래스로 예측)

### Baseline 실험 비교

**Experiment 1: Random Classifier**
- 무작위로 클래스 할당
- Expected Accuracy: 10%
- 목적: 최소 성능 기준

**Experiment 2: K-means Only**
- Feature 학습 없이 한 번만 clustering
- Expected Accuracy: 40-50%
- 목적: Iterative refinement 효과 검증

**Experiment 3: Supervised Learning**
- Ground-truth label로 직접 학습
- Expected Accuracy: ~90%
- 목적: 성능 상한선 파악

**Experiment 4: Self-supervised (본 연구)**
- Pseudo-label로 iterative 학습
- Target Accuracy: 70-80%
- 목적: Label 없이 얼마나 좋은 분류기를 만들 수 있는가

## 예상 결과

### 잘 분류될 것으로 예상
- 구별되는 형태: Trouser vs Dress vs Bag
- 명확한 특징: Sandal vs Sneaker vs Ankle boot

### 어려울 것으로 예상
- 유사한 형태: T-shirt vs Shirt vs Pullover
- 색상/질감 의존도 높은 경우

### 학습 곡선 예상
- Iteration 1-2: Accuracy ~30-40% (초기 clustering)
- Iteration 3-5: Accuracy ~60-70% (feature 개선)
- Iteration 6+: Accuracy plateau ~70-80%

## 최종 산출물

### Trained Classifier
```
입력: Fashion item image (28×28 grayscale)
    ↓
처리: Feature Extraction + Classification
    ↓
출력: Class prediction (0~9)
```

### 사용 시나리오
1. 새로운 패션 아이템 이미지 입력
2. Feature Extractor로 특징 추출
3. Classification Head로 클래스 예측
4. 예측된 클래스 출력 (예: "이것은 Sneaker입니다")

### Transfer Learning 가능
- 학습된 Feature Extractor를 다른 task에 재사용
- Fine-tuning으로 적은 labeled data로 성능 향상
- Domain adaptation에 활용
