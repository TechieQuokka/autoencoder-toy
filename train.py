"""
STL-10 Self-Supervised Image Classification Training
Features:
- Contrastive pre-training (SimCLR-style)
- Confidence-based pseudo-label filtering
- Strong data augmentation for natural images
- Learning rate scheduling
- GPU optimization for RTX 3060 12GB
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import numpy as np
from tqdm import tqdm

from models.classifier import SelfSupervisedClassifier
from models.contrastive import ContrastiveProjector, NTXentLoss, ClusterContrastiveLoss
from utils.clustering import KMeansClustering, ConfidenceFilter
from utils.metrics import evaluate_clustering


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_modules(config):
    """
    Load STL-10 dataset modules

    Returns:
        encoder_class, get_train_loaders, get_contrastive_loader, class_names
    """
    from models.stl10_encoder import STL10Encoder as EncoderClass
    from utils.stl10_loader import get_stl10_loaders, get_stl10_contrastive_loader

    STL10_CLASSES = [
        'airplane', 'bird', 'car', 'cat', 'deer',
        'dog', 'horse', 'monkey', 'ship', 'truck'
    ]

    return EncoderClass, get_stl10_loaders, get_stl10_contrastive_loader, STL10_CLASSES


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_scheduler(optimizer, config, total_steps):
    """Learning rate scheduler 생성"""
    scheduler_config = config['training'].get('scheduler', {})

    if not scheduler_config.get('enabled', False):
        return None

    scheduler_type = scheduler_config.get('type', 'cosine')
    warmup_epochs = scheduler_config.get('warmup_epochs', 5)
    min_lr = scheduler_config.get('min_lr', 1e-6)

    if scheduler_type == 'cosine':
        # Warmup + Cosine Annealing
        def lr_lambda(step):
            if step < warmup_epochs:
                return float(step) / float(max(1, warmup_epochs))
            progress = float(step - warmup_epochs) / float(max(1, total_steps - warmup_epochs))
            return max(min_lr / config['training']['learning_rate'],
                      0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler


def contrastive_pretrain(model, projector, contrastive_loader, optimizer,
                         device, config, scaler=None):
    """
    대조 학습 사전훈련

    SimCLR 스타일로 특징 추출기를 사전 훈련
    같은 이미지의 다른 augmentation은 가깝게, 다른 이미지는 멀게 학습
    """
    pretrain_epochs = config['contrastive'].get('pretrain_epochs', 20)
    temperature = config['contrastive'].get('temperature', 0.5)
    gpu_config = config.get('gpu', {})
    grad_accum = gpu_config.get('gradient_accumulation', 1)

    nt_xent_loss = NTXentLoss(temperature=temperature).to(device)

    print("\n" + "="*80)
    print("Phase 0: Contrastive Pre-training")
    print("="*80)
    print(f"  Epochs: {pretrain_epochs}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch size: {config['data']['batch_size']}")
    if grad_accum > 1:
        print(f"  Gradient Accumulation: {grad_accum} (effective batch = {config['data']['batch_size'] * grad_accum})")

    model.train()
    projector.train()

    use_amp = scaler is not None

    for epoch in range(pretrain_epochs):
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(contrastive_loader, desc=f"Contrastive Epoch {epoch+1}/{pretrain_epochs}",
                    leave=False)

        for batch_idx, (view1, view2) in enumerate(pbar):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            # Gradient accumulation 시작 시점에만 zero_grad
            if batch_idx % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    # 특징 추출
                    feat1 = model.extract_features(view1)
                    feat2 = model.extract_features(view2)

                    # 프로젝션
                    z1 = projector(feat1)
                    z2 = projector(feat2)

                    # NT-Xent Loss
                    loss = nt_xent_loss(z1, z2)

                # Gradient accumulation을 위해 loss 스케일링
                scaled_loss = scaler.scale(loss / grad_accum)
                scaled_loss.backward()

                # Accumulation 완료 시점에만 optimizer step
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(contrastive_loader):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                feat1 = model.extract_features(view1)
                feat2 = model.extract_features(view2)
                z1 = projector(feat1)
                z2 = projector(feat2)
                loss = nt_xent_loss(z1, z2)

                # Gradient accumulation을 위해 loss 스케일링
                (loss / grad_accum).backward()

                # Accumulation 완료 시점에만 optimizer step
                if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(contrastive_loader):
                    optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"  Epoch [{epoch+1}/{pretrain_epochs}] - Contrastive Loss: {avg_loss:.4f}")

    print("  ✓ Contrastive pre-training completed!")
    return model


def extract_features_with_cache(model, cached_images, batch_size, device):
    """캐시된 이미지에서 특징 추출"""
    model.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(cached_images), batch_size):
            batch = cached_images[i:i + batch_size].to(device, non_blocking=True)
            features = model.extract_features(batch)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0).numpy()


def train_one_epoch_weighted(model, dataloader, criterion, optimizer, device,
                            sample_weights=None, scaler=None, label_smoothing=0.1,
                            gradient_accumulation=1):
    """
    가중치를 적용한 1 epoch 학습 (Label Smoothing 포함)

    Args:
        model: 분류 모델
        dataloader: (images, pseudo_labels, indices) 반환하는 DataLoader
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        sample_weights: (N,) 샘플별 가중치
        scaler: AMP scaler
        label_smoothing: Label smoothing 계수 (noisy labels에 효과적)
        gradient_accumulation: Gradient accumulation steps (메모리 절약)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    # Label smoothing을 적용한 손실 함수
    smooth_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 3:
            images, pseudo_labels, indices = batch_data
        else:
            images, pseudo_labels = batch_data
            indices = None

        images = images.to(device, non_blocking=True)
        pseudo_labels = pseudo_labels.to(device, non_blocking=True)

        # Gradient accumulation 시작 시점에만 zero_grad
        if batch_idx % gradient_accumulation == 0:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(images)

                if sample_weights is not None and indices is not None:
                    # 가중치 적용 손실
                    weights = torch.from_numpy(sample_weights[indices.numpy()]).to(device)
                    loss_per_sample = nn.functional.cross_entropy(
                        logits, pseudo_labels, reduction='none', label_smoothing=label_smoothing
                    )
                    loss = (loss_per_sample * weights).mean()
                else:
                    loss = smooth_criterion(logits, pseudo_labels)

            # Gradient accumulation을 위해 loss를 스케일링
            scaled_loss = scaler.scale(loss / gradient_accumulation)
            scaled_loss.backward()

            # Accumulation 완료 시점에만 optimizer step
            if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(images)

            if sample_weights is not None and indices is not None:
                weights = torch.from_numpy(sample_weights[indices.numpy()]).to(device)
                loss_per_sample = nn.functional.cross_entropy(
                    logits, pseudo_labels, reduction='none', label_smoothing=label_smoothing
                )
                loss = (loss_per_sample * weights).mean()
            else:
                loss = smooth_criterion(logits, pseudo_labels)

            # Gradient accumulation을 위해 loss를 스케일링
            (loss / gradient_accumulation).backward()

            # Accumulation 완료 시점에만 optimizer step
            if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_on_test(model, test_loader, device, n_classes=10):
    """테스트 셋 평가"""
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu())
            all_true_labels.append(labels)

    all_predictions = torch.cat(all_predictions).numpy()
    all_true_labels = torch.cat(all_true_labels).numpy()

    results = evaluate_clustering(
        all_predictions, all_true_labels,
        n_clusters=n_classes, n_classes=n_classes
    )

    return results


class IndexedTensorDataset(torch.utils.data.Dataset):
    """인덱스도 함께 반환하는 TensorDataset"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], idx


def main():
    """Main training loop with all enhancements"""
    # Load configuration
    config = load_config('config/stl10_config.yaml')  # Use STL-10 config
    set_seed(config['seed'])

    # Load STL-10 dataset modules
    EncoderClass, get_train_loaders, get_contrastive_loader_fn, class_names = load_dataset_modules(config)

    print(f"=" * 80)
    print(f"Dataset: STL-10")
    print(f"Classes: {class_names}")
    print(f"=" * 80 + "\n")

    # Setup device
    device_name = config['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # GPU Optimization (RTX 3060 12GB + WSL2)
    if device.type == 'cuda':
        gpu_config = config.get('gpu', {})

        # cuDNN 자동 최적화
        if gpu_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark enabled")

        # GPU 정보 출력
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Mixed Precision 확인
        if gpu_config.get('mixed_precision', True):
            print("✓ Mixed Precision (FP16) enabled")

        # Gradient Accumulation 확인
        grad_accum = gpu_config.get('gradient_accumulation', 1)
        if grad_accum > 1:
            print(f"✓ Gradient Accumulation: {grad_accum} steps (effective batch = {config['data']['batch_size'] * grad_accum})")

    print()

    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)

    # Initialize STL-10 encoder
    print("Initializing STL-10 encoder...")
    encoder = EncoderClass(
        feature_dim=config['model']['feature_dim']
    ).to(device)

    # Wrap encoder in classifier
    model = SelfSupervisedClassifier(
        num_classes=config['model']['num_classes'],
        feature_dim=config['model']['feature_dim'],
        encoder=encoder
    ).to(device)

    # 대조 학습용 프로젝터
    projector = ContrastiveProjector(
        input_dim=config['model']['feature_dim'],
        hidden_dim=config['contrastive'].get('projection_hidden_dim', 256),
        output_dim=config['model'].get('projection_dim', 128)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Initialize optimizer (모델 + 프로젝터)
    all_params = list(model.parameters()) + list(projector.parameters())

    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            all_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            all_params,
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )

    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed Precision (AMP) enabled")

    # ===== Phase 0: Contrastive Pre-training =====
    if config['contrastive'].get('enabled', True):
        print("\nLoading data for contrastive learning...")
        contrastive_loader = get_contrastive_loader_fn(
            data_dir=config['data']['data_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )

        model = contrastive_pretrain(
            model, projector, contrastive_loader, optimizer, device, config, scaler
        )

        # 프로젝터는 사전훈련 후 버림 (특징 추출기만 사용)
        del projector
        torch.cuda.empty_cache()

    # Load STL-10 data for iterative training
    print("\nLoading STL-10 dataset...")
    train_loader, test_loader = get_train_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augmentation=config['data'].get('augmentation', 'strong'),
        use_unlabeled=config['data'].get('use_unlabeled', True)
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 분류기만을 위한 새 옵티마이저
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    total_steps = config['training']['num_iterations'] * config['training']['epochs_per_iteration']
    scheduler = get_scheduler(optimizer, config, total_steps)

    # Initialize clustering
    clustering = KMeansClustering(
        n_clusters=config['clustering']['n_clusters'],
        device=device,
        use_gpu=config['clustering']['use_gpu_clustering'],
        max_iter=config['clustering']['max_iter'],
        n_init=config['clustering']['n_init']
    )

    # 신뢰도 필터
    confidence_filter = ConfidenceFilter(
        threshold=config['confidence'].get('threshold', 0.5),
        soft_filtering=config['confidence'].get('soft_filtering', True)
    )

    # Cache training images
    print("\nCaching training images...")
    cached_images = []
    cached_labels = []
    for images, labels in tqdm(train_loader, desc="Caching", leave=False):
        cached_images.append(images)
        cached_labels.append(labels)
    cached_images = torch.cat(cached_images, dim=0)
    cached_labels = torch.cat(cached_labels, dim=0).numpy()
    print(f"Cached {len(cached_images)} images\n")

    # Tracking metrics
    history = {
        'iteration': [], 'nmi': [], 'purity': [], 'matched_accuracy': [],
        'test_f1': [], 'label_stability': [], 'train_loss': [],
        'high_conf_ratio': []
    }

    prev_pseudo_labels = None
    best_accuracy = 0.0
    patience_counter = 0

    print("="*80)
    print("Starting Iterative Self-Supervised Training (Enhanced)")
    print("="*80)

    # ===== Main Iterative Loop =====
    for iteration in range(config['training']['num_iterations']):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration + 1}/{config['training']['num_iterations']}")
        print(f"{'='*80}\n")

        # ===== STEP 1: Feature Extraction =====
        print("Step 1: Extracting features...")
        all_features = extract_features_with_cache(
            model, cached_images, config['data']['batch_size'], device
        )
        print(f"  Features shape: {all_features.shape}")

        # ===== STEP 2: K-means Clustering =====
        print("\nStep 2: Clustering...")
        pseudo_labels, centroids = clustering.fit_predict(all_features)
        print(f"  Initial cluster sizes: {np.bincount(pseudo_labels)}")

        # 클러스터 재균형화 (너무 불균형한 클러스터 방지)
        pseudo_labels = clustering.rebalance_clusters(all_features, pseudo_labels)
        print(f"  After rebalancing: {np.bincount(pseudo_labels)}")

        # Evaluate clustering
        clustering_results = evaluate_clustering(
            pseudo_labels, cached_labels,
            n_clusters=config['model']['num_classes'],
            n_classes=config['model']['num_classes']
        )
        print(f"\n  Clustering Quality:")
        print(f"    NMI: {clustering_results['nmi']:.4f}")
        print(f"    Purity: {clustering_results['purity']:.4f}")
        print(f"    Matched Accuracy: {clustering_results['matched_accuracy']:.4f}")

        # Label stability
        if prev_pseudo_labels is not None:
            stability = clustering.compute_stability(prev_pseudo_labels, pseudo_labels)
            print(f"    Label Stability: {stability:.4f}")
            history['label_stability'].append(stability)

            # 조기 수렴 조건 강화:
            # 1. 최소 15 iteration 이상 진행해야 함
            # 2. Stability가 threshold 이상이어야 함
            # 3. 성능 개선이 없어야 함 (patience 기반으로 처리)
            min_iterations = config['convergence'].get('min_iterations', 15)
            if config['convergence']['early_stopping'] and \
               stability > config['convergence']['label_stability_threshold'] and \
               iteration >= min_iterations:
                print(f"\n  ✓ Converged after {iteration+1} iterations! Stopping early.")
                break
        else:
            history['label_stability'].append(0.0)

        prev_pseudo_labels = pseudo_labels.copy()

        # ===== STEP 3: Confidence-based Filtering =====
        sample_weights = None
        if config['confidence'].get('enabled', True) and \
           iteration >= config['confidence'].get('warmup_iterations', 3):

            print("\nStep 3: Computing confidence scores...")
            confidence = confidence_filter.compute_confidence(
                all_features, centroids, pseudo_labels
            )
            sample_weights = confidence_filter.get_sample_weights(confidence)

            high_conf_ratio = (confidence >= config['confidence']['threshold']).mean()
            print(f"  High confidence samples: {high_conf_ratio:.1%}")
            history['high_conf_ratio'].append(high_conf_ratio)
        else:
            history['high_conf_ratio'].append(1.0)
            print("\nStep 3: Confidence filtering (warmup - skipped)")

        # ===== STEP 4: Train Classifier =====
        print(f"\nStep 4: Training classifier for {config['training']['epochs_per_iteration']} epochs...")

        # 인덱스 포함 DataLoader 생성
        if sample_weights is not None:
            train_dataset = IndexedTensorDataset(
                cached_images,
                torch.from_numpy(pseudo_labels).long()
            )
        else:
            train_dataset = TensorDataset(
                cached_images,
                torch.from_numpy(pseudo_labels).long()
            )

        labeled_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True if config['data']['num_workers'] > 0 else False
        )

        epoch_losses = []
        gpu_config = config.get('gpu', {})
        grad_accum = gpu_config.get('gradient_accumulation', 1)
        empty_cache_freq = gpu_config.get('empty_cache_freq', 0)

        for epoch in range(config['training']['epochs_per_iteration']):
            avg_loss = train_one_epoch_weighted(
                model, labeled_loader, criterion, optimizer, device,
                sample_weights=sample_weights, scaler=scaler,
                gradient_accumulation=grad_accum
            )
            epoch_losses.append(avg_loss)

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # GPU 캐시 비우기 (메모리 관리)
            if empty_cache_freq > 0 and (epoch + 1) % empty_cache_freq == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch [{epoch+1}/{config['training']['epochs_per_iteration']}], "
                      f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # ===== STEP 5: Evaluate =====
        if config['evaluation']['eval_every_iteration']:
            print("\nStep 5: Evaluating on test set...")
            test_results = evaluate_on_test(model, test_loader, device, config['model']['num_classes'])

            print(f"  Test Performance:")
            print(f"    Matched Accuracy: {test_results['matched_accuracy']:.4f}")
            print(f"    NMI: {test_results['nmi']:.4f}")
            print(f"    F1 Score: {test_results['f1']:.4f}")

            # Per-class F1
            print(f"\n  Per-class F1:")
            for i, (f1, cls_name) in enumerate(zip(test_results['per_class_f1'], class_names)):
                status = "✓" if f1 >= 0.6 else "✗"
                print(f"    {status} {cls_name:15s}: {f1:.4f}")

            # Save best model
            if test_results['matched_accuracy'] > best_accuracy:
                best_accuracy = test_results['matched_accuracy']
                patience_counter = 0
                best_model_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'accuracy': best_accuracy,
                    'mapping': test_results['mapping']
                }, best_model_path)
                print(f"\n  ★ New best model! Accuracy: {best_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['convergence'].get('patience', 5):
                    print(f"\n  ✗ No improvement for {patience_counter} iterations. Early stopping.")
                    break

            # Record history
            history['iteration'].append(iteration + 1)
            history['nmi'].append(clustering_results['nmi'])
            history['purity'].append(clustering_results['purity'])
            history['matched_accuracy'].append(test_results['matched_accuracy'])
            history['test_f1'].append(test_results['f1'])
            history['train_loss'].append(np.mean(epoch_losses))

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
    for i, (f1, cls_name) in enumerate(zip(final_results['per_class_f1'], class_names)):
        status = "✓" if f1 >= 0.6 else "✗"
        print(f"  {status} {i}: {cls_name:15s} - F1: {f1:.4f}")

    # Save final model and history
    final_model_path = os.path.join(config['paths']['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_results': final_results,
        'history': history,
        'config': config
    }, final_model_path)

    history_path = os.path.join(config['paths']['results_dir'], 'training_history.npy')
    np.save(history_path, history)

    print(f"\n✓ Training completed!")
    print(f"  Best accuracy: {best_accuracy:.4f}")
    print(f"  Final model: {final_model_path}")

    # Success criteria check
    print(f"\n{'='*80}")
    print("Success Criteria Check:")
    print(f"{'='*80}")
    print(f"  Overall Accuracy > 70%:     {'✓' if final_results['matched_accuracy'] > 0.70 else '✗'} ({final_results['matched_accuracy']:.2%})")
    print(f"  NMI > 0.65:                 {'✓' if final_results['nmi'] > 0.65 else '✗'} ({final_results['nmi']:.4f})")
    print(f"  Purity > 0.75:              {'✓' if final_results['purity'] > 0.75 else '✗'} ({final_results['purity']:.4f})")

    all_f1_good = all(f1 > 0.60 for f1 in final_results['per_class_f1'])
    print(f"  All Per-Class F1 > 0.60:    {'✓' if all_f1_good else '✗'}")

    return model, history, final_results


if __name__ == '__main__':
    model, history, results = main()
