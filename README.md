# Self-Supervised Image Classification with Contrastive Learning

PyTorch implementation of self-supervised image classification using contrastive learning and iterative pseudo-labeling on STL-10 dataset.

## 🎯 Project Goal

Build a classification system that learns to categorize images **without using ground-truth labels during training**, achieving >70% accuracy through self-supervised learning with contrastive pre-training.

## 📊 Current Dataset: STL-10

**STL-10** is the primary dataset for this project - specifically designed for self-supervised learning:
- **96×96 RGB** images (vs 28×28 grayscale in Fashion-MNIST)
- **10 classes**: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- **100,000 unlabeled images** for contrastive learning
- **5,000 training + 8,000 test** labeled images for evaluation

### Why STL-10?
✅ **High resolution** → More discriminative features
✅ **RGB colors** → Can use color augmentation (jittering)
✅ **Distinct classes** → Visually different objects (vs similar clothing items)
✅ **Unlabeled data** → Ideal for contrastive learning (100k images!)
✅ **Standard benchmark** → Used in self-supervised learning research

*Note: Fashion-MNIST support is still available but STL-10 is recommended.*

## 🏗️ Architecture

### Overall Pipeline
```
Phase 0: Contrastive Pre-training (100k unlabeled images)
  ↓
Phase 1-N: Iterative Pseudo-Labeling
  Feature Extraction → Clustering → Classifier Training → Repeat
```

### Contrastive Pre-training (SimCLR-style)
```
Image → [Augmentation 1] → View 1 ─┐
                                     ├→ Contrastive Loss → Learn Features
Image → [Augmentation 2] → View 2 ─┘

Augmentations: RandomResizedCrop, ColorJitter, Flip, Blur, Rotation
Temperature: 0.5 | Epochs: 100 | Unlabeled: 100k images
```

### Iterative Pseudo-Labeling
```
Input (96×96 RGB) → STL10Encoder (CNN) → 512-dim features (L2-normalized)
                                             ↓
                                       K-means Clustering
                                             ↓
                                       Pseudo-labels (0-9)
                                             ↓
                             Classification Head → Class Predictions
```

### STL10Encoder Architecture
- **4 Conv blocks**: 64 → 128 → 256 → 512 channels
- BatchNorm + ReLU + MaxPool
- Global Average Pooling
- Output: 512-dim L2-normalized features
- **Parameters**: 4.7M

## 📁 Project Structure

```
toy/
├── config/
│   ├── stl10_config.yaml        # STL-10 configuration (default)
│   └── config.yaml              # Fashion-MNIST configuration
├── dataset/
│   ├── STL10/                   # STL-10 dataset
│   │   ├── train_images/        # 5,000 images
│   │   ├── test_images/         # 8,000 images
│   │   └── unlabeled_images/    # 100,000 images (contrastive learning)
│   └── FashionMNIST/           # Fashion-MNIST dataset (legacy)
├── models/
│   ├── stl10_encoder.py        # STL-10 encoder (96×96 RGB)
│   ├── feature_extractor.py    # Fashion-MNIST encoder (28×28 grayscale)
│   ├── classifier.py           # Classification head
│   └── contrastive.py          # Contrastive learning components
├── utils/
│   ├── stl10_loader.py         # STL-10 data loading
│   ├── stl10_augmentation.py   # RGB image augmentations
│   ├── data_loader.py          # Fashion-MNIST data loading
│   ├── augmentation_v2.py      # Fashion-MNIST augmentations
│   ├── clustering.py           # K-means with GPU support
│   ├── metrics.py              # Hungarian matching, NMI, Purity
│   └── visualization.py        # Plotting utilities
├── train.py                    # Main training script (auto-detects dataset)
├── CHANGES.md                  # Recent changes and migration guide
├── STL10_SETUP.md             # Detailed STL-10 setup guide
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision pyyaml scikit-learn scipy tqdm pillow kaggle
```

**Optional (Recommended):** Install FAISS for GPU-accelerated clustering:
```bash
pip install faiss-gpu
```

### 2. Download STL-10 Dataset

Already downloaded and placed in `dataset/STL10/`!

To download manually:
```bash
kaggle datasets download -d jessicali9530/stl10
unzip stl10.zip -d dataset/STL10/
```

### 3. Train the Model

```bash
python train.py
```

The script automatically:
- Loads `config/stl10_config.yaml`
- Runs contrastive pre-training on 100k unlabeled images (100 epochs)
- Performs iterative pseudo-labeling (30 iterations)
- Saves checkpoints to `checkpoints_stl10/`

**Configuration:** Edit `config/stl10_config.yaml` to adjust:
- `pretrain_epochs`: Contrastive pre-training epochs (default: 100)
- `num_iterations`: Refinement cycles (default: 30)
- `epochs_per_iteration`: Classifier training epochs (default: 15)
- `batch_size`: Batch size (default: 256)
- `temperature`: Contrastive learning temperature (default: 0.5)

### 4. Monitor Training

```bash
# Training creates these directories:
checkpoints_stl10/  # Model checkpoints
results_stl10/      # Results and plots
logs_stl10/         # Training logs
```

## 📈 Expected Performance

### STL-10 (Current)
| Iteration | Accuracy | NMI | Purity | Status |
|-----------|----------|-----|--------|--------|
| 1 | 45-55% | 0.45-0.55 | 0.50-0.60 | Post contrastive learning |
| 5-10 | 60-70% | 0.55-0.65 | 0.65-0.75 | Improving |
| 15-20 | **70-80%** | **0.65-0.75** | **0.75-0.85** | **Target** |
| 25-30 | **75-85%** | **0.70-0.80** | **0.80-0.90** | **Converged** |

### Fashion-MNIST (Legacy - Not Recommended)
| Iteration | Accuracy | NMI | Purity | Status |
|-----------|----------|-----|--------|--------|
| Final | 30-40% | 0.25-0.35 | 0.30-0.40 | Limited by dataset |

**Why Fashion-MNIST struggles:**
- ❌ Low resolution (28×28)
- ❌ Grayscale only (no color augmentation)
- ❌ Similar classes (T-shirt vs Shirt vs Pullover)
- ❌ No unlabeled data for contrastive learning

## ✅ Success Criteria

### STL-10 Targets
- ✅ **Overall Accuracy > 75%** (vs supervised ~94%)
- ✅ **NMI > 0.70** (clustering-class alignment)
- ✅ **Purity > 0.80** (cluster homogeneity)
- ✅ **All Per-Class F1 > 0.65** (no class left behind)
- ✅ **Label Stability > 0.98** (convergence)

## 🔑 Key Implementation Details

### 1. Contrastive Pre-training (Critical!)
```python
# Two augmented views of same image
view1, view2 = augment(image), augment(image)

# Project to contrastive space
z1, z2 = projector(encoder(view1)), projector(encoder(view2))

# NT-Xent Loss (Temperature-scaled)
loss = -log(exp(sim(z1, z2) / τ) / Σ exp(sim(z1, zk) / τ))
```
- Learns invariant features before clustering
- Uses 100k unlabeled STL-10 images
- Temperature τ = 0.5 for balanced learning

### 2. Strong Augmentation for RGB
```python
RandomResizedCrop(96, scale=(0.2, 1.0))  # SimCLR-style
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # RGB!
RandomHorizontalFlip(p=0.5)  # OK for natural images
GaussianBlur(kernel_size=9, p=0.5)
RandomRotation(degrees=15)
```

### 3. L2 Feature Normalization
```python
features = F.normalize(features, p=2, dim=1)
```
- Applied after encoder, before clustering
- Ensures cosine similarity clustering

### 4. Hungarian Algorithm for Evaluation
```python
from scipy.optimize import linear_sum_assignment
mapping, accuracy = compute_cluster_to_class_mapping(pseudo_labels, true_labels)
```
- Optimal cluster-to-class assignment
- Cluster IDs don't match class IDs

### 5. GPU-Accelerated Clustering
- **FAISS** (recommended): 10x faster for 100k samples
- **Fallback**: sklearn K-means if FAISS unavailable

## 📊 Outputs

### Checkpoints (STL-10)
- `checkpoints_stl10/best_model.pth`: Model with highest test accuracy
- `checkpoints_stl10/final_model.pth`: Final model after all iterations
- `checkpoints_stl10/model_iter_*.pth`: Per-iteration checkpoints

### Results
- `results_stl10/training_history.npy`: Metrics across iterations
- `results_stl10/confusion_matrix.png`: Final confusion matrix
- `results_stl10/clustering_quality.png`: NMI/Purity over iterations

## 🔄 Switching Between Datasets

### Use STL-10 (Default)
```bash
python train.py  # Uses config/stl10_config.yaml
```

### Use Fashion-MNIST
Edit `train.py` line 303:
```python
config = load_config('config/config.yaml')  # Change from stl10_config.yaml
```

Or edit config file's `dataset_name`:
```yaml
data:
  dataset_name: 'fashion_mnist'  # or 'stl10'
```

## 🛠️ Troubleshooting

### CUDA Out of Memory
**Solution**: Reduce batch size in `stl10_config.yaml`:
```yaml
data:
  batch_size: 128  # or 64 instead of 256
```

### Slow Contrastive Pre-training
**Solution**: Reduce pre-training epochs:
```yaml
contrastive:
  pretrain_epochs: 50  # instead of 100
```

### Poor Clustering Quality (NMI < 0.4)
**Solution**:
1. Increase contrastive pre-training epochs
2. Check if contrastive loss is decreasing
3. Verify augmentations are working (views should be different)

## 📚 STL-10 Classes

0. Airplane ✈️
1. Bird 🐦
2. Car 🚗
3. Cat 🐱
4. Deer 🦌
5. Dog 🐕
6. Horse 🐴
7. Monkey 🐵
8. Ship 🚢
9. Truck 🚚

## 🔬 Technical Details

**Why Contrastive Learning?**
- Pre-trains encoder on large unlabeled data (100k images)
- Learns invariant features before clustering
- Significantly improves initial clustering quality

**Why STL-10 over Fashion-MNIST?**
- Higher resolution enables more discriminative features
- RGB colors enable color-based augmentation
- Visually distinct classes are easier to cluster
- 100k unlabeled images perfect for contrastive learning

**Key Hyperparameters:**
- `feature_dim=512`: Feature vector size (increased from 256)
- `temperature=0.5`: Contrastive learning temperature
- `pretrain_epochs=100`: Contrastive pre-training duration
- `projection_dim=128`: Contrastive projection head output

## 📖 References

- Contrastive Learning: [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- STL-10 Dataset: [Coates et al., 2011](https://cs.stanford.edu/~acoates/stl10/)
- Deep Clustering: [Caron et al., 2018](https://arxiv.org/abs/1807.05520)
- Hungarian Algorithm: [scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

## 📄 Additional Documentation

- **`STL10_SETUP.md`**: Detailed STL-10 setup and usage guide
- **`CHANGES.md`**: Recent changes and migration guide from Fashion-MNIST
- **`config/stl10_config.yaml`**: All hyperparameters with comments

---

**Built with PyTorch** | **Self-Supervised Learning** | **Contrastive Pre-training** | **STL-10 Dataset**

Last Updated: 2025-12-07 | Project migrated from Fashion-MNIST to STL-10 for better performance
