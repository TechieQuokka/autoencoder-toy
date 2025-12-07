# Self-Supervised Image Classification for Fashion-MNIST

PyTorch implementation of self-supervised image classification using iterative pseudo-labeling and feature refinement.

## 🎯 Project Goal

Build a classification system that learns to categorize Fashion-MNIST images **without using ground-truth labels during training**, achieving >70% accuracy through self-supervised learning.

## 📊 Approach

**Iterative Pseudo-Labeling Strategy:**
1. **Feature Extraction**: CNN extracts 256-dim features from images
2. **Clustering**: K-means groups similar images → pseudo-labels
3. **Classifier Training**: Train on pseudo-labels for 50 epochs
4. **Feature Refinement**: Improved features → better clustering
5. **Repeat**: Iterate until pseudo-labels stabilize

## 🏗️ Architecture

```
Input (28×28) → Feature Extractor (CNN) → 256-dim features (L2-normalized)
                                            ↓
                                      K-means Clustering
                                            ↓
                                      Pseudo-labels (0-9)
                                            ↓
                            Classification Head → Class Predictions
```

**Feature Extractor:**
- 3 Conv blocks: 32 → 64 → 128 channels
- BatchNorm + ReLU + MaxPool
- FC layers: 1152 → 512 → 256
- L2 normalization (critical for clustering stability)

**Classifier:**
- Feature Extractor + Classification Head (256 → 128 → 10)
- Trained end-to-end with pseudo-labels

## 📁 Project Structure

```
toy/
├── config/
│   └── config.yaml              # Hyperparameters
├── models/
│   ├── feature_extractor.py     # CNN feature extractor
│   └── classifier.py            # Complete classification model
├── utils/
│   ├── data_loader.py           # Fashion-MNIST data loading
│   ├── clustering.py            # K-means with GPU support
│   ├── metrics.py               # Hungarian matching, NMI, Purity
│   └── visualization.py         # Plotting utilities
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Single image prediction
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional (Recommended):** Install FAISS for GPU-accelerated clustering:
```bash
pip install faiss-gpu
```

### 2. Train the Model

```bash
python train.py
```

**Configuration:** Edit `config/config.yaml` to adjust hyperparameters:
- `num_iterations`: Number of refinement cycles (default: 10)
- `epochs_per_iteration`: Classifier training epochs (default: 50)
- `batch_size`: Batch size (default: 128)
- `learning_rate`: Learning rate (default: 0.001)

### 3. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/final_model.pth --verbose
```

### 4. Inference on Single Image

```bash
python inference.py --image path/to/image.png --checkpoint checkpoints/final_model.pth
```

## 📈 Expected Performance

| Iteration | Accuracy | NMI | Purity | Status |
|-----------|----------|-----|--------|--------|
| 1 | 30-40% | 0.35-0.45 | 0.40-0.50 | Baseline |
| 2-3 | 50-60% | 0.50-0.60 | 0.60-0.70 | Improving |
| 4-6 | 65-75% | 0.60-0.70 | 0.70-0.80 | Approaching target |
| 7+ | **70-80%** | **0.65+** | **0.75+** | **Target achieved** |

## ✅ Success Criteria

- ✅ **Overall Accuracy > 70%** (vs supervised ~90%, k-means only ~40-50%)
- ✅ **NMI > 0.65** (clustering-class alignment)
- ✅ **Purity > 0.75** (cluster homogeneity)
- ✅ **All Per-Class F1 > 0.60** (no class left behind)
- ✅ **Label Stability > 0.95** (convergence)

## 🔑 Key Implementation Details

### 1. L2 Feature Normalization (Critical!)
```python
features = F.normalize(features, p=2, dim=1)
```
- Ensures distance-based clustering focuses on direction, not magnitude
- Applied before every clustering operation

### 2. Hungarian Algorithm for Evaluation
```python
from scipy.optimize import linear_sum_assignment
mapping, accuracy = compute_cluster_to_class_mapping(pseudo_labels, true_labels)
```
- Cluster IDs (0-9) don't match class IDs → need optimal mapping
- Maximizes agreement between predicted clusters and true classes

### 3. GPU-Accelerated Clustering
- **FAISS** (recommended): 10x faster than sklearn for 60K samples
- **Fallback**: sklearn K-means if FAISS unavailable

### 4. Convergence Detection
```python
stability = 1.0 - (changed_labels / total_labels)
if stability > 0.95:
    break  # Pseudo-labels have stabilized
```

## 📊 Outputs

### Checkpoints
- `checkpoints/best_model.pth`: Model with highest test accuracy
- `checkpoints/final_model.pth`: Final model after all iterations
- `checkpoints/model_iter_*.pth`: Per-iteration checkpoints

### Results
- `results/training_history.npy`: Metrics across iterations
- `results/confusion_matrix.png`: Final confusion matrix
- `results/clustering_quality.png`: NMI/Purity over iterations
- `results/training_progress.png`: Comprehensive training plots

## 🛠️ Troubleshooting

### Poor Initial Clustering (Accuracy ~20-30% at Iteration 1)
- **Solution**: Features are random → Normal! Should improve by Iteration 3
- **Fix if persistent**: Implement autoencoder pre-training (in `models/autoencoder.py`)

### Clusters Form by Color/Texture Instead of Shape
- **Solution**: Strengthen data augmentation in `utils/data_loader.py`
- Add color jitter, increase rotation range

### Training Unstable (Loss oscillates)
- **Solution**: Reduce learning rate (0.001 → 0.0005)
- Enable label smoothing: `nn.CrossEntropyLoss(label_smoothing=0.1)`

### Slow Clustering (>5 min per iteration)
- **Solution**: Install FAISS (`pip install faiss-gpu`)
- Or reduce `n_init` in `config/config.yaml` (20 → 10)

## 📚 Fashion-MNIST Classes

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## 🔬 Technical Details

**Why Self-Supervised?**
- Real-world scenarios often lack labeled data
- Demonstrates unsupervised feature learning capability
- Iterative refinement mimics human learning

**Why Iterative Pseudo-Labeling?**
- Single clustering on random features → poor results (~40%)
- Training improves features → better clustering → better features
- Virtuous cycle: each iteration refines both features and labels

**Key Hyperparameters:**
- `feature_dim=256`: Feature vector size
- `n_clusters=10`: Must match number of true classes
- `recluster_interval=10`: Re-cluster every N epochs (within iteration)
- `convergence_threshold=0.95`: Stop when labels stabilize

## 📖 References

- Architecture inspired by: [Deep Clustering](https://arxiv.org/abs/1807.05520)
- Hungarian Algorithm: [scipy.optimize.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
- Fashion-MNIST: [Xiao et al., 2017](https://arxiv.org/abs/1708.07747)

## 📄 License

This project is part of a deep learning research implementation.

## 🙋 Support

For issues or questions:
1. Check `documents/architecture.md` for detailed architecture description
2. Review hyperparameters in `config/config.yaml`
3. Examine training logs in `logs/` directory

---

**Built with PyTorch** | Self-Supervised Learning | Iterative Pseudo-Labeling
