# Dataset Preparation

This tutorial covers how to prepare your extracted activations for training a sparse autoencoder. Proper dataset preparation is crucial for effective SAE training.

## Overview

DeepLens provides utilities for loading, normalizing, and creating DataLoaders from activation tensors:

| Class | Description |
|-------|-------------|
| `ActivationsDatasetBuilder` | Main class for loading activations and creating train/eval DataLoaders |
| `GetDataLoaders` | General-purpose DataLoader factory for any PyTorch Dataset |
| `AudioDatasetBuilder` | Specialized Dataset for audio processing (advanced use) |

---

## Loading Activations

### From Saved Feature Files

After extracting features with `FromHuggingFace`, load them for training:

```python
from deeplens.utils.dataset import ActivationsDatasetBuilder

# Load saved activations
dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_100000.pt",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)

# Create DataLoaders
train_loader, eval_loader = dataset.get_dataloaders()

print(f"Training batches: {len(train_loader)}")
print(f"Evaluation batches: {len(eval_loader)}")
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `activations` | Path to `.pt` file containing activation tensors | None |
| `splits` | Train/validation split proportions (must sum to 1.0) | [0.8, 0.2] |
| `batch_size` | Number of samples per batch | 16 |
| `norm` | Whether to apply z-score normalization | True |

---

## Normalization

Normalization is crucial for stable SAE training. By default, `ActivationsDatasetBuilder` applies z-score normalization (standardization).

### What Normalization Does

```python
# x_normalized = (x - mean) / (std + epsilon)
```

This ensures:

- Zero mean across features
- Unit variance across features
- Stable gradients during training

### When to Disable Normalization

In most cases, keep `norm=True`. Disable it only if:
- Your activations are already normalized
- You want to preserve the original scale for specific analysis

---

## Choosing Batch Size

Batch size affects both training speed and quality:

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (8-16) | Lower memory, more gradient updates | Slower training, noisier gradients |
| Medium (32-64) | Good balance | - |
| Large (128-256) | Faster training, smoother gradients | Higher memory usage |

### Recommendations

```python
# For most GPUs (8-16GB VRAM)
dataset = ActivationsDatasetBuilder(
    activations="path/to/features.pt",
    batch_size=32
)

# For limited memory
dataset = ActivationsDatasetBuilder(
    activations="path/to/features.pt",
    batch_size=8
)

# For large GPUs (40GB+ VRAM)
dataset = ActivationsDatasetBuilder(
    activations="path/to/features.pt",
    batch_size=128
)
```

---

## Train/Validation Splits

The `splits` parameter controls how data is divided:

```python
dataset = ActivationsDatasetBuilder(
    activations="path/to/features.pt",
    splits=[0.8, 0.2]
)
```

**Note**: Splits must sum to 1.0.

---

## DataLoader Best Practices

### Pin Memory

Enable `pin_memory` for faster GPU transfers (automatically enabled by `ActivationsDatasetBuilder`):

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True  # Faster CPU→GPU transfer
)
```

---

## Troubleshooting

### Out of Memory When Loading

```python
# Load with memory mapping
features = torch.load("path/to/large_features.pt", mmap=True)

# Or load a subset
features = torch.load("path/to/features.pt")
features = features[:50000]  # Use first 50k samples
```

---

## Next Steps

Now that your dataset is ready:

1. **[Train a Sparse Autoencoder](sae.md)** - Train your SAE on the prepared data
2. **[Analyze Features](analysis.md)** - Understand what your SAE learned
3. **[Feature Interventions](interventions.md)** - Test causal effects of features