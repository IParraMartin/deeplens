# Training Sparse Autoencoders

This tutorial covers how to train sparse autoencoders (SAEs) on extracted activations using DeepLens. SAEs learn to discover interpretable features from neural network representations.

## Overview

DeepLens provides two main classes for SAE training:

| Class | Description |
|-------|-------------|
| `SparseAutoencoder` | The SAE model architecture |
| `SAETrainer` | Training framework with logging, checkpointing, and scheduling |

---

## SAE Architecture

The `SparseAutoencoder` class implements a standard sparse autoencoder with:

- **Encoder**: Projects activations to a higher-dimensional sparse feature space
- **Decoder**: Reconstructs activations from sparse features
- **Sparsity**: Either TopK selection or L1 regularization

```
Input (d_model) → Encoder → Activation → Sparsity → Decoder → Output (d_model)
   768/3072         ×4-8x      ReLU     TopK/L1       ×4-8x      768/3072
```

---

## Basic Training

### Step 1: Create Configuration

Create a `config.yaml` file with your SAE hyperparameters:

```yaml
input_dims: 3072        # GPT-2 MLP intermediate dimension
n_features: 24576       # Number of learned features (8x expansion)
activation: 'relu'      # Activation function
input_norm: True        # Apply LayerNorm to inputs
k: 768                  # TopK sparsity (keep top 768 features)
beta_l1: null           # L1 coefficient (null when using TopK)
tie_weights: False      # Whether to tie encoder/decoder weights
unit_norm_decoder: True # Constrain decoder columns to unit norm
```

### Step 2: Prepare Data

```python
from deeplens.utils.dataset import ActivationsDatasetBuilder

dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_100000.pt",
    splits=[0.8, 0.2],
    batch_size=32,
    norm=True
)

train_loader, eval_loader = dataset.get_dataloaders()
```

### Step 3: Initialize Model and Optimizer

```python
from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
import torch

# Load config and create model
config = SAETrainer.config_from_yaml("config.yaml")
model = SparseAutoencoder(**config)

# Initialize optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.99),
    weight_decay=0
)
```

### Step 4: Train

```python
trainer = SAETrainer(
    model=model,
    model_name="gpt2_layer3_sae",
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optim=optimizer,
    epochs=3,
    bf16=True,
    device="auto",
    save_checkpoints=True,
    eval_steps=5000,
    log_to_wandb=False
)

trainer.train()
```

---

## Configuration Deep Dive

### Input Dimensions

Match `input_dims` to your model's MLP intermediate size:

| Model | Layer MLP Dimension |
|-------|-------------------|
| GPT-2 | 3072 |
| GPT-2 Medium | 4096 |
| GPT-2 Large | 5120 |
| GPT-2 XL | 6400 |

### Feature Expansion

The ratio `n_features / input_dims` is your **expansion factor**:

```python
# Expansion factor examples
n_features = 8 * input_dims   # 8x expansion (common choice)
n_features = 4 * input_dims   # 4x expansion (fewer features)
n_features = 16 * input_dims  # 16x expansion (more features)
```

**Recommendations:**

- Start with 8x expansion
- Use 4x for faster training/smaller models
- Use 16x for richer feature spaces (requires more data)

### Sparsity Methods

DeepLens supports two sparsity approaches:

#### TopK Sparsity (Recommended)

Keeps only the k largest activations per sample:

```yaml
k: 768         # Keep top 768 features active
beta_l1: null  # Disable L1
```

**Choosing k:**

- Good starting point: ~3% of active features
- Lower k = sparser, more interpretable features
- Higher k = better reconstruction, less sparse

```python
# TopK configuration
model = SparseAutoencoder(
    input_dims=3072,
    n_features=24576,
    k=768,            # ~3% of features active
    beta_l1=None
)
```

#### L1 Regularization

Adds L1 penalty to encourage sparsity:

```yaml
k: None        # Disable TopK
beta_l1: 0.001 # L1 coefficient
```

```python
# L1 configuration
model = SparseAutoencoder(
    input_dims=3072,
    n_features=24576,
    k=None,
    beta_l1=0.001
)
```

**Tuning beta_l1:**

- Too high → features become too sparse, poor reconstruction
- Too low → features not sparse enough
- Start with 0.001 and adjust based on reconstruction loss and sparsity

### Weight Tying

Controls whether decoder weights are tied to encoder weights:

```python
# Untied weights (recommended)
model = SparseAutoencoder(
    tie_weights=False,
    unit_norm_decoder=True
)

# Tied weights (fewer parameters)
model = SparseAutoencoder(
    tie_weights=True,
    unit_norm_decoder=False  # No separate decoder to normalize
)
```

**Recommendations:**

- Use `tie_weights=False` for better reconstruction
- Use `tie_weights=True` to reduce parameters (faster training)

### Activation Function

```python
# ReLU (standard, recommended)
model = SparseAutoencoder(activation='relu')

# SiLU (may improve reconstruction)
model = SparseAutoencoder(activation='silu')
```

---

## Trainer Configuration

### Basic Trainer Setup

```python
trainer = SAETrainer(
    model=model,
    model_name="my_sae",
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optim=optimizer,
    epochs=3,
    device="auto"
)
```

### All Trainer Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | SparseAutoencoder instance | None |
| `model_name` | Name for saving checkpoints | "sae" |
| `train_dataloader` | Training data DataLoader | None |
| `eval_dataloader` | Evaluation data DataLoader | None |
| `optim` | Optimizer instance | Adam |
| `epochs` | Number of training epochs | 20 |
| `bf16` | Enable bfloat16 mixed precision | False |
| `random_seed` | Seed for reproducibility | 42 |
| `save_checkpoints` | Save model checkpoints | True |
| `device` | Training device | "auto" |
| `grad_clip_norm` | Gradient clipping threshold | None |
| `lrs_type` | LR scheduler type | None |
| `eval_steps` | Steps between evaluations | 5000 |
| `warmup_fraction` | Warmup fraction for LR | 0.1 |
| `save_best_only` | Only save best checkpoint | True |
| `log_to_wandb` | Enable W&B logging | True |

### Learning Rate Scheduling

DeepLens supports three learning rate schedulers:

```python
# Cosine annealing with warmup (recommended)
trainer = SAETrainer(
    lrs_type='cosine',
    warmup_fraction=0.1  # 10% warmup
)

# Linear decay with warmup
trainer = SAETrainer(
    lrs_type='linear',
    warmup_fraction=0.1
)

# Reduce on plateau
trainer = SAETrainer(
    lrs_type='plateau'  # Reduces LR when loss plateaus
)
```

### Gradient Clipping

Prevent gradient explosion:

```python
trainer = SAETrainer(
    grad_clip_norm=3.0  # Clip gradients to max norm of 3.0
)
```

### Mixed Precision Training

Enable bfloat16 for faster training on modern GPUs:

```python
trainer = SAETrainer(
    bf16=True  # Requires CUDA with bf16 support
)
```

---

## Monitoring Training

### Console Output

Training progress is printed every 100 steps:

```
Step [0/1000] - train_loss: 0.245 - train_nz_frac: 0.032 - lr: 3.00e-04
Step [100/1000] - train_loss: 0.189 - train_nz_frac: 0.031 - lr: 3.00e-04
```

Key metrics:

- **train_loss**: Reconstruction MSE (lower is better)
- **train_nz_frac**: Fraction of non-zero features (sparsity indicator)
- **lr**: Current learning rate

### Weights & Biases

Enable W&B for detailed tracking:

```python
trainer = SAETrainer(
    log_to_wandb=True
)
```

Logged metrics:

- `train/loss`: Training reconstruction loss
- `train/non_zero_frac`: Sparsity level
- `train/lr`: Learning rate
- `eval/loss`: Evaluation loss

### Interpreting Metrics

**Good training signs:**

- Reconstruction loss decreases steadily
- Non-zero fraction stays around target sparsity
- Evaluation loss tracks training loss (no overfitting)

**Warning signs:**

- Loss increases or oscillates wildly → reduce learning rate
- Non-zero fraction approaches 1.0 → increase sparsity (lower k or higher L1)
- Non-zero fraction near 0 → decrease sparsity (higher k or lower L1)

---

## Checkpointing

### Automatic Checkpointing

Models are saved when evaluation loss improves:

```python
trainer = SAETrainer(
    save_checkpoints=True,
    save_best_only=True,  # Only keep best model
    eval_steps=5000       # Evaluate every 5000 steps
)
```

Checkpoints saved to: `saved_models/{model_name}/run_{timestamp}/best_model.pt`

---

## Advanced Training

### Longer Training

For more training data, increase epochs:

```python
trainer = SAETrainer(
    epochs=10,
    eval_steps=10000  # Adjust eval frequency
)
```

---

## Model Architecture Details

### Forward Pass

```python
# The model's forward pass
def forward(self, x):
    z_pre = self.encode(x)       # Encode to latent space
    z = self.topk_mask(z_pre)    # Apply sparsity (if using TopK)
    x_hat = self.decode(z)       # Decode back
    return x_hat, z, z_pre
```

### Loss Function

```python
# The model's loss computation
loss = MSE(x_hat, x) + beta_l1 * L1(z)  # L1 mode
# or
loss = MSE(x_hat, x)                     # TopK mode
```

### Unit Norm Decoder

The decoder weights are normalized after each step:

```python
# After each optimizer step
model.post_step()  # Renormalizes decoder columns to unit norm
```

This is automatically handled by `SAETrainer`.

---

## Full Training Script

Here's a complete training script:

```python
from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder
import torch

# 1. Load data
dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_1171436.pt",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)
train, eval = dataset.get_dataloaders(ddp=False)

config = SAETrainer.config_from_yaml('demo/config.yaml')
model = SparseAutoencoder(**config)

# 3. Setup optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.0003, 
    betas=(0.9,0.99),
    weight_decay=1e-4 # Just when using untied weights! Else set to 0
)

# 4. Train
trainer = SAETrainer(
    model=model,
    model_name="gpt2_layer3_sae",
    train_dataloader=train,
    eval_dataloader=eval,
    optim=optimizer,
    epochs=3,
    bf16=True,
    random_seed=42,
    save_checkpoints=True,
    device="auto",
    grad_clip_norm=3.0,
    lrs_type='cosine',
    eval_steps=5000,
    warmup_fraction=0.1,
    save_best_only=True,
    log_to_wandb=True
)

trainer.train()
```

---

## Next Steps

After training your SAE:

1. **[Feature Analysis](analysis.md)** - Analyze and visualize learned features
2. **[Feature Interventions](interventions.md)** - Test causal effects of features

---

## Troubleshooting

### Loss Not Decreasing

- Lower learning rate
- Check data normalization
- Increase model capacity (more features)

### Out of Memory

- Reduce batch size
- Reduce `n_features`
- Enable `bf16=True`

### NaN Loss

- Enable gradient clipping: `grad_clip_norm=1.0`
- Lower learning rate
- Check input data for NaN/Inf values

### Poor Reconstruction

- Increase `k` (more active features)
- Decrease `beta_l1`
- Train longer (more epochs)
- Use more training data
