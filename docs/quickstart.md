# Quickstart

This guide will walk you through the core workflow of DeepLens in just a few minutes. You'll learn how to extract activations, train a sparse autoencoder (SAE), and perform feature interventions.

## Prerequisites

Make sure you have DeepLens installed. If not, check the [Installation Guide](installation.md).

## The Three-Step Workflow

DeepLens follows a simple three-step pipeline:

1. **Extract activations** from a language model
2. **Train a sparse autoencoder** on those activations
3. **Intervene on features** to understand their causal effects

Let's walk through each step.

---

## Step 1: Extract MLP Activations

First, we'll extract activations from GPT-2's MLP layers using a dataset from HuggingFace. DeepLens handles streaming automatically, so you don't need to download the entire dataset.

```python
from deeplens.extractor import FromHuggingFace

# Extract activations from layer 3 of GPT-2
extractor = FromHuggingFace(
    hf_model="gpt2",
    layer=3,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=50000,
    seq_length=1024,
    inference_batch_size=16,
    device="auto",
    save_features=True  # Saves to 'saved_features' directory
)

# This will take a few minutes depending on your hardware
features = extractor.extract_features()
print(f"Extracted features shape: {features.shape}")
```

**What's happening:**
- We're loading GPT-2 and extracting activations from layer 3's MLP
- Using 50,000 samples from the FineWeb dataset
- Features are automatically saved to disk for the next step

---

## Step 2: Train a Sparse Autoencoder

Now we'll train an SAE to discover interpretable features in those activations.

### 2.1 Create a Configuration File

First, create a `config.yaml` file with your SAE hyperparameters:

```yaml
input_dims: 3072        # GPT-2 layer 3 MLP dimension
n_features: 24576       # 8x expansion factor
activation: 'relu'
input_norm: True
k: 768                  # Top-k sparsity
beta_l1: None           # Use None for TopK, or a value like 0.001 for L1
tie_weights: False
unit_norm_decoder: True
```

### 2.2 Prepare the Dataset

```python
from deeplens.utils.dataset import ActivationsDatasetBuilder

# Load the saved features from Step 1
dataset = ActivationsDatasetBuilder(
    activations="saved_features/gpt2_layer_3_features.pt",  # Your saved file
    splits=[0.8, 0.2],  # 80% train, 20% eval
    batch_size=16,
    norm=True
)

train_loader, eval_loader = dataset.get_dataloaders()
```

### 2.3 Train the Model

```python
from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
import torch

# Load configuration and initialize model
config = SAETrainer.config_from_yaml('config.yaml')
model = SparseAutoencoder(**config)

# Set up optimizer
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=3e-4,
    betas=(0.9, 0.99),
    weight_decay=0  # Use 1e-4 if tie_weights=False
)

# Initialize trainer
trainer = SAETrainer(
    model=model,
    model_name="gpt2_layer3_sae",
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optim=optimizer,
    epochs=3,
    bf16=True,
    random_seed=42,
    save_checkpoints=True,
    device="auto",
    grad_clip_norm=3.0,
    lrs_type='cosine',
    eval_steps=1000,
    warmup_fraction=0.1,
    save_best_only=True,
    log_to_wandb=False  # Set to True if you want W&B logging
)

# Start training
trainer.train()
```

**Training tips:**
- Training will take from minutes to a hours depending on your dataset size and hardware
- The best model checkpoint is automatically saved to `models/`
- Monitor the reconstruction loss and sparsity metrics

---

## Step 3: Intervene on Features

Now comes the fun part! Let's analyze what features the SAE learned and test their causal effects.

### 3.1 Extract Features from a Single Sample

```python
from deeplens.extractor import ExtractSingleSample

# Initialize extractor for the same model and layer
extractor = ExtractSingleSample(hf_model="gpt2", layer=3)

# Extract activations from a specific text
text = "The Eiffel Tower is located in Paris, France."
activations = extractor.get_mlp_acts(text)
```

### 3.2 Find Active Features

```python
from deeplens.intervene import InterveneFeatures

# Load your trained SAE
intervene = InterveneFeatures(
    sae_model="models/gpt2_layer3_sae/best_model.pt",
    sae_config="config.yaml"
)

# Get features active at the last token position
active_features = intervene.get_alive_features(
    activations, 
    token_position=-1
)

print(f"Found {len(active_features)} active features")
print(f"Top features: {active_features[:5]}")
```

### 3.3 Intervene on a Specific Feature

```python
from deeplens.intervene import ReinjectSingleSample

# Initialize reinjection module
reinject = ReinjectSingleSample(hf_model="gpt2")

# Pick a feature to modify (e.g., the most active one)
feature_to_modify = active_features[0].item()

# Intervene by amplifying the feature
_, original_acts, modified_acts = intervene.intervene_feature(
    activations=activations,
    feature=feature_to_modify,
    alpha=5.0,  # Amplify 5x
    token_positions=-1  # Only modify last token
)

# Generate text with the modified activations
output = reinject.reinject_and_generate(
    text=text,
    modified_activations=modified_acts,
    layer=3,
    generate=True,
    max_new_tokens=20
)

print("Original text:", text)
print("Modified continuation:", output)
```

### 3.4 Compare Original vs Modified

```python
# Generate with original activations
original_output = reinject.reinject_and_generate(
    text=text,
    modified_activations=original_acts,
    layer=3,
    generate=True,
    max_new_tokens=20
)

print("Original:", original_output)
print("Modified:", output)
```

---

## Next Steps

Congratulations! You've completed the full DeepLens pipeline. Here's what to explore next:

- **[Feature Analysis](tutorials/analysis.md)** - Learn how to analyze and visualize learned features
- **[Intervention Techniques](tutorials/interventions.md)** - Explore different ways to modify features
- **[Advanced Training](tutorials/sae.md)** - Fine-tune your SAE training for better results
- **[API Reference](api/core.md)** - Dive into the full API documentation

## Common Issues

**Out of memory during extraction:**
- Reduce `inference_batch_size` or `seq_length`
- Use a smaller model like GPT-2 instead of larger variants

**SAE training is slow:**
- Enable `bf16=True` for faster training on modern GPUs
- Reduce `num_samples` or `n_features`
- Ensure you're using GPU with `device="cuda"`

**Features don't seem interpretable:**
- Try different sparsity values (adjust `k` or `beta_l1`)
- Train longer (more epochs)
- Use more training samples
- Experiment with different layers

---

Need more help? Check out the [tutorials](tutorials/extraction.md) or open an issue on GitHub.
