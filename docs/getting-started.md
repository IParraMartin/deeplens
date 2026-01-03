# Getting Started

Welcome to **DeepLens**! This guide will help you get up and running with mechanistic interpretability using Sparse Autoencoders (SAEs).

## What is DeepLens?

DeepLens is a comprehensive library for mechanistic interpretability that provides end-to-end tools for:

- **Feature Extraction**: Extract MLP activations from transformer models
- **SAE Training**: Train sparse autoencoders with TopK or L1 regularization
- **Feature Analysis**: Decode and analyze learned features
- **Feature Intervention**: Manipulate features to understand their causal effects

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- CUDA-compatible GPU (recommended for training)
- Basic knowledge of ```transformers``` and ```PyTorch```

## Installation

### 1. Create a Virtual Environment

I strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
conda create -n deeplens python=3.11
conda activate deeplens
```

### 2. Install DeepLens

Clone the repository and install in editable mode:

```bash
git clone https://github.com/IParraMartin/deeplens.git
cd deeplens
pip install -e .
```

### 3. Install PyTorch

Install PyTorch with CUDA support (if you have a GPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only installation:

```bash
pip install torch torchvision torchaudio
```

### 4. Verify Installation

Test that everything is working:

```python
import deeplens
from deeplens.extractor import ExtractSingleSample
print("DeepLens installed successfully!")
```

## Common Installation Issues

### NumPy Compatibility Error

If you encounter a NumPy 2.x compatibility error with scipy:

```bash
pip install "numpy<2.0" --force-reinstall
```

### Manual Installation

If automatic installation fails, try installing dependencies manually:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Quick Start: Your First Pipeline

Here's a minimal example to get you started with feature intervention:

```python
from deeplens.pipeline import pipeline

# Run end-to-end intervention pipeline
original, decoded, modified = pipeline(
    text="The Eiffel Tower is in",
    sae_model="path/to/your/sae_model.pt",
    sae_config="path/to/config.yaml",
    layer=3,
    hf_model="gpt2",
    feature=-1,
    alpha=5.0,
    generate=True,
    max_new_tokens=20
)

print("Original:", original)
print("Modified:", modified)
```

## Basic Workflow

DeepLens follows a simple four-step workflow:

### 1. Extract Activations

Extract MLP activations from a transformer model:

```python
from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    hf_model="gpt2",
    layer=3,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=1000,
    seq_length=128,
    save_features=True
)

features = extractor.extract_features()
```

### 2. Train a Sparse Autoencoder

Train an SAE on the extracted features:

```python
from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder

# Prepare dataset
dataset = ActivationsDatasetBuilder(
    activations="saved_features.pt",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)
train_loader, eval_loader = dataset.get_dataloaders()

# Configure model
config = SAETrainer.config_from_yaml('config.yaml')
model = SparseAutoencoder(**config)

# Train
trainer = SAETrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    epochs=3
)
trainer.train()
```

### 3. Analyze Features

Decode activations and find active features:

```python
from deeplens.extractor import ExtractSingleSample
from deeplens.intervene import InterveneFeatures

# Extract from single sample
extractor = ExtractSingleSample(hf_model="gpt2", layer=3)
acts = extractor.get_mlp_acts("The capital of France is")

# Get alive features
intervene = InterveneFeatures(
    sae_model="best_model.pt",
    sae_config="config.yaml"
)
features = intervene.get_alive_features(acts, token_position=-1)
print(f"Found {len(features)} active features")
```

### 4. Intervene on Features

Manipulate specific features to test causal effects:

```python
from deeplens.intervene import ReinjectSingleSample

# Intervene on feature
original, decoded, modified = intervene.intervene_feature(
    activations=acts,
    feature=features[0].item(),
    alpha=10.0,
    token_positions=-1
)

# Reinject and generate
reinject = ReinjectSingleSample(hf_model="gpt2")
output = reinject.reinject_and_generate(
    text="The capital of France is",
    modified_activations=modified,
    layer=3,
    generate=True,
    max_new_tokens=10
)
```

## Supported Models

Currently, DeepLens supports these GPT-2 variants:

- `gpt2` (124M parameters)
- `gpt2-medium` (355M parameters)
- `gpt2-large` (774M parameters)
- `gpt2-xl` (1.5B parameters)

## Next Steps

Now that you have DeepLens installed, explore these resources:

- **[Quickstart Guide](quickstart.md)**: Complete example workflows
- **[Tutorials](tutorials/basic.md)**: In-depth guides for each component
- **[API Reference](api/core.md)**: Detailed API documentation
- **[Examples](examples.md)**: Real-world use cases

## Getting Help

If you encounter issues:

1. Check the [documentation](https://github.com/IParraMartin/deeplens)
2. Search existing [GitHub issues](https://github.com/IParraMartin/deeplens/issues)
3. Open a new issue with a **minimal** reproducible example

---

Ready to dive deeper? Check out the [Quickstart Guide](quickstart.md) for complete examples!