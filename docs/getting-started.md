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

## Basic Workflow

DeepLens follows a simple four-step workflow:

1. Collect Activations
2. Create a Dataset
3. Train a Sparse Autoencoder
4. Analyze Features
5. Intervene on Features


## Currently Supported Models

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

Now let's learn how to [Install](installation.md) DeepLens!