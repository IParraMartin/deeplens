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

Currently, DeepLens supports:

- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-3.2-1B`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `microsoft/phi-2`
- `microsoft/Phi-3.5-mini-instruct`
- `microsoft/Phi-4-mini-instruct`
- `mistralai/Mistral-7B-v0.1`
- `google/gemma-3-270m`
- `google/gemma-7b-it`
- `tiiuae/falcon-7b`
- `Qwen/Qwen2.5-7B-Instruct`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

If any errors arise, feel free to [open an issue](https://github.com/IParraMartin/deeplens/issues) on GitHub.

## Next Steps

Now that you have DeepLens installed, explore these resources:

- **[Quickstart Guide](quickstart.md)**: Complete example workflows
- **[Tutorials](tutorials/extraction.md)**: In-depth guides for each component
- **[API Reference](api/core.md)**: Detailed API documentation

## Getting Help

If you encounter issues:

1. Check the [documentation](https://github.com/IParraMartin/deeplens)
2. Search existing [GitHub issues](https://github.com/IParraMartin/deeplens/issues)
3. Open a new issue with a **minimal** reproducible example

---

Now let's learn how to [Install](installation.md) DeepLens!