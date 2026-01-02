# DeepLens

An end-to-end library for mechanistic interpretability research on transformer language models.

## What is this?

DeepLens is a comprehensive toolkit that provides everything you need to understand the internal computations of transformer models. From extracting activations to training sparse autoencoders and analyzing learned features, DeepLens offers a complete pipeline for mechanistic interpretability research. Whether you're investigating individual neurons, discovering interpretable features, or running intervention experiments, this library streamlines the entire workflow.

## Key Features

- **Activation Extraction** — Extract and cache any internal activation from transformer models for analysis
- **Dataset Building** — Construct custom datasets from model activations for training and analysis
- **SAE Training** — Train sparse autoencoders (SAEs) from scratch to discover interpretable feature directions
- **Feature Analysis** — Analyze learned features, compute activation patterns, and understand what features represent
- **Feature Interventions** — Scale, ablate, or modify specific features and observe downstream effects on model behavior
- **End-to-End Pipeline** — Seamlessly go from raw model activations to trained SAEs to mechanistic insights

## Quick Example

```python

```

## Installation
Install from ```pip```
```bash
pip install deeplens
```

Or install from source (currently recomended):

```bash
git clone https://github.com/iparramartin/deeplens
cd deeplens
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Getting Started

Check out the [Getting Started guide](getting-started.md) for a walkthrough of the core concepts, or dive into the [tutorials](tutorials/basic.md) for hands-on examples.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{yourlibrary2024,
  author = {Iñigo Parra},
  title = {DeepLens: An End-to-end Tool for Mechanistic Interpretability},
  year = {2026},
  url = {https://github.com/iparramartin/deeplens}
}
```