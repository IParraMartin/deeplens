# DeepLens

A library for mechanistic interpretability research on transformer language models.

## What is this?

This library provides tools for understanding the internal computations of transformer models through techniques like sparse autoencoders (SAEs), activation patching, and feature analysis.

## Key Features

- **Activation Extraction** — Cache and inspect any internal activation in the model
- **Feature Interventions** — Scale, ablate, or modify specific features and observe effects
- **SAE Integration** — Work with sparse autoencoders to find interpretable directions
- **Easy Experimentation** — Designed for fast iteration and exploratory research

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