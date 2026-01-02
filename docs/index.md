# DeepLens

![DeepLens](assets/header.png)

An end-to-end library for mechanistic interpretability research on transformer language models.

## What is this?

DeepLens is a comprehensive toolkit that provides everything you need to understand the internal computations of transformer models. From extracting activations to training sparse autoencoders and analyzing learned features, DeepLens offers a complete pipeline for mechanistic interpretability research. Whether you're investigating individual neurons, discovering interpretable features, or running intervention experiments, this library streamlines the entire workflow.

It includes a full set of tools that allow end-to-end interpretability pipelines: from feature extraction, to feature steering. The library includes Sparse Autoencoders (TopK and L1), feature extractors, feature dataset modules, and intervention modules. 

## Key Features

- **Activation Extraction** — Extract and cache any internal activation from transformer models for analysis
- **Dataset Building** — Construct custom datasets from model activations for training and analysis
- **SAE Training** — Train sparse autoencoders (SAEs) from scratch to discover interpretable feature directions
- **Feature Analysis** — Analyze learned features, compute activation patterns, and understand what features represent
- **Feature Interventions** — Scale, ablate, or modify specific features and observe downstream effects on model behavior
- **End-to-End Pipeline** — Seamlessly go from raw model activations to trained SAEs to mechanistic insights

## Quick Example

```python
from deeplens.extractor import ExtractSingleSample
from deeplens.intervene import InterveneFeatures, ReinjectSingleSample
from deeplens.utils.analysis import plot_topk_distribution, get_top_k_tokens

HF_MODEL = "gpt2"
SAE_MODEL_PT_FILE = "yourfile.pt"
SAE_CONFIG_YAML_FILE = "yourfile.yaml"
LAYER = -1
TOKEN_POSITION = -1
ALPHA = 100.0
TEXT = "Hellow world!"

extractor = ExtractSingleSample(hf_model=HF_MODEL, layer=LAYER)
intervene = InterveneFeatures(sae_model=SAE_MODEL_PT_FILE, sae_config=SAE_CONFIG_YAML_FILE)
reinject = ReinjectSingleSample(hf_model=HF_MODEL)

# Extract the activations
acts = extractor.get_mlp_acts(TEXT)

# Get alive features from the extracted activations
features = intervene.get_alive_features(acts, token_position=TOKEN_POSITION)
print(f"{len(features)} alive features discovered at position {TOKEN_POSITION}.")

# intervene on the features and return the modified activations for the 
# selected token position
print(f"Modifying feature {features[0].item()}")

# Modify the selected feature
_, _, modified_decoded = intervene.intervene_feature(
    activations=acts, 
    feature=features[0].item(),
    alpha=ALPHA, 
    token_positions=TOKEN_POSITION
)

# Compute the output logits
out = reinject.reinject_and_generate(
      text=TEXT,
      modified_activations=modified_decoded,
      layer=LAYER,
      generate=False
    )

```

## Installation

Install from source:

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