# Feature Interventions

This tutorial covers how to perform causal interventions on sparse autoencoder features. By modifying feature activations and observing changes in model behavior, you can establish causal relationships between features and model outputs.

## Overview

DeepLens provides two main classes for feature intervention:

| Class | Description |
|-------|-------------|
| `InterveneFeatures` | Modify feature activations in the SAE latent space |
| `ReinjectSingleSample` | Reinject modified activations back into the model |

The intervention workflow:
1. Extract activations from input text
2. Encode through SAE to get feature activations
3. Modify specific features (amplify, suppress, or ablate)
4. Decode back to activation space
5. Reinject into model to observe effects

---

## Basic Intervention

### Setup

```python
from deeplens.extractor import ExtractSingleSample
from deeplens.intervene import InterveneFeatures, ReinjectSingleSample

# Initialize components
extractor = ExtractSingleSample(hf_model="gpt2", layer=3)
intervene = InterveneFeatures(
    sae_model="models/best_model.pt",
    sae_config="config.yaml"
)
reinject = ReinjectSingleSample(hf_model="gpt2")

# Extract activations
text = "The capital of France is"
activations = extractor.get_mlp_acts(text)
```

### Amplify a Feature

Increase a feature's activation to strengthen its effect:

```python
# Find active features at the last token
active_features = intervene.get_alive_features(activations, token_position=-1)
feature_to_modify = active_features[0].item()

# Amplify the feature by 5x
_, original_acts, modified_acts = intervene.intervene_feature(
    activations=activations,
    feature=feature_to_modify,
    alpha=5.0,  # Multiply by 5
    token_positions=-1  # Only modify last token
)
```

### See the Effect

```python
# Generate with original activations
original_output = reinject.reinject_and_generate(
    text=text,
    modified_activations=original_acts,
    layer=3,
    generate=True,
    max_new_tokens=10
)

# Generate with modified activations
modified_output = reinject.reinject_and_generate(
    text=text,
    modified_activations=modified_acts,
    layer=3,
    generate=True,
    max_new_tokens=10
)

print(f"Original: {original_output}")
print(f"Modified: {modified_output}")
```

---

## Intervention Types

### Amplification (alpha > 1)

Strengthen a feature's influence:

```python
# Amplify by different amounts
for alpha in [2.0, 5.0, 10.0, 20.0]:
    _, _, modified = intervene.intervene_feature(
        activations, feature=0, alpha=alpha, token_positions=-1
    )
    output = reinject.reinject_and_generate(
        text, modified, layer=3, generate=True, max_new_tokens=10
    )
    print(f"Alpha {alpha:5.1f}: {output}")
```

### Suppression (0 < alpha < 1)

Weaken a feature's influence:

```python
# Suppress the feature
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=0.1,  # Reduce to 10%
    token_positions=-1
)
```

### Ablation (alpha = 0)

Completely remove a feature's contribution:

```python
# Zero out the feature
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=0.0,  # Complete ablation
    token_positions=-1
)
```

### Negation (alpha < 0)

Reverse a feature's effect:

```python
# Negate the feature
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=-1.0,  # Flip sign
    token_positions=-1
)
```

---

## Token Position Control

### Single Position

Modify only one token position:

```python
# Modify only the last token
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=5.0,
    token_positions=-1  # Last token only
)
```

### Multiple Positions

Modify several specific positions:

```python
# Modify positions 2, 3, and 4
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=5.0,
    token_positions=[2, 3, 4]
)
```

### All Positions

Apply modification to the entire sequence:

```python
# Modify all tokens
_, _, modified = intervene.intervene_feature(
    activations,
    feature=feature_to_modify,
    alpha=5.0,
    token_positions=None  # All positions
)
```

---

## Comparing Logits

Instead of generating text, compare the model's logit distributions:

```python
# Get logits (not text)
original_logits = reinject.reinject_and_generate(
    text=text,
    modified_activations=original_acts,
    layer=3,
    generate=False  # Return logits instead
)

modified_logits = reinject.reinject_and_generate(
    text=text,
    modified_activations=modified_acts,
    layer=3,
    generate=False
)

print(f"Original logits shape: {original_logits.shape}")
print(f"Modified logits shape: {modified_logits.shape}")
```

---

## Troubleshooting

### No Effect from Intervention

- Feature may not be important for this context
- Try larger alpha values (10-20x)
- Check that the feature is actually active at your chosen position

### Incoherent Generated Text

- Lower alpha values (strong interventions can break coherence)
- Try intervening on fewer positions
- Use lower temperature for generation

### Memory Issues

```python
# Clear CUDA cache between experiments
import torch
torch.cuda.empty_cache()
```

### Feature Index Out of Range

```python
# Check valid range
print(f"Number of features: {intervene.model.encoder.out_features}")
# Feature indices: 0 to (n_features - 1)
```
