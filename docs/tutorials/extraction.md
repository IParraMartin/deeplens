# Activation Extraction

This tutorial covers how to extract MLP activations from transformer models using DeepLens. Activation extraction is the first step in the mechanistic interpretability pipeline—you need to collect the neural network's internal representations before you can train a sparse autoencoder to discover interpretable features.

## Overview

DeepLens provides two main classes for extracting activations:

| Class | Use Case |
|-------|----------|
| `FromHuggingFace` | Extract activations from large datasets for SAE training |
| `ExtractSingleSample` | Extract activations from individual texts for analysis |

---

## Extracting from Large Datasets

Use `FromHuggingFace` when you need to collect many activations for training a sparse autoencoder. This class streams data from HuggingFace datasets, so you don't need to download the entire dataset upfront.

### Basic Usage

```python
from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    hf_model="gpt2",
    layer=3,
    dataset_name="HuggingFaceFW/fineweb",
    num_samples=25000,
    seq_length=1024,
    inference_batch_size=16,
    device="auto",
    save_features=True
)

features = extractor.extract_features()
print(f"Shape: {features.shape}")  # (total_tokens, hidden_dim)
# Saved to: saved_features/features_layer_3_XXXXX.pt
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hf_model` | HuggingFace model identifier (e.g., "gpt2", "gpt2-medium") | "gpt2" |
| `layer` | Transformer layer index to extract from (0-indexed) | 6 |
| `dataset_name` | HuggingFace dataset to stream | "HuggingFaceFW/fineweb" |
| `num_samples` | Number of text samples to process | 25000 |
| `seq_length` | Maximum sequence length for tokenization | 128 |
| `inference_batch_size` | Batch size for model inference | 16 |
| `device` | "auto", "cuda", "mps", or "cpu" | "auto" |
| `save_features` | Whether to save extracted features to disk | True |

### Choosing the Right Layer

Different layers capture different types of features:

- **Early layers (0-3)**: Low-level features like token identity and basic syntax
- **Middle layers (4-8)**: Compositional features and semantic relationships
- **Late layers (9-11)**: Task-specific and high-level abstract features

### Memory Optimization

If you're running out of memory, try these strategies:

```python
# Reduce batch size for lower memory usage
extractor = FromHuggingFace(
    hf_model="gpt2",
    inference_batch_size=4,  # Lower batch size
    seq_length=512,          # Shorter sequences
    num_samples=2500         # Collect less features
)
```

### Custom Datasets

You can use any HuggingFace dataset with a `text` field:

```python
# Use a different dataset
extractor = FromHuggingFace(
    hf_model="gpt2",
    dataset_name="wikitext/wikitext-103-v1",
    num_samples=1000
)
```

---

## Extracting from Single Samples

Use `ExtractSingleSample` when you want to analyze specific texts, such as when testing feature interventions or debugging.

### Basic Usage

```python
from deeplens.extractor import ExtractSingleSample

extractor = ExtractSingleSample(
    hf_model="gpt2",
    layer=3,
    max_length=1024,
    device="auto"
)

# Extract activations from a specific text
text = "The capital of France is Paris."
activations = extractor.get_mlp_acts(text)
print(f"Shape: {activations.shape}")  # (seq_length, hidden_dim)
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hf_model` | HuggingFace model identifier | "gpt2" |
| `layer` | Transformer layer to extract from | 3 |
| `max_length` | Maximum sequence length | 1024 |
| `device` | "auto", "cuda", "mps", or "cpu" | "auto" |

### Analyzing Token-Level Activations

Each position in the output corresponds to a token:

```python
extractor = ExtractSingleSample(hf_model="gpt2", layer=3)

text = "Hello world"
activations = extractor.get_mlp_acts(text)

# Access tokenizer to see which tokens correspond to which positions
tokens = extractor.tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"Activation shape: {activations.shape}")

# activations[0] = activation for "Hello"
# activations[1] = activation for " world"
```

---

## Next Steps

Now that you have extracted activations, you can:

1. **[Create a Dataset](dataset.md)** - Prepare your activations for training
2. **[Train a Sparse Autoencoder](sae.md)** - Discover interpretable features
3. **[Analyze Features](analysis.md)** - Visualize and understand what you've learned

---

## Troubleshooting

### Out of Memory

```python
# Solutions:
# 1. Reduce batch size
extractor = FromHuggingFace(inference_batch_size=4)
# 2. Use shorter sequences
extractor = FromHuggingFace(seq_length=128)
# 3. Process fewer samples
extractor = FromHuggingFace(num_samples=1000)
```

### Slow Extraction

```python
# Use GPU if available
extractor = FromHuggingFace(device="cuda")
# Increase batch size if you have memory headroom
extractor = FromHuggingFace(inference_batch_size=32)
```

### Dataset Not Found

Make sure the dataset exists on HuggingFace and has a `text` field:

```python
from datasets import load_dataset

# Check dataset structure
ds = load_dataset("your-dataset", split="train", streaming=True)
sample = next(iter(ds))
print(sample.keys())  # Should include 'text'
```
