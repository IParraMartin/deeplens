![DeepLens](resources/header.png)

## Overview
__DeepLens__ is a library for mechanistic interpretability. It includes a full set of tools that allow end-to-end interpretability pipelines: from feature extraction, to feature steering. The library includes Sparse Autoencoders (TopK and L1), feature extractors, feature dataset modules, and intervention modules. 

## Quick How To
### Installation
Before installing any dependency, I recommend creating a new virtual envoronment to avoid library conflicts.
```
conda create -n deeplens python=3.11
conda activate deeplens
```

The following command should install the necessary dependencies and tools:
```
pip install -e .
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If any errors arise, you may alternatively use the manual installation:
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### 1. MLP Feature Extraction
```
from deeplens.extractor import FromHuggingFace

extractor = FromHuggingFace(
    model="gpt2",
    layer=3,
    dataset_name="HuggingFaceFW/fineweb", # uses dataset streaming!
    num_samples=500,
    seq_length=1024,
    inference_batch_size=16,
    device="auto",
    save_features=True
)

features = extractor.extract_features()
```

### 2. Training
```
from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder
import torch

dataset = ActivationsDatasetBuilder(
    activations="YOUR_SAVED_FEATURES_PT_FILE",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)
train, eval = dataset.get_dataloaders()

config = SAETrainer.config_from_yaml('demo/config.yaml')
model = SparseAutoencoder(**config)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.0003, 
    betas=(0.9,0.99),
    weight_decay=1e-4 # Just when using untied weights! Else set to 0
)

trainer = SAETrainer(
    model=model,
    model_name="YOUR_GIVEN_MODEL_NAME",
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
    eval_steps=1000,
    warmup_fraction=0.1,
    save_best_only=True,
    log_to_wandb=True
)

trainer.train()
```

### 3. SAE Feature Extraction
```
text = "What color is the car next to Mary's house?"
sample = ExtractSingleSample(
    model="SAVED_MODEL_DIR",
    sample=text,
    layer=3,
    max_length=512,
    device="auto"
)

acts = sample.get_mlp_acts()
```

### 4. Feature Intervention
```
text = "What color is the car next to Mary's house?"
sample = ExtractSingleSample(
    model="SAVED_MODEL_DIR",
    sample=text,
    layer=3,
    max_length=512,
    device="auto"
)

acts = sample.get_mlp_acts()
```

## Currently Supported Models

- gpt2
- gpt2-medium
- gpt2-large
- gpt2-xl