![DeepLens](resources/header.png)

## Overview
__DeepLens__ is a library for mechanistic interpretability. It includes a full set of tools that allow end-to-end interpretability pipelines: from feature extraction, to feature steering. The library includes Sparse Autoencoders (TopK and L1), feature extractors, feature dataset modules, and intervention modules. 

## Quick How To
### Installation
Install the requirements file `` and the recommended Pytorch CUDA versions:
```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
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
    activations="saved_features/features_layer_3_1024000.pt",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)
train, eval = dataset.get_dataloaders()

config = SAETrainer().config_from_yaml('PATH_TO_MODEL_CONFIG')
model = SparseAutoencoder(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9,0.99))

trainer = SAETrainer(
    model=model,
    train_dataloader=train,
    eval_dataloader=eval,
    optim=optimizer,
    epochs=10,
    bf16=False,
    random_seed=42,
    save_checkpoints=True,
    device="cuda",
    grad_clip_norm=3.0,
    lrs_type='cosine',
    eval_steps=5000,
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