![DeepLens](resources/header.png)

## Overview
__DeepLens__ is a library for mechanistic interpretability. It includes a full set of tools that allow end-to-end interpretability pipelines: from feature extraction, to feature steering. The library includes Sparse Autoencoders (TopK and L1), feature extractors, feature dataset modules, and intervention modules. 

## Quick How To
### MLP Feature Extraction

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

### Training
```
from src.sae import SparseAutoencoder
from src.train import SAETrainer
from src.utils.dataset import ActivationsDatasetBuilder
import torch

dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_512000.pt"
)
train, eval = dataset.get_dataloaders()

config = SAETrainer().config_from_yaml('configurations/test.yaml')
model = SparseAutoencoder(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

trainer = SAETrainer(
    model=model,
    train_dataloader=train,
    eval_dataloader=eval,
    optim=optimizer,
    epochs=100,
    bf16=False,
    random_seed=42,
    save_checkpoints=True,
    device="auto",
    grad_clip_norm=2.0,
    lrs_type='cosine'
)

trainer.train()
```

### Feature Intervention
```
from sklearn.datasets import load_digits
from deeplens.intervene import InterveneFeatures

intervention = InterveneFeatures(
    sae_model='saved_models/run_20251225_222241/sae_best.pt',
    sae_config='configurations/test.yaml'
)

X, y = load_digits(return_X_y=True)
features = intervention.get_alive_features(X[0])
original, modified = intervention.intervene_feature(
    example=X[0],
    feature=172,
    alpha=5
)
print(features)
print(original.shape)
print(modified.shape)
```

### Note on CUDA
Make sure all torch libraries are compatible. In the most recent version, I used: 
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```