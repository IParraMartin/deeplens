![DeepLens](resources/header.png)

## Overview
__DeepLens__ is a library for mechanistic interpretability. It includes a full set of tools that allow end-to-end interpretability pipelines: from feature extraction, to feature steering. The library includes Sparse Autoencoders (TopK and L1), feature extractors, feature dataset modules, and intervention modules. 

## Tutorial


```
import torch
from src.sae import SparseAutoencoder
from src.train import SAETrainer
from src.utils.dataset import ActivationsDatasetBuilder

dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_512000.pt"
)
train, eval = dataset.get_dataloaders()
config = SAETrainer().config_from_yaml('configurations/test.yaml')
model = SparseAutoencoder(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
print(model)

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