from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder
import torch

dataset = ActivationsDatasetBuilder(
    activations="saved_features/features_layer_3_1171436.pt",
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
    model_name="gpt2L3-untied",
    train_dataloader=train,
    eval_dataloader=eval,
    optim=optimizer,
    epochs=10,
    bf16=False,
    random_seed=42,
    save_checkpoints=True,
    device="auto",
    grad_clip_norm=3.0,
    lrs_type='cosine',
    eval_steps=5000,
    warmup_fraction=0.1,
    save_best_only=True,
    log_to_wandb=True
)

trainer.train()