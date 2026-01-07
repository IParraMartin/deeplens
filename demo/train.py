from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder
import torch
import argparse

<<<<<<< HEAD
=======
dataset = ActivationsDatasetBuilder(
    activations=r"C:\code\deeplens\saved_features\features_layer_-1_1171436.pt",
    splits=[0.8, 0.2],
    batch_size=16,
    norm=True
)
train, eval = dataset.get_dataloaders(ddp=False)

    config = SAETrainer.config_from_yaml(args.config)
    model = SparseAutoencoder(**config)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

trainer = SAETrainer(
    model=model,
    model_name="gpt2LL-untied-1.1M",
    train_dataloader=train,
    eval_dataloader=eval,
    optim=optimizer,
    epochs=10,
    bf16=True,
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
>>>>>>> origin/main

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    trainer = SAETrainer(
        model=model,
        model_name=args.model_name,
        train_dataloader=train,
        eval_dataloader=eval,
        optim=optimizer,
        epochs=args.epochs,
        bf16=args.bf16,
        random_seed=args.seed,
        save_checkpoints=args.save_checkpoints,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        lrs_type=args.lrs_type,
        eval_steps=args.eval_steps,
        warmup_fraction=args.warmup_fraction,
        save_best_only=args.save_best_only,
        log_to_wandb=not args.no_wandb
    )

    trainer.train()