from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.utils.dataset import ActivationsDatasetBuilder
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    # Model config
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Path to YAML config file"
    )
    parser.add_argument(
        '--model_name',
        type=str, 
        default="gpt2L3-untied-2M", 
        help="Name for the model"
    )
    # Dataset
    parser.add_argument(
        '--activations', 
        type=str, 
        required=True, 
        help="Path to activations file"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16, 
        help="Batch size for training"
    )
    # Optimizer
    parser.add_argument(
        '--lr', 
        type=float, 
        default=3e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-5, 
        help="Weight decay (use 0 for tied weights)"
    )
    parser.add_argument(
        '--beta1', 
        type=float, 
        default=0.9, 
        help="Adam beta1"
    )
    parser.add_argument(
        '--beta2', 
        type=float, 
        default=0.99, 
        help="Adam beta2"
    )
    # Training
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help="Number of training epochs"
    )
    parser.add_argument(
        '--bf16', 
        action='store_true', 
        default=True, 
        help="Use bfloat16 precision"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default="auto", 
        help="Device to train on"
    )
    parser.add_argument(
        '--grad_clip_norm', 
        type=float, 
        default=3.0, 
        help="Gradient clipping norm"
    )
    parser.add_argument(
        '--lrs_type', 
        type=str, 
        default='cosine', 
        choices=['cosine', 'linear', 'constant'], 
        help="LR scheduler type"
    )
    parser.add_argument(
        '--eval_steps', 
        type=int, 
        default=8000, 
        help="Evaluation frequency in steps"
    )
    parser.add_argument(
        '--warmup_fraction', 
        type=float, 
        default=0.1, 
        help="Warmup fraction of total steps"
    )
    # Checkpointing & logging
    parser.add_argument(
        '--save_checkpoints', 
        action='store_true', 
        default=True, 
        help="Save checkpoints"
    )
    parser.add_argument(
        '--save_best_only', 
        action='store_true', 
        default=True, 
        help="Save only the best checkpoint"
    )
    parser.add_argument(
        '--log_to_wandb', 
        action='store_true', 
        default=True, 
        help="Log to Weights & Biases"
    )
    parser.add_argument(
        '--no_wandb', 
        action='store_true', 
        help="Disable Weights & Biases logging"
    )
        
    args = parser.parse_args()

    dataset = ActivationsDatasetBuilder(
        activations=args.activations,
        splits=[0.8, 0.2],
        batch_size=args.batch_size,
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