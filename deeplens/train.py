import os
import yaml
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import wandb

from deeplens.utils.tools import get_device


__all__ = [
    "SAETrainer"
]


class SAETrainer():
    """Training framework for Sparse Autoencoders with comprehensive logging and checkpointing.

    Handles the complete training loop including optimization, learning rate scheduling,
    gradient clipping, mixed precision training (bfloat16), periodic evaluation, model
    checkpointing, and Weights & Biases logging. Designed specifically for training
    sparse autoencoders on neural network activation data.
    """
    def __init__(
            self, 
            train_dataloader: DataLoader = None, 
            eval_dataloader: DataLoader = None, 
            model: torch.nn.Module = None, 
            model_name: str = "sae",
            optim: torch.optim.Optimizer = torch.optim.Adam,
            epochs: int = 20, 
            bf16: bool = False,
            random_seed: int = 42,
            save_checkpoints: bool = True,
            device: str = "auto",
            grad_clip_norm: float = None,
            lrs_type: str = None,
            eval_steps: int = 5000,
            warmup_fraction: float = 0.1,
            save_best_only: bool = True,
            log_to_wandb: bool = True
        ) -> None:
        """Initialize the SAETrainer with model, data, and training configuration.

        Sets up the training environment including device placement, learning rate scheduling,
        and Weights & Biases logging if enabled.

        Args:
            train_dataloader (DataLoader, optional): DataLoader for training data containing
                batches of activation tensors. Defaults to None.
            eval_dataloader (DataLoader, optional): DataLoader for evaluation/validation data.
                Used for periodic evaluation during training. Defaults to None.
            model (torch.nn.Module, optional): Sparse autoencoder model to train. Should be
                an instance of SparseAutoencoder or compatible architecture. Defaults to None.
            model_name (str, optional): Name used for organizing saved checkpoints and W&B
                project naming. Creates directory structure at saved_models/{model_name}/.
                Defaults to "sae".
            optim (torch.optim.Optimizer, optional): Initialized optimizer instance with
                model parameters already attached. Defaults to torch.optim.Adam.
            epochs (int, optional): Number of complete passes through the training dataset.
                Defaults to 20.
            bf16 (bool, optional): Whether to use bfloat16 mixed precision training for
                faster computation and reduced memory usage. Requires CUDA support.
                Defaults to False.
            random_seed (int, optional): Random seed for reproducibility across numpy,
                PyTorch, and CUDNN. Defaults to 42.
            save_checkpoints (bool, optional): Whether to save model checkpoints during
                training. Checkpoints saved when evaluation loss improves. Defaults to True.
            device (str, optional): Device for training. Can be "auto" for automatic selection
                (prefers cuda > mps > cpu), "cuda", "mps", or "cpu". Defaults to "auto".
            grad_clip_norm (float, optional): Maximum gradient norm for gradient clipping.
                Helps prevent gradient explosion. If None, no clipping is applied.
                Defaults to None.
            lrs_type (str, optional): Learning rate scheduler type. Options: 'cosine',
                'plateau', 'linear'. If None, uses constant learning rate. Defaults to None.
            eval_steps (int, optional): Evaluate model every N training steps and save
                checkpoint if loss improves. Defaults to 5000.
            warmup_fraction (float, optional): Fraction of total training steps to use for
                learning rate warmup. Applies to 'cosine' and 'linear' schedulers.
                Defaults to 0.1 (10% warmup).
            save_best_only (bool, optional): If True, only saves the best checkpoint
                (overwrites previous best). If False, saves all improving checkpoints.
                Defaults to True.
            log_to_wandb (bool, optional): Whether to log metrics to Weights & Biases.
                Logs training/eval loss, non-zero fraction, learning rate, and config.
                Defaults to True.

        Note:
            The optimizer must be initialized with the model's parameters before passing
            to the trainer. Learning rate and other optimizer settings should be configured
            in the optimizer instance.
        """
        self.model = model
        self.model_name = model_name
        self.optim = optim
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.bf16 = bf16
        self.random_seed = random_seed
        self.save_checkpoints = save_checkpoints
        self.grad_clip_norm = grad_clip_norm
        self.eval_steps = eval_steps
        self.warmup_fraction = warmup_fraction
        self.save_best_only = save_best_only
        self.log_wandb = log_to_wandb

        self.device = get_device(device)
        print(f"Running on device: {self.device}")

        if lrs_type is not None:
            self.scheduler = self.set_lr_scheduler(lrs_type)
        else:
            self.scheduler = None

        if log_to_wandb:
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.init(
                project=f"sparse-autoencoder",
                name=f"run-{self.model_name}-{time}",
                config={
                    "epochs": epochs,
                    "in_dims": self.model.encoder.in_features,
                    "sparse_dims": self.model.encoder.out_features,
                    "activations": self.model.activation,
                    "input_norm": self.model.input_norm,
                    "top_k": self.model.k, 
                    "l1": self.model.beta_l1,
                    "tie_weights": self.model.tie_weights,
                    "unit_norm_decoder": self.model.unit_norm_decoder,
                    "lr": optim.param_groups[0]['lr'],
                    "lr_scheduler": lrs_type,
                    "warmup_fraction": warmup_fraction,
                    "seed": random_seed,
                    "grad_clip_norm": grad_clip_norm,
                    "bf16": bf16
                }
            )

    def train_one_epoch(
            self, 
            model: torch.nn.Module, 
            train_dataloader: torch.utils.data.DataLoader, 
            optim: torch.optim.Optimizer, 
            bf16: bool = True,
            global_step: int = 0,
            timestamp: str = None,
            best_loss: float = float('inf')
        ) -> tuple[int, float]:
        """Execute one complete training epoch with periodic evaluation and checkpointing.

        Iterates through all training batches, performs forward/backward passes with optional
        mixed precision, applies gradient clipping, updates learning rate, and periodically
        evaluates and saves checkpoints.

        Args:
            model (torch.nn.Module): The sparse autoencoder model to train.
            train_dataloader (torch.utils.data.DataLoader): DataLoader providing training batches.
            optim (torch.optim.Optimizer): Optimizer for updating model parameters.
            bf16 (bool, optional): Whether to use bfloat16 mixed precision. Defaults to True.
            global_step (int, optional): Current global training step count across all epochs.
                Used for logging and checkpoint naming. Defaults to 0.
            timestamp (str, optional): Timestamp string for checkpoint directory naming.
                Format: "YYYYMMDD_HHMMSS". Defaults to None.
            best_loss (float, optional): Best evaluation loss achieved so far. Used to
                determine when to save new checkpoints. Defaults to float('inf').

        Returns:
            tuple[int, float]: A tuple containing:
                - global_step: Updated global step count after this epoch
                - best_loss: Updated best evaluation loss (may be unchanged)

        Note:
            Prints training metrics every 100 steps. Evaluates every eval_steps and saves
            checkpoints when evaluation loss improves. Logs to W&B if enabled.
        """
        model.train()

        if bf16:
            scaler = torch.amp.GradScaler("cuda")
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        for idx, inputs in enumerate(train_dataloader):
            optim.zero_grad()
            if torch.cuda.is_available() and bf16:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    inputs = inputs.to(self.device)
                    loss, logs = model.loss(inputs)
                scaler.scale(loss).backward()

                if self.grad_clip_norm is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_clip_norm
                    )
                
                scaler.step(optim)
                scaler.update()

            else:
                inputs = inputs.to(self.device)
                loss, logs = model.loss(inputs)

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_clip_norm
                    )
                
                loss.backward()
                optim.step()

            if self.scheduler is not None and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            model.post_step()
            global_step += 1

            if self.log_wandb:
                wandb.log({
                    "train/loss": logs['mse'].item(),
                    "train/non_zero_frac": logs['non_zero_frac'].item(),
                    "train/lr": self.optim.param_groups[0]['lr'],
                    "global_step": global_step
                }, step=global_step)

            if (idx % 100) == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(
                    f"Step [{idx}/{len(train_dataloader)}] - "
                    f"train_loss: {round(logs['mse'].item(), 3)} - "
                    f"train_nz_frac: {round(logs['non_zero_frac'].item(), 3)} - "
                    f"lr: {current_lr:.2e}"
                )

            if global_step % self.eval_steps == 0:
                print(f"\n{'='*60}")
                print(f"Intermediate Evaluation at step {global_step}")
                print(f"{'='*60}")
                eval_loss = self.evaluate(
                    model=model,
                    eval_dataloader=self.eval_dataloader,
                    bf16=bf16
                )

                if self.log_wandb:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "global_step": global_step
                    }, step=global_step)
                
                if self.save_checkpoints and eval_loss < best_loss:
                    if self.save_best_only:
                        save_path = f"saved_models/{self.model_name}/run_{timestamp}/best_model.pt"
                    else:
                        save_path = f"saved_models/{self.model_name}/run_{timestamp}/sae_step_{global_step}.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved (loss: {eval_loss:.6f})")
                    best_loss = eval_loss
                
                model.train()
        
        return global_step, best_loss

    @torch.no_grad()
    def evaluate(
            self,
            model: torch.nn.Module, 
            eval_dataloader: torch.utils.data.DataLoader, 
            bf16: bool = True
        ) -> float:
        """Evaluate the model on the validation dataset without gradient computation.

        Computes average loss across all evaluation batches with optional mixed precision.
        Useful for monitoring generalization and preventing overfitting.

        Args:
            model (torch.nn.Module): The sparse autoencoder model to evaluate.
            eval_dataloader (torch.utils.data.DataLoader): DataLoader providing evaluation batches.
            bf16 (bool, optional): Whether to use bfloat16 mixed precision for evaluation.
                Should match training setting. Defaults to True.

        Returns:
            float: Average evaluation loss across all batches. Computed as total loss
                divided by number of batches.

        Note:
            Prints evaluation metrics every 100 steps. Sets model to eval mode and
            restores training mode afterward if called during training.
        """
        model.eval()
        n_batches = 0
        total_loss = 0.0

        if bf16:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        for idx, inputs in enumerate(eval_dataloader):
            if torch.cuda.is_available() and bf16:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    inputs = inputs.to(self.device)
                    loss, logs = model.loss(inputs)
            else:
                inputs = inputs.to(self.device)
                loss, logs = model.loss(inputs)

            total_loss += loss.item()
            n_batches += 1

            if (idx % 100) == 0:
                current_lr = self.optim.param_groups[0]['lr']
                print(
                    f"Step [{idx}/{len(eval_dataloader)}] - "
                    f"eval_loss: {round(logs['mse'].item(), 3)} - "
                    f"eval_nz_frac: {round(logs['non_zero_frac'].item(), 3)} - "
                    f"lr: {current_lr:.2e}"
                )

        avg_loss = total_loss / n_batches
        print(f"Avg loss: {avg_loss:.3f}")
        return avg_loss
    
    def train(self) -> None:
        """Execute the complete training loop for all epochs.

        Main training orchestrator that:
        1. Sets random seed for reproducibility
        2. Moves model to appropriate device
        3. Creates checkpoint directories if saving enabled
        4. Runs training epochs with periodic evaluation
        5. Saves best models based on evaluation loss
        6. Applies learning rate scheduling
        7. Logs metrics to W&B if enabled
        8. Finalizes W&B run on completion

        Note:
            Checkpoints saved to saved_models/{model_name}/run_{timestamp}/.
            Each epoch includes full pass through training data followed by evaluation.
            Best model determined by lowest evaluation loss.
        """
        self.set_seed(self.random_seed)
        self.model.to(self.device)

        if self.save_checkpoints:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(f"saved_models/{self.model_name}/run_{timestamp}", exist_ok=True)

        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            global_step, best_loss = self.train_one_epoch(
                model=self.model, 
                train_dataloader=self.train_dataloader, 
                optim=self.optim, 
                bf16=self.bf16,
                global_step=global_step,
                timestamp=timestamp,
                best_loss=best_loss
            )

            print(f"\n{'='*60}")
            print(f"End of epoch {epoch+1} evaluation")
            print(f"{'='*60}")
            loss = self.evaluate(
                model=self.model,
                eval_dataloader=self.eval_dataloader,
                bf16=self.bf16
            )

            if self.log_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "eval/epoch_loss": loss,
                }, step=global_step)
            
            if self.save_checkpoints and loss < best_loss:
                if self.save_best_only:
                    save_path = f"saved_models/{self.model_name}/run_{timestamp}/best_model.pt"
                else:
                    save_path = f"saved_models/{self.model_name}/run_{timestamp}/sae_epoch_{epoch+1}.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved (loss: {loss:.3f})")
                best_loss = loss

            if self.scheduler is not None and isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss)
        
        if self.log_wandb:
            wandb.finish()

        print("Finished training!")
    
    def set_lr_scheduler(self, lr_type: str = 'cosine') -> lr_scheduler:
        """Configure and initialize learning rate scheduler with warmup.

        Creates a learning rate scheduler based on the specified type. 'cosine' and 'linear'
        schedulers include warmup phase for training stability. 'plateau' reduces learning
        rate when validation loss plateaus.

        Args:
            lr_type (str, optional): Type of learning rate scheduler. Options:
                - 'cosine': Linear warmup followed by cosine annealing to 10% of initial LR
                - 'plateau': Reduces LR by factor when validation loss stops improving
                - 'linear': Linear warmup followed by linear decay to 10% of initial LR
                Defaults to 'cosine'.

        Returns:
            lr_scheduler: Configured PyTorch learning rate scheduler instance.

        Note:
            For 'cosine' and 'linear', warmup steps = warmup_fraction * total_steps.
            'plateau' scheduler requires manual .step(loss) calls, which are handled
            automatically at epoch end.
        """
        assert lr_type in ['cosine', 'plateau', 'linear'], "Use 'cosine', 'plateau', or 'linear'"

        total_steps = self.epochs * len(self.train_dataloader)
        warmup_steps = int(total_steps * self.warmup_fraction)
        
        if lr_type == 'cosine':
            warmup = lr_scheduler.LinearLR(
                self.optim, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine = lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=total_steps - warmup_steps,
                eta_min=self.optim.param_groups[0]['lr'] * 0.1
            )
            self.scheduler = lr_scheduler.SequentialLR(
                self.optim, schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            )
        elif lr_type == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optim, patience=10, threshold=1e-4, min_lr=1e-6
            )
        else:
            warmup = lr_scheduler.LinearLR(
                self.optim, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_steps
            )
            decay = lr_scheduler.LinearLR(
                self.optim, start_factor=1.0, end_factor=0.1,
                total_iters=total_steps - warmup_steps
            )
            self.scheduler = lr_scheduler.SequentialLR(
                self.optim, schedulers=[warmup, decay],
                milestones=[warmup_steps]
            )
        return self.scheduler

    def set_seed(self, seed: int = 1) -> None:
        """Set random seeds for reproducible training across all libraries.

        Configures random number generators for NumPy, PyTorch (CPU and CUDA), and
        makes CUDNN operations deterministic. Essential for experiment reproducibility.

        Args:
            seed (int, optional): Random seed value to use. Same seed should produce
                identical results across runs (assuming same hardware/software).
                Defaults to 1.

        Note:
            Setting deterministic=True may reduce performance. Disables CUDNN benchmarking
            for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Using random seed: {seed}")

    @staticmethod
    def config_from_yaml(file: str) -> dict:
        """Load sparse autoencoder configuration from a YAML file.

        Static utility method for loading model or training configurations from YAML.
        Useful for maintaining configuration files separately from code.

        Args:
            file (str): Path to the YAML configuration file.

        Returns:
            dict: Dictionary containing parsed configuration parameters.
        """
        with open(file, "r") as f:
            config = yaml.safe_load(f)
        return config
