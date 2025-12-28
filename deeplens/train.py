import os
import yaml
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np

import wandb


class SAETrainer():
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
            save_best_only: bool = True,
            log_to_wandb: bool = True
        ) -> None:
        """Sparse Autoencoder trainer class.

        Args:
            train_dataloader (DataLoader): _description_
            eval_dataloader (DataLoader): _description_
            model (torch.nn.Module): _description_
            model_name (str): 
            optim (torch.optim.Optimizer, optional): _description_. Defaults to torch.optim.Adam.
            epochs (int, optional): _description_. Defaults to 20.
            bf16 (bool, optional): _description_. Defaults to False.
            save_model (bool, optional): _description_. Defaults to True.
            random_seed (int, optional): _description_. Defaults to 42.
            save_checkpoints (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to "auto".
            grad_clip_norm (float): 
            lrs_type (str):
            eval_steps (int): 
            save_best_only (bool): True
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
        self.save_best_only = save_best_only
        self.log_wandb = log_to_wandb

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
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
                    "lr_scheduler": lrs_type,
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
        """Training step (one epoch) for the training loop
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
        """Evaluation step of the training loop
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
        """Ensembled training loop.
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
        """Sets a lr scheduler if name is provided
        """
        assert lr_type in ['cosine', 'plateau', 'linear'], "Use 'cosine', 'plateau', or 'linear'"

        total_steps = self.epochs * len(self.train_dataloader)
        warmup_steps = int(total_steps * 0.05)
        
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
        """Sets a random seed for reproducibility
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Using random seed: {seed}")

    @staticmethod
    def config_from_yaml(file: str) -> dict:
        """Returns a sparse autoencoder configuration from a 
        yaml file
        """
        with open(file, "r") as f:
            config = yaml.safe_load(f)
        return config
