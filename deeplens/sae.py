import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
            self, 
            input_dims: int = 512, 
            n_features: int = 2048, 
            activation: str = "relu",
            input_norm: bool = True,
            k: int | None = None,
            beta_l1: float | None = None,
            tie_weights: bool = False,
            unit_norm_decoder: bool = True
        ) -> None:
        """Sparse Autoencoder for learning interpretable features.

        Args:
            input_dims (int): Input dimension (e.g., 3072 for GPT-2 MLP).
            n_features (int): Number of SAE features (expansion factor * input_dims).
            activation (str): Activation function ('relu' or 'silu').
            input_norm (bool): Whether to apply LayerNorm to inputs.
            k (int | None): If set, use top-k sparsity instead of L1.
            beta_l1 (float): L1 sparsity coefficient (ignored if k is set).
            tie_weights (bool): Whether to tie encoder and decoder weights.
            unit_norm_decoder (bool): Whether to normalize decoder weights to unit norm.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dims) if input_norm else nn.Identity()
        self.encoder = nn.Linear(input_dims, n_features, bias=True)
        self.decoder = None if tie_weights else nn.Linear(n_features, input_dims, bias=False)
        self.unit_norm_decoder = unit_norm_decoder
        self.input_norm = input_norm

        if activation == "relu":
            self.activation = nn.ReLU()
            kaiming_activation = "relu"
        elif activation == "silu":
            self.activation = nn.SiLU()
            kaiming_activation = "linear"
        else:
            raise ValueError("Activation must be 'relu' or 'silu'")
        
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity=kaiming_activation)
        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.unit_norm_decoder:
                self._renorm_decoder()

        self.k = k
        self.beta_l1 = beta_l1
        self.tie_weights = tie_weights

    @torch.no_grad()
    def _renorm_decoder(self, eps: float = 1e-8) -> None:
        """_summary_
        """
        if self.decoder is not None and self.unit_norm_decoder:
            W = self.decoder.weight.data
            norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
            self.decoder.weight.data = W / norms

    def encode(self, x) -> torch.Tensor:
        """Encoder module of the model
        """
        x = self.norm(x)
        return self.activation(self.encoder(x))

    def decode(self, z) -> torch.Tensor:
        """Decoder module of the model
        """
        if self.tie_weights:
            return F.linear(z, self.encoder.weight.t(), bias=None)
        else:
            return self.decoder(z)
        
    def post_step(self) -> None:
        """Function to renorm the decoder after each step.
        """
        self._renorm_decoder()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model
        """
        z_pre = self.encode(x)
        z = self.topk_mask(z_pre, self.k) if self.k is not None else z_pre
        x_hat = self.decode(z)
        return x_hat, z, z_pre

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Loss computation with logs to print.
        """
        x_hat, z, _ = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        if self.k is None:
            sparsity = z.abs().mean()
            total = recon + self.beta_l1 * sparsity
            logs = {
                "mse": recon.detach(),
                "l1": sparsity.detach(),
                "non_zero_frac": (z != 0).float().mean().detach()
            }
        else:
            total = recon
            logs = {
                "mse": recon.detach(),
                "non_zero_frac": (z != 0).float().mean().detach()
            }
        return total, logs
    
    def topk_mask(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k masking for the 
        """
        if k is None or k <= 0 or k >= z.size(-1):
            return z
        vals, idx = torch.topk(z.abs(), k, dim=-1)
        out = torch.zeros_like(z)
        return out.scatter(-1, idx, z.gather(-1, idx))
