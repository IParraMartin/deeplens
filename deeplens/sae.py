import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SparseAutoencoder"
]


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features from neural network activations.

    This implementation supports multiple sparsity methods (L1 regularization or top-k activation),
    optional weight tying between encoder and decoder, and unit-norm decoder constraints for
    improved feature interpretability. Designed for mechanistic interpretability research
    on transformer models.

    The architecture consists of:
    - Optional input normalization layer
    - Encoder: Linear projection to expanded feature space with nonlinear activation
    - Decoder: Linear projection back to original space (optionally tied to encoder)
    - Sparsity constraint: Either L1 penalty or top-k activation selection
    """
    def __init__(
            self, 
            input_dims: int = 512, 
            n_features: int = 2048, 
            activation: str = "relu",
            input_norm: bool = True,
            k: int | None = None,
            beta_l1: float | None = 0.0,
            tie_weights: bool = False,
            unit_norm_decoder: bool = True
        ) -> None:
        """Initialize the Sparse Autoencoder with specified architecture and sparsity settings.

        Args:
            input_dims (int, optional): Dimensionality of input activations (e.g., 3072 for
                GPT-2 layer 3 MLP output). Defaults to 512.
            n_features (int, optional): Number of learned features in the latent space.
                Typically set as expansion_factor × input_dims where expansion_factor is 2-8.
                Defaults to 2048.
            activation (str, optional): Nonlinearity applied after encoder. Must be 'relu'
                or 'silu'. ReLU is standard for interpretability; SiLU may improve reconstruction.
                Defaults to "relu".
            input_norm (bool, optional): Whether to apply LayerNorm to inputs before encoding.
                Helps stabilize training with varying activation magnitudes. Defaults to True.
            k (int | None, optional): If set, uses top-k sparsity (keeps only k largest activations
                per sample) instead of L1 regularization. Mutually exclusive with beta_l1.
                Defaults to None.
            beta_l1 (float | None, optional): L1 regularization coefficient for sparsity.
                Higher values encourage sparser activations. Ignored if k is set. Defaults to 0.0.
            tie_weights (bool, optional): If True, decoder weights are the transpose of encoder
                weights (no separate decoder parameters). Reduces parameters but may hurt performance.
                Defaults to False.
            unit_norm_decoder (bool, optional): If True, constrains decoder weight columns to
                unit norm. Improves feature interpretability by removing scale ambiguity.
                Defaults to True.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dims) if input_norm else nn.Identity()
        self.encoder = nn.Linear(input_dims, n_features, bias=True)
        self.decoder = None if tie_weights else nn.Linear(n_features, input_dims, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dims))
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
        """Normalize decoder weight columns to unit norm for improved interpretability.

        Constrains each feature's decoder weight vector to have L2 norm of 1, removing
        scale ambiguity from the learned features. This is a common technique in dictionary
        learning and sparse coding to ensure features represent directions rather than
        directions with varying magnitudes.

        Args:
            eps (float, optional): Small epsilon value to prevent division by zero for
                near-zero norm weights. Defaults to 1e-8.

        Returns:
            None: Modifies decoder weights in-place. No-op if decoder is None or
                unit_norm_decoder is False.
        """
        if self.decoder is not None and self.unit_norm_decoder:
            W = self.decoder.weight.data
            norms = W.norm(dim=0, keepdim=True).clamp_min(eps)
            self.decoder.weight.data = W / norms

    def encode(self, x) -> torch.Tensor:
        """Encode input activations into the sparse latent feature space.

        Applies optional normalization, linear transformation to expanded feature space,
        and nonlinear activation function.

        Args:
            x (torch.Tensor): Input activation tensor with shape (..., input_dims).
                Typically (batch_size, seq_length, input_dims) or (batch_size, input_dims).

        Returns:
            torch.Tensor: Encoded features with shape (..., n_features) after applying
                normalization (if enabled), linear encoding, and activation function.
                These are the pre-sparsity latent activations.
        """
        x = self.norm(x)
        if self.input_norm:
            return self.activation(self.encoder(x)) # norm already centers
        return self.activation(self.encoder(x - self.b_dec))

    def decode(self, z) -> torch.Tensor:
        """Decode sparse latent features back to the original activation space.

        Applies linear transformation from feature space back to input space. Uses either
        a separate decoder or transposed encoder weights depending on tie_weights setting.

        Args:
            z (torch.Tensor): Latent feature activations with shape (..., n_features).
                Typically the output of encode() or after applying sparsity constraints.

        Returns:
            torch.Tensor: Reconstructed activations with shape (..., input_dims).
                Should approximate the original input when z contains sufficient information.
        """
        if self.tie_weights:
            return F.linear(z, self.encoder.weight.t(), bias=None) + self.b_dec
        else:
            return self.decoder(z) + self.b_dec
        
    def post_step(self) -> None:
        """Renormalize decoder weights after each optimization step.

        Should be called after each parameter update (e.g., after optimizer.step()) to
        maintain the unit norm constraint on decoder weights. This ensures the constraint
        is enforced throughout training.

        Returns:
            None: Modifies decoder weights in-place if unit_norm_decoder is True.

        Note:
            This is a no-op if unit_norm_decoder is False or tie_weights is True.
        """
        self._renorm_decoder()

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a complete forward pass through the sparse autoencoder.

        Encodes input, applies sparsity constraint (top-k or none), and decodes to
        reconstruct the input. Returns both the reconstruction and intermediate activations.

        Args:
            x (torch.Tensor): Input activation tensor with shape (..., input_dims).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x_hat: Reconstructed activations with shape (..., input_dims)
                - z: Sparse latent activations after sparsity constraint, shape (..., n_features)
                - z_pre: Dense latent activations before sparsity constraint, shape (..., n_features)

        Note:
            If k is None, z and z_pre are identical. If k is set, z contains only the
            top-k activations (others are zeroed).
        """
        z_pre = self.encode(x)
        z = self.topk_mask(z_pre, self.k) if self.k is not None else z_pre
        x_hat = self.decode(z)
        return x_hat, z, z_pre

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the training loss with reconstruction and optional sparsity penalty.

        Calculates MSE reconstruction loss and, if using L1 sparsity (k is None), adds
        L1 penalty on latent activations. Also computes diagnostic metrics for logging.

        Args:
            x (torch.Tensor): Input activation tensor with shape (..., input_dims).

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing:
                - total_loss: Scalar loss tensor for backpropagation. If k is None,
                    equals mse + beta_l1 * l1_sparsity. If k is set, equals mse only.
                - logs: Dictionary of detached metrics for logging:
                    - 'mse': Reconstruction loss (MSE between x_hat and x)
                    - 'l1': L1 norm of activations (only if k is None)
                    - 'non_zero_frac': Fraction of non-zero latent activations

        Note:
            The logs dictionary is intended for monitoring training progress and all
            values are detached from the computation graph.
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
        """Apply top-k sparsity constraint by keeping only the k largest activations.

        Zeros out all but the k largest magnitude activations in the latent space,
        enforcing a fixed sparsity level. This is an alternative to L1 regularization
        that provides more predictable and controllable sparsity.

        Args:
            z (torch.Tensor): Dense latent activations with shape (..., n_features).
            k (int): Number of top activations to keep per sample. If None, ≤0, or
                ≥n_features, returns z unchanged.

        Returns:
            torch.Tensor: Sparse latent activations with same shape as z, but with all
                except the k largest (by absolute value) activations set to zero. The
                values of the top-k activations are preserved from the input.

        Note:
            Selection is based on absolute value, but original signed values are preserved
            for the top-k features.
        """
        if k is None or k <= 0 or k >= z.size(-1):
            return z
        vals, idx = torch.topk(z.abs(), k, dim=-1)
        out = torch.zeros_like(z)
        return out.scatter(-1, idx, z.gather(-1, idx))
