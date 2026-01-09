import os
import yaml
import warnings

os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "cache"
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import torch
from deeplens.sae import SparseAutoencoder
from deeplens.utils.tools import get_device

warnings.filterwarnings('ignore')


__all__ = [
    "InterveneFeatures",
    "ReinjectSingleSample"
]


class InterveneFeatures():
    """Manipulate and intervene on sparse autoencoder latent features.

    This class loads a trained sparse autoencoder and provides methods to analyze
    and modify its latent feature space, enabling causal analysis of feature effects
    on model behavior.
    """
    def __init__(
            self,
            sae_model: str = None,
            sae_config: str | dict = None,
            device: str = "auto"
        ):
        """Initialize the InterveneFeatures class for manipulating sparse autoencoder latent features.

        This class provides functionality to load a trained sparse autoencoder model and
        intervene on its latent feature space to analyze and modify activations.

        Args:
            sae_model (str, optional): Path to the trained sparse autoencoder model weights file.
                Should be a .pt or .pth file containing the model state dict. Defaults to None.
            sae_config (str | dict, optional): Configuration for the sparse autoencoder.
                Can be either a dictionary containing model hyperparameters or a path to a
                YAML configuration file. Defaults to None.
            device (str, optional): Device to run computations on. Can be "auto" for automatic
                selection, "cuda" for GPU, or "cpu" for CPU. Defaults to "auto".
        """
        self.model_dir = sae_model

        self.device = get_device(device)
        print(f"Running on device: {self.device}")

        if isinstance(sae_config, dict):
            self.model_config = sae_config
        elif isinstance(sae_config, str) and sae_config.endswith(".yaml"):
            self.model_config = self.config_from_yaml(sae_config)
        else:
            raise ValueError("sae_config must be dict or path to .yaml file.")
        
        self.model = self.load_model()

    @torch.no_grad()
    def get_decoded(self, activations) -> torch.Tensor:
        """Encode input activations through the sparse autoencoder to get latent features.

        Passes the input activations through the sparse autoencoder's forward pass and
        returns the latent feature representation (z) from the encoded space.

        Args:
            activations (torch.Tensor | array-like): Input activations to encode. Can be a
                PyTorch tensor or any array-like structure that can be converted to a tensor.

        Returns:
            torch.Tensor: The latent feature representation (z) from the sparse autoencoder's
                encoded space.
        """
        if not isinstance(activations, torch.Tensor):
            activations = torch.Tensor(activations)
        activations = activations.to(self.device)
        _, z, _ = self.model(activations)
        return z

    @torch.no_grad()
    def get_alive_features(
            self, 
            activations: torch.Tensor, 
            token_position: int = -1, 
            k: int | None = None,
            return_values: bool = False
        ) -> torch.Tensor:
        """Get indices of non-zero (active) features in the latent space for a specific token.

        Encodes the input activations through the sparse autoencoder and identifies which
        latent features are active (non-zero) at the specified token position.

        Args:
            activations (torch.Tensor | array-like): Input activations to encode. Can be a
                PyTorch tensor or any array-like structure that can be converted to a tensor.
            token_position (int, optional): Position of the token in the sequence to analyze.
                Use -1 for the last token. Defaults to -1.
            k (int, optional): If provided, returns only the top-k most active features
                instead of all non-zero features. Defaults to None.
            return_values (bool, optional): If True, returns both indices and values.
                Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: If return_values is False,
                returns a 1D tensor containing the indices of non-zero features. If True,
                returns a tuple of (indices, values).
        """
        z = self.get_decoded(activations)
        z_token = z[token_position]
        if k is not None:
            topk_result = torch.topk(z_token, k=k)
            feature_idxs = topk_result.indices
            feature_vals = topk_result.values
        else:
            feature_idxs = torch.nonzero(z_token != 0, as_tuple=False).squeeze(-1)
            feature_vals = z_token[feature_idxs]
        if return_values:
            return feature_idxs, feature_vals
        return feature_idxs
    
    @torch.no_grad()
    def intervene_feature(
            self, 
            activations, 
            feature: int, 
            alpha: float = 2.0,
            token_positions: int | list[int] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Intervene on a specific latent feature by scaling its activation.

        Encodes the input activations, multiplies the specified feature by alpha at the
        given token positions, and returns both the original and modified decoded outputs
        for comparison.

        Args:
            activations (torch.Tensor | array-like): Input activations to encode and modify.
                Can be a PyTorch tensor or any array-like structure.
            feature (int): Index of the latent feature to intervene on.
            alpha (float, optional): Scaling factor to apply to the feature. Values > 1
                amplify the feature, values < 1 suppress it. Defaults to 2.0.
            token_positions (int | list[int] | None, optional): Token position(s) at which
                to apply the intervention. If None, applies to all tokens. If int, applies
                to a single position. If list, applies to multiple positions. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - activations: The original input activations
                - original_decoded: Decoded output without intervention
                - modified_decoded: Decoded output with the feature intervention applied
        """
        if not isinstance(activations, torch.Tensor):
            activations = torch.Tensor(activations).unsqueeze(0)
        
        activations = activations.to(self.device)
        _, z, _ = self.model(activations)
        modified = z.clone()
    
        if token_positions is None:
            modified[:, feature] *= alpha
        elif isinstance(token_positions, int):
            modified[token_positions, feature] *= alpha
        else:
            for pos in token_positions:
                modified[pos, feature] *= alpha
        
        original_decoded = self.model.decode(z)
        modified_decoded = self.model.decode(modified)
        return activations, original_decoded, modified_decoded

    def load_model(self) -> torch.nn.Module:
        """Load the sparse autoencoder model from disk.

        Loads the model weights from the specified path and initializes a
        SparseAutoencoder instance with the provided configuration.

        Returns:
            torch.nn.Module: The loaded sparse autoencoder model moved to the
                appropriate device.
        """
        weights = torch.load(self.model_dir, map_location=self.device)
        model = SparseAutoencoder(**self.model_config)
        model.load_state_dict(state_dict=weights)
        return model.to(self.device)
    
    def config_from_yaml(self, file) -> dict:
        """Load sparse autoencoder configuration from a YAML file.

        Reads and parses a YAML configuration file containing the hyperparameters
        for the sparse autoencoder model.

        Args:
            file (str): Path to the YAML configuration file.

        Returns:
            dict: Dictionary containing the model configuration parameters.
        """
        try:
            with open(file, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file}: {e}")
    

class ReinjectSingleSample():
    """Reinject modified activations into a language model for causal inference.

    This class enables injecting modified activations back into a transformer model's
    forward pass to observe the causal effects on model predictions and generated text.
    Useful for validating feature interventions and conducting mechanistic interpretability
    experiments.
    """
    def __init__(self, hf_model: str, device: str = "auto"):
        """Initialize the ReinjectSingleSample class for causal inference with modified activations.

        Loads a HuggingFace causal language model and tokenizer to enable reinjection of
        modified activations into the model's forward pass for text generation and analysis.

        Args:
            hf_model (str): Name or path of the HuggingFace model to load.
                Should be a valid model identifier from the HuggingFace model hub
                (e.g., "gpt2", "meta-llama/Llama-2-7b").
            device (str, optional): Device to run computations on. Can be "auto" for automatic
                selection, "cuda" for GPU, or "cpu" for CPU. Defaults to "auto".
        """
        self.device = get_device(device)
        print(f"Running on device: {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(hf_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model.eval()
        
    @torch.no_grad()
    def reinject_and_generate(
            self, 
            text, 
            modified_activations, 
            layer: int = 3, 
            generate: bool = False, 
            max_new_tokens: int = 25, 
            temperature: float = 1.0
        ) -> torch.Tensor | str:
        """Reinject modified activations into a model layer and optionally generate text.

        Replaces the activations at the specified layer with the provided modified activations
        during the forward pass. Can either return logits for the input text or generate
        new tokens autoregressively.

        Args:
            text (str): Input text to tokenize and process through the model.
            modified_activations (torch.Tensor): The modified activations to inject at the
                specified layer. Should have the appropriate shape for the layer's output.
            layer (int, optional): Index of the transformer layer where activations should
                be replaced. Defaults to 3.
            generate (bool, optional): If True, generates new tokens autoregressively.
                If False, only returns logits for the input. Defaults to False.
            max_new_tokens (int, optional): Maximum number of new tokens to generate when
                generate=True. Defaults to 25.
            temperature (float, optional): Sampling temperature for generation. Higher values
                (>1.0) make output more random, lower values (<1.0) more deterministic.
                Set to 0 for greedy decoding. Defaults to 1.0.

        Returns:
            torch.Tensor | str: If generate=False, returns the model's logits as a tensor.
                If generate=True, returns the generated text as a string.

        Note:
            The hook is automatically removed after execution to prevent interference
            with subsequent forward passes. For generation mode, the hook only affects
            the first forward pass to avoid applying the intervention to newly generated tokens.
        """
        modified_activations = modified_activations.to(self.device)
        call_count = [0]
        def replacement_hook(module, input, output):
            if generate and call_count[0] > 0:
                return output
            call_count[0] += 1
            return modified_activations
        
        mlp_module = self.get_module_for_replacement_hook(layer_idx=layer)
        hook = mlp_module.register_forward_hook(replacement_hook)
        tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        try:
            if generate:
                generated_ids = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
                return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                out = self.model(**tokens)
                return out.logits
        finally:
            hook.remove()

    def get_module_for_replacement_hook(self, layer_idx) -> torch.nn.Module:
        """Get the MLP activation module for a specific layer.

        Retrieves the MLP activation function module at the specified layer,
        which can be used to register forward hooks for activation replacement.

        Args:
            layer_idx (int): Index of the transformer layer (0-indexed).

        Returns:
            torch.nn.Module: The MLP activation module at the specified layer.
        """
        if isinstance(self.model, (
            transformers.GPT2LMHeadModel, 
            transformers.FalconForCausalLM
        )):
            module = self.model.transformer.h[layer_idx].mlp.act
        elif isinstance(self.model, (
            transformers.LlamaForCausalLM, 
            transformers.MistralForCausalLM, 
            transformers.Gemma3ForCausalLM, 
            transformers.GemmaForCausalLM, 
            transformers.Qwen2ForCausalLM,
            transformers.Qwen3ForCausalLM
        )):
            module = self.model.model.layers[layer_idx].mlp.act_fn
        elif isinstance(self.model, (
            transformers.PhiForCausalLM, 
            transformers.Phi3ForCausalLM
        )):
            module = self.model.model.layers[layer_idx].mlp.activation_fn
        else:
            raise NotImplementedError(f"Model type {type(self.model).__name__} is not currently supported.")
    
        return module