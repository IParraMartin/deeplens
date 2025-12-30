import os
import yaml
import warnings

os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "cache"
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import torch
from deeplens.sae import SparseAutoencoder

warnings.filterwarnings('ignore')


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)


class InterveneFeatures():
    def __init__(
            self,
            sae_model: str = None,
            sae_config: dict = None,
            device: str = "auto"
        ):
        """Class to intervene features of the autoencoder's latent
        features

        Args:
            model_dir (str, optional): _description_. Defaults to None.
            model_config (dict, optional): _description_. Defaults to None.
        """
        self.model_dir = sae_model

        self.device = get_device(device)
        print(f"Running on device: {self.device}")

        if str(sae_config).endswith(".yaml"):
            self.model_config = self.config_from_yaml(sae_config)
        elif type(sae_config) == dict:
            self.model_config = sae_config
        else:
            raise ValueError("sae_config must be dict or path to .yaml file")
        
        self.model = self.load_model()

    @torch.no_grad()
    def get_alive_features(self, activations, token_position: int = -1) -> torch.Tensor:
        """Returns non-zero features of the latent space for a given token
        in a sequence
        """
        if not isinstance(activations, torch.Tensor):
            activations = torch.Tensor(activations)
        activations = activations.to(self.device)
        _, z, _ = self.model(activations)

        feature_idxs = torch.nonzero(z[token_position] != 0, as_tuple=False).squeeze(-1)
        return feature_idxs
    
    @torch.no_grad()
    def intervene_feature(
            self, 
            activations, 
            feature: int, 
            alpha: float = 2.0,
            token_positions: int | list[int] | None = None
        ) -> tuple:
        """Encodes an example, intervenes a given feature from the learned 
        sparse latent space, and returns the decoded and original
        tensors.
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
        
        modified = self.model.decode(modified)
        original = self.model.decode(z)
        return original, modified

    def load_model(self) -> torch.nn.Module:
        """Loads the sparse autoencoder
        """
        weights = torch.load(self.model_dir, map_location=self.device)
        model = SparseAutoencoder(**self.model_config)
        model.load_state_dict(state_dict=weights)
        return model.to(self.device)
    
    def config_from_yaml(self, file) -> dict:
        """Returns a sparse autoencoder configuration
        from a yaml file
        """
        with open(file, "r") as f:
            config = yaml.safe_load(f)
        return config
    

class ReinjectSingleSample():
    def __init__(self, hf_model: str, device: str = "auto"):
        """Reinjects the modified activations and generates text
        for causal inference
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
        ):
        """Injects the modified features to the respective layer of the model.
        """
        modified_activations = modified_activations.to(self.device)
        call_count = [0]
        def replacement_hook(module, input, output):
            if generate and call_count[0] > 0:
                return output
            call_count[0] += 1
            return modified_activations
        
        hook = self.model.h[layer].mlp.act.register_forward_hook(replacement_hook)
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


