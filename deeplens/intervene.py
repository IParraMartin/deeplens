import os
import yaml
import warnings

os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "cache"
from transformers import AutoModel, AutoTokenizer

import torch
from deeplens.sae import SparseAutoencoder

warnings.filterwarnings('ignore')


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

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        print(f"Running on device: {self.device}")

        if str(sae_config).endswith(".yaml"):
            self.model_config = self.config_from_yaml(sae_config)
        elif type(sae_config) == dict:
            self.model_config = sae_config
        else:
            raise ValueError("Unsupported configuration.")
        
        self.model = self.load_model()

    @torch.no_grad()
    def get_alive_features(self, example) -> torch.Tensor:
        """Returns non-zero features of the latent space
        """
        if type(example) != torch.Tensor:
            example = torch.Tensor(example)
        _, z, _ = self.model(example)
        features = torch.nonzero(z, as_tuple=False).squeeze()
        return features
    
    @torch.no_grad()
    def intervene_feature(self, example, feature, alpha) -> tuple:
        """Encodes an example, intervenes a given feature from the learned 
        sparse latent space, and returns the decoded and original
        tensors.
        """
        if type(example) != torch.Tensor:
            example = torch.Tensor(example).unsqueeze(0)
        _, z, _ = self.model(example)
        modified = z.clone()
        modified[:, feature] *= alpha
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
    def __init__(self, hf_model, features):
        self.model = AutoModel.from_pretrained(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    @torch.no_grad()
    def reinject_features(features, model):
        """Injects the modified features to the respective layer of the model
        """
        pass
