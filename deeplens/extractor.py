import os
import warnings
from tqdm import tqdm

import torch

os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "cache"
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

warnings.filterwarnings('ignore')


class FromHuggingFace():
    def __init__(
            self, 
            model: str = "gpt2", 
            layer: int = 6,
            dataset_name: str = "HuggingFaceFW/fineweb",
            num_samples: int = 100000,
            seq_length: int = 128,
            inference_batch_size: int = 16, 
            device: str = "auto",
            save_features: bool = True
        ):

        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.layer = layer
        self.batch_size = inference_batch_size
        self.save_features = save_features
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.dataset = load_dataset(
            dataset_name, 
            split='train',
            streaming=True
        ).take(num_samples)

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, examples) -> torch.Tensor:
        """Tokenize text examples
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.seq_length,
            return_tensors='pt'
        )

    def get_activations(self, layer_idx) -> tuple:
        """Register hook to capture MLP activations
        """
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        hook = self.model.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        return hook, activations

    @torch.no_grad()
    def extract_features(self) -> torch.Tensor:
        """Extract MLP activations from the specified layer
        """
        hook, activations = self.get_activations(self.layer)
        all_activations = []
        batch_texts = []     
        for example in tqdm(self.dataset, desc=f"Extracting from L{self.layer}", total=self.num_samples):
            batch_texts.append(example['text'])
            if len(batch_texts) == self.batch_size:
                tokens = self.tokenize({'text': batch_texts})
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                _ = self.model(**tokens)
                batch_acts = activations[-1]
                attention_mask = tokens["attention_mask"].cpu()
                for i in range(batch_acts.shape[0]):
                    non_pad_mask = attention_mask[i].bool()
                    valid_acts = batch_acts[i][non_pad_mask]
                    all_activations.append(valid_acts)
                batch_texts = []
        
        # for residual text not batched
        if batch_texts:
            tokens = self.tokenize({'text': batch_texts})
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            _ = self.model(**tokens)
            batch_acts = activations[-1]
            attention_mask = tokens["attention_mask"].cpu()
            for i in range(batch_acts.shape[0]):
                non_pad_mask = attention_mask[i].bool()
                valid_acts = batch_acts[i][non_pad_mask]
                all_activations.append(valid_acts)
    
        hook.remove()

        features = torch.cat(all_activations, dim=0)
        print(f"Extracted features shape: {features.shape}")
        
        if self.save_features:
            os.makedirs('saved_features', exist_ok=True)
            save_path = f"saved_features/features_layer_{self.layer}_{features.shape[0]}.pt"
            torch.save(features, save_path)
            print(f"Features saved to {save_path}")
    
        return features

class ExtractSingleSample():
    def __init__(
            self, 
            model: str = "gpt2", 
            layer: int = 3, 
            max_length: int = 1024, 
            device: str = "auto"
        ):
        """_summary_

        Args:
            model (str, optional): _description_. Defaults to "gpt2".
            sample (str, optional): _description_. Defaults to None.
            layer (int, optional): _description_. Defaults to 3.
            max_length (int, optional): _description_. Defaults to 1024.
            device (str, optional): _description_. Defaults to "auto".
        """
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.layer = layer
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_mlp_acts(self, sample):
        hook, activations = self.get_activations(self.layer)
        tokens = self.tokenize(sample)
        _ = self.model(**tokens)
        acts = activations[-1].squeeze()
        hook.remove()
        return acts
    
    def tokenize(self, sample) -> torch.Tensor:
        """Tokenize text examples
        """
        return self.tokenizer(
            sample,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
    
    def get_activations(self, layer_idx) -> tuple:
        """Register hook to capture MLP activations
        """
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        hook = self.model.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        return hook, activations

