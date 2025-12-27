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
        print(f"Extracting features from layer {self.layer}...")
        hook, activations = self.get_activations(self.layer)
        all_activations = []
        batch_texts = []     
        for example in tqdm(self.dataset, desc="Extracting features:", total=self.num_samples):
            batch_texts.append(example['text'])
            if len(batch_texts) == self.batch_size:
                tokens = self.tokenize({'text': batch_texts})
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                _ = self.model(**tokens)
                batch_acts = activations[-1]
                all_activations.append(batch_acts)
                batch_texts = []
        
        # for residual text not batched
        if batch_texts:
            tokens = self.tokenize({'text': batch_texts})
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            _ = self.model(**tokens)
            batch_acts = activations[-1]
            all_activations.append(batch_acts)
        
        hook.remove()
        features = torch.cat(all_activations, dim=0)
        features = features.reshape(-1, features.shape[-1])
        print(f"Extracted features (shape): {features.shape}")
        
        if self.save_features:
            os.makedirs('saved_features', exist_ok=True)
            save_path = f"saved_features/features_layer_{self.layer}_{self.seq_length * self.num_samples}.pt"
            torch.save(features, save_path)
            print(f"Features saved to {save_path}")
    
        return features


if __name__ == "__main__":
    extractor = FromHuggingFace(
        model="gpt2",
        layer=3,
        dataset_name="HuggingFaceFW/fineweb",
        num_samples=500,
        seq_length=1024,
        inference_batch_size=16,
        device="auto",
        save_features=True
    )
    
    features = extractor.extract_features()
