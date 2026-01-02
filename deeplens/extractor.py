import os
import warnings
from tqdm import tqdm

import torch

os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from deeplens.utils.tools import get_device, get_mlp_module

warnings.filterwarnings('ignore')


__all__ = [
    "FromHuggingFace",
    "ExtractSingleSample"
]


class FromHuggingFace():
    """Extract MLP activations from transformer models using HuggingFace datasets.

    This class loads a pre-trained transformer model and processes samples from a streaming
    dataset to extract and save intermediate layer activations. Designed for collecting
    training data for sparse autoencoders.
    """
    def __init__(
            self, 
            hf_model: str = "gpt2", 
            layer: int = 6,
            dataset_name: str = "HuggingFaceFW/fineweb",
            num_samples: int = 100000,
            seq_length: int = 128,
            inference_batch_size: int = 16, 
            device: str = "auto",
            save_features: bool = True
        ) -> None:
        """Initialize the activation extractor with model and dataset configuration.

        Loads the specified model and tokenizer, sets up dataset streaming, and configures
        extraction parameters. The model is set to evaluation mode and moved to the
        appropriate device.

        Args:
            model (str, optional): Name or path of the HuggingFace model to load.
                Should be a valid model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b").
                Defaults to "gpt2".
            layer (int, optional): Index of the transformer layer to extract activations from.
                0-indexed. Defaults to 6.
            dataset_name (str, optional): Name of the HuggingFace dataset to stream.
                Must be a valid dataset identifier. Defaults to "HuggingFaceFW/fineweb".
            num_samples (int, optional): Number of samples to extract from the dataset.
                Defaults to 100000.
            seq_length (int, optional): Maximum sequence length for tokenization. Sequences
                will be truncated or padded to this length. Defaults to 128.
            inference_batch_size (int, optional): Batch size for processing samples through
                the model. Higher values increase memory usage but improve speed.
                Defaults to 16.
            device (str, optional): Device for model inference. Can be "auto" for automatic
                selection, "cuda", "mps", or "cpu". Defaults to "auto".
            save_features (bool, optional): Whether to save extracted features to disk in
                the 'saved_features' directory. Defaults to True.
        """

        self.model = AutoModelForCausalLM.from_pretrained(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
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

        self.device = get_device(device)
        print(f"Running on device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, examples) -> dict:
        """Tokenize text examples with padding and truncation.

        Converts raw text into token IDs suitable for model input, applying padding to
        seq_length and truncation as needed.

        Args:
            examples (dict): Dictionary containing a 'text' key with a list of text strings
                to tokenize.

        Returns:
            dict: Dictionary with tokenized outputs including 'input_ids', 'attention_mask',
                and other tokenizer-specific keys. All tensors have shape (batch_size, seq_length).
        """
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.seq_length,
            return_tensors='pt'
        )

    def set_forward_hook_and_return_activations(self, layer_idx: int) -> tuple:
        """Register a forward hook to capture MLP activations from a specific layer.

        Creates a hook function that captures the output of the MLP activation function
        at the specified layer during forward passes. Activations are detached and moved
        to CPU to save GPU memory.

        Args:
            layer_idx (int): Index of the transformer layer to hook (0-indexed).

        Returns:
            tuple: A tuple containing:
                - hook (torch.utils.hooks.RemovableHandle): Handle to remove the hook later
                - activations (list): List that will be populated with activation tensors
                    during forward passes
        """
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        if hasattr(self.model, "transformer"):
            hook = self.model.transformer.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        else:
            hook = self.model.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        return hook, activations

    @torch.no_grad()
    def extract_features(self) -> torch.Tensor:
        """Extract MLP activations from the specified layer across the entire dataset.

        Processes the dataset in batches, extracting activations from the configured layer.
        Filters out padding tokens to ensure only valid activations are collected. Optionally
        saves the extracted features to disk.

        The extraction process:
        1. Batches text samples for efficient processing
        2. Tokenizes and pads/truncates to seq_length
        3. Runs forward pass and captures activations via hook
        4. Filters out activations from padding tokens using attention mask
        5. Concatenates all valid activations into a single tensor

        Returns:
            torch.Tensor: Concatenated activation tensor with shape (total_tokens, hidden_dim),
                where total_tokens is the sum of all non-padding tokens across all samples.
                The tensor is saved to 'saved_features/features_layer_{layer}_{num_tokens}.pt'
                if save_features=True.

        Note:
            The hook is automatically removed after extraction to prevent memory leaks.
            Progress is displayed via tqdm progress bar.
        """
        hook, activations = self.set_forward_hook_and_return_activations(self.layer)
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
    """Extract MLP activations from individual text samples for analysis and intervention.

    This class provides functionality to extract activations from single text inputs,
    useful for interactive analysis, debugging, and testing feature interventions on
    specific examples.
    """
    def __init__(
            self, 
            hf_model: str = "gpt2", 
            layer: int = 3, 
            max_length: int = 1024, 
            device: str = "auto"
        ) -> None:
        """Initialize the single sample extractor with model configuration.

        Loads the specified model and tokenizer, and configures extraction parameters.
        The model is set to evaluation mode and moved to the appropriate device.

        Args:
            model (str, optional): Name or path of the HuggingFace model to load.
                Should match the model used for sparse autoencoder training for consistency.
                Defaults to "gpt2".
            layer (int, optional): Index of the transformer layer to extract activations from.
                Should match the layer used for SAE training. 0-indexed. Defaults to 3.
            max_length (int, optional): Maximum sequence length for tokenization. Longer
                sequences will be truncated. Defaults to 1024.
            device (str, optional): Device for model inference. Can be "auto" for automatic
                selection, "cuda", "mps", or "cpu". Defaults to "auto".
        """
        self.model = AutoModelForCausalLM.from_pretrained(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.layer = layer
        self.max_length = max_length

        self.device = get_device(device)
        print(f"Running on device: {self.device}")
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_mlp_acts(self, sample: str) -> torch.Tensor:
        """Extract MLP activations for a single text sample.

        Processes the input text through the model and captures the MLP activations
        from the configured layer. The hook is automatically removed after extraction.

        Args:
            sample (str): Input text to process. Can be a word, phrase, or full sentence.
                Will be tokenized according to the model's tokenizer.

        Returns:
            torch.Tensor: Activation tensor with shape (sequence_length, hidden_dim),
                where sequence_length depends on the tokenized length of the input.
                The batch dimension is squeezed out.

        Note:
            The activations are automatically moved to CPU to save GPU memory.
        """
        hook, activations = self.set_forward_hook_and_return_activations(self.layer)
        tokens = self.tokenize(sample)
        _ = self.model(**tokens)
        acts = activations[-1].squeeze()
        hook.remove()
        return acts
    
    def tokenize(self, sample: str) -> dict:
        """Tokenize a single text sample without padding.

        Converts the input text into token IDs suitable for model input. No padding is
        applied since this is for single sample processing.

        Args:
            sample (str): Text string to tokenize.

        Returns:
            dict: Dictionary containing tokenized outputs with 'input_ids', 'attention_mask',
                and other tokenizer-specific keys as tensors on the configured device.
                Shape is (1, actual_length) where actual_length ≤ max_length.
        """
        return self.tokenizer(
            sample,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
    
    def set_forward_hook_and_return_activations(self, layer_idx: int) -> tuple:
        """Register a forward hook to capture MLP activations from a specific layer.

        Creates a hook function that captures the output of the MLP activation function
        at the specified layer during forward passes. Activations are detached and moved
        to CPU to save GPU memory.

        Args:
            layer_idx (int): Index of the transformer layer to hook (0-indexed).

        Returns:
            tuple: A tuple containing:
                - hook (torch.utils.hooks.RemovableHandle): Handle to remove the hook later
                - activations (list): List that will be populated with activation tensors
                    during forward passes
        """
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        if hasattr(self.model, 'transformer'):
            hook = self.model.transformer.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        else:
            hook = self.model.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        return hook, activations
