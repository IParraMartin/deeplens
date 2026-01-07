from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from deeplens.extractor import ExtractSingleSample
from deeplens.intervene import InterveneFeatures


__all__ = [
    "AnalysisUtils",
    "FeatureResult"
]


@dataclass
class FeatureResult:
    features: torch.Tensor
    values: torch.Tensor | None = None


class AnalysisUtils():
    """Unified analysis utilities for model logits and SAE features.

    This class provides methods for visualizing and analyzing transformer model
    outputs, including logit heatmaps, token probability distributions, and
    sparse autoencoder (SAE) feature extraction.
    """
    def __init__(
            self,
            hf_model: str = None,
            sae_model: str = None,
            sae_config: str | dict = None,
            layer: int = None
        ):
        """Initialize the AnalysisUtils class for analyzing sparse autoencoder results.

        Args:
            hf_model (str, optional): Name or path of the HuggingFace model for extracting
                MLP activations (e.g., "gpt2", "meta-llama/Llama-2-7b"). Required for
                `get_most_active_features`. Defaults to None.
            sae_model (str, optional): Path to the trained sparse autoencoder model weights
                file. Required for `get_most_active_features`. Defaults to None.
            sae_config (str | dict, optional): Path to the YAML configuration file or a
                dictionary containing SAE hyperparameters. Required for 
                `get_most_active_features`. Defaults to None.
            layer (int, optional): Index of the transformer layer to extract activations
                from (0-indexed). Required for `get_most_active_features`. Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.hf_model = hf_model
        self.sae_model = sae_model
        self.sae_config = sae_config
        self.layer = layer
        
        self._mlp_extractor = None
        self._sae_extractor = None

    @property
    def mlp_extractor(self):
        if self._mlp_extractor is None:
            if self.hf_model is None or self.layer is None:
                raise ValueError("hf_model and layer required for MLP extraction")
            self._mlp_extractor = ExtractSingleSample(hf_model=self.hf_model, layer=self.layer)
        return self._mlp_extractor
    
    @property
    def sae_extractor(self):
        if self._sae_extractor is None:
            if self.sae_model is None or self.sae_config is None:
                raise ValueError("sae_model and sae_config required for SAE extraction")
            self._sae_extractor = InterveneFeatures(
                sae_model=self.sae_model,
                sae_config=self.sae_config
            )
        return self._sae_extractor

    def generate_logit_heatmap(
            self,
            logits: torch.Tensor, 
            save_name: str = None,
            k: int | None = None
        ) -> None:
        """Generate and display a heatmap visualization of logits across token positions.

        Creates a color-coded heatmap showing the distribution of logit values across the
        vocabulary for each token position in a sequence. Uses the 'inferno' colormap for
        visualization.

        Args:
            logits (torch.Tensor): Logits tensor with shape (sequence_length, vocab_size)
                or similar. Will be automatically squeezed and moved to CPU if needed.
            save_name (str, optional): Filename (without extension) to save the plot.
                If provided, saves the figure as a PNG file with 300 DPI. If None, only
                displays the plot. Defaults to None.
            k (int): top k vocabulary indexes to plot. Defaults to None.

        Returns:
            None: Displays the plot and optionally saves it to disk.

        Note:
            The figure size is set to (20, 6) for optimal visualization of typical
            sequence lengths and vocabulary sizes.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.squeeze().detach().cpu().numpy()

        if k is not None:
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits)
            logits = torch.topk(logits, k=k, dim=-1).values.numpy()

        plt.figure(figsize=(20, 6))
        plt.imshow(logits, cmap="inferno", aspect="auto")
        plt.colorbar()
        plt.xlabel("Vocabulary Index", size=18)
        plt.ylabel("Token Position", size=18)
        plt.xticks(size=12)
        plt.yticks(size=12)
        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_topk_distribution(
            self,
            logits: torch.Tensor, 
            k: int = 20, 
            position: int = 0, 
            save_name: str = None,
            use_softmax: bool = True,
            title: str = None
        ) -> None:
        """Plot a bar chart showing the top-k most probable tokens at a specific sequence position.

        Creates a horizontal bar chart displaying the highest-probability token predictions
        at a given position in the sequence. Useful for analyzing model predictions and
        understanding which tokens are most likely at specific positions.

        Args:
            logits (torch.Tensor): Logits tensor with shape (sequence_length, vocab_size)
                or similar. Will be automatically squeezed and moved to CPU if needed.
            k (int, optional): Number of top predictions to display in the bar chart.
                Defaults to 20.
            position (int, optional): Token position in the sequence to analyze. Use 0-based
                indexing. Defaults to 0 (first token).
            save_name (str, optional): Filename (without extension) to save the plot.
                If provided, saves the figure as a PNG file with 300 DPI. If None, only
                displays the plot. Defaults to None.
            use_softmax (bool, optional): If True, applies softmax to convert logits to
                probabilities before plotting. If False, plots raw logit values.
                Defaults to True.
            title (str, optional): Custom title for the plot. If None, no title is displayed.
                Defaults to None.

        Returns:
            None: Displays the plot and optionally saves it to disk.

        Note:
            Token labels are rotated 45 degrees for readability. The y-axis label changes
            based on use_softmax: shows probability notation if True, "Logit Value" if False.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.squeeze().detach().cpu().numpy()
        
        pos_logits = logits[position]
        
        if use_softmax:
            exp_logits = np.exp(pos_logits - np.max(pos_logits))
            pos_logits = exp_logits / np.sum(exp_logits)
        
        top_idx = np.argsort(pos_logits)[-k:][::-1]
        top_vals = pos_logits[top_idx]
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_idx]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(k), top_vals)
        plt.xticks(range(k), top_tokens, rotation=45, ha='right', fontsize=10)
        plt.xlabel('Token', fontsize=14)
        ylabel = r'$P$(token)' if use_softmax else 'Logit Value'
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def get_top_k_tokens(
            self,
            logits: torch.Tensor, 
            k: int = 10, 
            to_dataframe: bool = True,
            verbose: bool = False
        ) -> pd.DataFrame | list[dict]:
        """Get the top-k predicted tokens and their probabilities for each position in a sequence.

        Iterates through all positions in the sequence and extracts the k highest-scoring tokens
        at each position. Useful for detailed analysis of model predictions across the entire
        sequence.

        Args:
            logits (torch.Tensor): Logits tensor with shape (sequence_length, vocab_size)
                or similar. Will be automatically squeezed and moved to CPU if needed.
            k (int, optional): Number of top predictions to display per position.
                Defaults to 10.
            to_dataframe (bool, optional): If True, returns the top-k predicted tokens at each 
                position and their probabilities in a pandas DataFrame. Defaults to True.
            verbose (bool, optional): If True, prints the results to the console. Defaults
                to False. 

        Returns:
            pd.DataFrame | list[dict]: If to_dataframe=True, returns a DataFrame with columns
                'position', 'rank', 'token', and 'probability'. Otherwise returns a list of
                dictionaries with the same fields.

        Example Output:
            Top-10 predicted tokens per position
            
            Position 0:
                Token ('the'): 12.45
                Token ('a'): 11.32
                ...
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.squeeze().detach().cpu().numpy()
        if verbose:
            print(f"Top-{k} predicted tokens per position")

        exp_logits = np.exp(logits - np.max(logits))
        logits = exp_logits / np.sum(exp_logits)

        results = {}
        for pos in range(logits.shape[0]):
            top_idx = np.argsort(logits[pos])[-k:][::-1]
            top_vals = logits[pos][top_idx]
            if verbose:
                print(f'\nPosition {pos}:')
            tokens = []
            for idx, val in zip(top_idx, top_vals):
                if self.tokenizer is not None:
                    token = self.tokenizer.decode([idx])
                    tokens.append(token)
                    if verbose:
                        print(f"\t'{token}': {val:.2f}")
                else:
                    if verbose:
                        print(f"\tToken {idx}: {val:.2f}")
            
            results[pos] = {
                'tokens': tokens,
                'values': top_vals.tolist()
            }

        out = []
        for position, data in results.items():
            for i, (token, prob) in enumerate(zip(data['tokens'], data['values'])):
                out.append({
                    'position': position,
                    'rank': i + 1,
                    'token': token,
                    'probability': prob
                })
        
        if to_dataframe:
            return pd.DataFrame(out)
        else:
            return out

    def get_most_active_features(
            self,
            sentences: list[str], 
            target: int | str | None = None,
            k: int | None = None,
            case_sensitive: bool = True,
            return_values: bool = False
        ) -> dict[str, FeatureResult]:
        """Extract SAE latent features for specific tokens across multiple sentences.

        This method processes a list of sentences through a transformer model and sparse
        autoencoder to extract active features at specified token positions. Useful for
        analyzing which SAE features activate for particular tokens or syntactic patterns.

        Note:
            Requires `hf_model`, `layer`, `sae_model`, and `sae_config` to be set during
            class initialization.

        Args:
            sentences (list[str]): List of input sentences to process. Each sentence will
                be tokenized and processed independently.
            target (int | str | None, optional): Specifies which token positions to analyze.
                - If int: Token position index (supports negative indexing, e.g., -1 for last token).
                - If str: Token string to match (e.g., "What"). Will find all occurrences.
                - If None: Extracts features for all token positions.
                Defaults to None.
            k (int | None, optional): If provided, returns only the top-k most active features
                by activation magnitude. If None, returns all non-zero features. Defaults to None.
            case_sensitive (bool, optional): Whether string matching for target tokens should
                be case-sensitive. Only applies when target is a string. Defaults to True.
            return_values (bool, optional): Whether the function will return features and activation
                values of each feature. Defaults to False.

        Returns:
            dict[str, FeatureResult]: Dictionary mapping descriptive keys to FeatureResult objects.
                Keys follow the format "sent_{idx}_pos_{pos}_tok_{token}" where:
                - idx: 1-indexed sentence number
                - pos: 0-indexed token position within the sentence
                - token: The decoded token string (stripped of whitespace)
                Values are FeatureResult dataclass instances with:
                - features: Tensor containing indices of active/top-k features
                - values: Tensor of activation values (or None if return_values=False)
        """
        features = {}
        for idx, sent in enumerate(sentences):
            acts = self.mlp_extractor.get_mlp_acts(sample=sent)
            input_ids = self.tokenizer.encode(sent, add_special_tokens=False)
            num_tokens = acts.shape[0]
            
            if target is None:
                positions = list(range(num_tokens))
            elif isinstance(target, int):
                pos = target if target >= 0 else num_tokens + target
                if 0 <= pos < num_tokens:
                    positions = [pos]
                else:
                    raise ValueError(
                        f"Position {target} out of range for sentence {idx+1} ({num_tokens} tokens)"
                    )
            elif isinstance(target, str):
                target_to_match = target if case_sensitive else target.lower()
                positions = []
                for i, tok_id in enumerate(input_ids):
                    decoded = self.tokenizer.decode([tok_id])
                    decoded_to_match = decoded if case_sensitive else decoded.lower()
                    if decoded_to_match.strip() == target_to_match.strip():
                        positions.append(i)
                if not positions:
                    print(f"Warning: '{target}' not found in sentence '{sent}'")
                    continue
            else:
                raise TypeError(f"target must be int, str, or None, got {type(target)}")
            
            for pos in positions:
                if pos >= num_tokens:
                    continue

                vals = None
                if return_values:
                    feats, vals = self.sae_extractor.get_alive_features(
                        activations=acts, 
                        token_position=pos, 
                        k=k,
                        return_values=True
                    )
                else:
                    feats = self.sae_extractor.get_alive_features(
                        activations=acts, 
                        token_position=pos, 
                        k=k,
                        return_values=False
                    )
                token_str = self.tokenizer.decode([input_ids[pos]])
                key = f"sent_{idx+1}_pos_{pos}_tok_{token_str.strip()}"
                features[key] = FeatureResult(
                    feats.cpu(), 
                    vals.cpu() if vals is not None else None
                )
        return features
