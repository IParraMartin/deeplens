from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "generate_feature_heatmap",
    "plot_topk_distribution",
    "get_top_k_tokens"
]


def generate_feature_heatmap(
        logits: torch.Tensor, 
        save_name: str = None
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

    Returns:
        None: Displays the plot and optionally saves it to disk.

    Note:
        The figure size is set to (20, 6) for optimal visualization of typical
        sequence lengths and vocabulary sizes.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().detach().cpu().numpy()

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
        logits: torch.Tensor, 
        k: int = 20, 
        tokenizer: str = "gpt2", 
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
        tokenizer (str, optional): Name or path of the HuggingFace tokenizer to use
            for decoding token IDs into readable strings. Should match the tokenizer
            used during model training. Defaults to "gpt2".
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
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    pos_logits = logits[position]
    
    if use_softmax:
        exp_logits = np.exp(pos_logits - np.max(pos_logits))
        pos_logits = exp_logits / np.sum(exp_logits)
    
    top_idx = np.argsort(pos_logits)[-k:][::-1]
    top_vals = pos_logits[top_idx]
    top_tokens = [tokenizer.decode([idx]) for idx in top_idx]
    
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
        logits: torch.Tensor, 
        k: int = 10, 
        tokenizer: str = None,
        verbose: bool = False
    ) -> dict:
    """Print the top-k predicted tokens and their logit values for each position in a sequence.

    Iterates through all positions in the sequence and prints the k highest-scoring tokens
    at each position. Useful for detailed analysis of model predictions across the entire
    sequence.

    Args:
        logits (torch.Tensor): Logits tensor with shape (sequence_length, vocab_size)
            or similar. Will be automatically squeezed and moved to CPU if needed.
        k (int, optional): Number of top predictions to display per position.
            Defaults to 10.
        tokenizer (str, optional): Name or path of the HuggingFace tokenizer to use
            for decoding token IDs into readable strings. If None, only token IDs and
            logit values are displayed without decoded text. Should match the tokenizer
            used during model training. Defaults to None.
        verbose (bool, optional): If True, prints the results to the console. Defaults
            to False. 

    Returns:
        dict: Dictionary mapping position indices to their top-k predictions.

    Example Output:
        Top-10 predicted tokens per position
        
        Position 0:
            Token 262 ('the'): 12.45
            Token 290 ('a'): 11.32
            ...
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().detach().cpu().numpy()
    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
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
            if tokenizer is not None:
                token = tokenizer.decode([idx])
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

    return out
