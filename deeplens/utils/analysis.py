from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_feature_heatmap(logits: torch.Tensor, save_name: str = None):
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
        tokenizer_name: str = "gpt2", 
        position: int = 0, 
        save_name: str = None, 
        use_softmax: bool = False,
        title: str = None
    ):
    """Plot bar chart of top-k token predictions at a specific position
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().detach().cpu().numpy()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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

def get_top_k_tokens(logits: torch.Tensor, k: int = 10, tokenizer: str = None):
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().detach().cpu().numpy()
    if tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    print(f"Top-{k} predicted tokens per position")
    for pos in range(logits.shape[0]):
        top_idx = np.argsort(logits[pos])[-k:][::-1]
        top_vals = logits[pos][top_idx]
        print(f'\nPosition {pos}:')
        for idx, val in zip(top_idx, top_vals):
            if tokenizer is not None:
                token = tokenizer.decode([idx])
                print(f"\tToken {idx} ('{token}'): {val:.2f}")
            else:
                print(f"\tToken {idx}: {val:.2f}")