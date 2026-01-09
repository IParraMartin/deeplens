import torch
from transformers import AutoConfig, AutoModelForCausalLM
import transformers
import warnings
warnings.filterwarnings('ignore')


def get_device(device: str = "auto") -> torch.device:
    """Utility to set up the torch device. If 'auto', it selects
    the most appropriate device in your machine. 
    
    It can be set manually to 'mps', 'cuda', or 'cpu', but 'auto' is 
    recommended.

    Args:
        device: The device in which the given process will be allocated. 
            Defaults to 'auto'.

    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)

def get_architecture(hf_model):
    config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print(model)

# DONE: 
#   openai-community/gpt2
#   openai-community/gpt2-medium
#   openai-community/gpt2-large
#   openai-community/gpt2-xxl
#   meta-llama/Llama-2-7b-chat-hf 
#   meta-llama/Llama-3.2-1B 
#   TinyLlama/TinyLlama-1.1B-Chat-v1.0 
#   microsoft/phi-2 
#   microsoft/Phi-3.5-mini-instruct
#   microsoft/Phi-4-mini-instruct
#   mistralai/Mistral-7B-v0.1 
#   google/gemma-3-270m
#   google/gemma-7b-it
#   tiiuae/falcon-7b
#   Qwen/Qwen2.5-7B-Instruct 
#   deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#   deepseek-ai/DeepSeek-R1-Distill-Llama-8B
