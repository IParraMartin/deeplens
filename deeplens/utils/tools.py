import torch

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

def get_mlp_module(hf_model: str) -> None:
    """Allows to access the correct MLP module of a given model architecture.
    It is currently under development.

    Args:
        hf_model: Hugging Face model identificator to extract the MLP 
            module
    """
    pass