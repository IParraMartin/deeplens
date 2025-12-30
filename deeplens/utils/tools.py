import torch

def get_device(device: str = "auto") -> torch.device:
    """Get the device in which to process run the Pytorch
    operations
    """
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)

def get_mlp_module():
    """Access the correct MLP module of a given model architecture
    """
    pass