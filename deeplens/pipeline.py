from deeplens.extractor import ExtractSingleSample
from deeplens.intervene import *

import torch

def pipeline(
        text: str,
        sae_model: str,
        sae_config: str,
        layer: int = 3,
        hf_model: str = "gpt2",
        feature: int = -1,
        alpha: float = 5.0,
        tok_position: int = -1,
        generate: bool = False,
        max_new_tokens: int = 25,
        temperature: float = 1.0,
        verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end pipeline for extracting, intervening on, and analyzing SAE features.

    This function provides a complete workflow for mechanistic interpretability analysis:
    extracts MLP activations from a language model, decodes them through a sparse autoencoder,
    intervenes on specific features, and reinjections the modified activations back into the
    model to observe behavioral changes.

    Args:
        text (str): Input text to process. Will be tokenized and processed by the language model.
        sae_model (str): Path to the trained sparse autoencoder model weights file (.pt or .pth).
        sae_config (str): Path to the SAE configuration YAML file or dictionary with hyperparameters.
        layer (int, optional): Transformer layer to extract activations from. Supports negative
            indexing (e.g., -1 for last layer). Defaults to 3.
        hf_model (str, optional): Name or path of the HuggingFace transformer model to use
            (e.g., "gpt2-medium"). Defaults to "gpt2".
        feature (int, optional): Index of the SAE feature to intervene on from the alive features
            list. Supports negative indexing. Defaults to -1 (last alive feature).
        alpha (float, optional): Intervention strength multiplier. Larger values create stronger
            feature manipulations. Can be negative to suppress features. Defaults to 5.0.
        tok_position (int, optional): Token position to analyze and intervene on. Supports
            negative indexing (e.g., -1 for last token). Defaults to -1.
        generate (bool, optional): Whether to generate continuation text after reinjection.
            If False, returns only the reconstructed input. Defaults to False.
        max_new_tokens (int, optional): Maximum number of new tokens to generate if generate=True.
            Must be positive. Defaults to 25.
        temperature (float, optional): Sampling temperature for text generation. Higher values
            increase randomness. Must be positive. Defaults to 1.0.
        verbose (bool, optional): Whether to print diagnostic information during execution.
            Defaults to False.

    Returns:
        A 3-tuple containing:
            - original: Model output with original unmodified activations
            - decoded: Model output with SAE-reconstructed activations (no intervention)
            - modified: Model output with intervened feature activations
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    if not isinstance(layer, int):
        raise ValueError(f"Layer must be an integer, got {layer}")
    
    if not isinstance(feature, int):
        raise ValueError(f"Feature must be an integer, got {feature}")
    
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"Alpha must be numeric, got {alpha}")
    
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive integer, got {max_new_tokens}")

    
    try:
        extractor = ExtractSingleSample(hf_model=hf_model, layer=layer)
        intervene = InterveneFeatures(sae_model=sae_model, sae_config=sae_config)
        reinject = ReinjectSingleSample(hf_model=hf_model)

        acts = extractor.get_mlp_acts(text)
        if acts is None:
            raise RuntimeError("Failed to extract activations. Returned 'None'.")
        
        alive_features = intervene.get_alive_features(acts, token_position=tok_position)
        if alive_features is None or len(alive_features) == 0:
            raise RuntimeError(f"No alive features found at position {tok_position}")
        
        if feature < -len(alive_features) or feature >= len(alive_features):
            raise ValueError(
                f"Feature index {feature} out of bounds for {len(alive_features)} features"
            )

        if verbose:
            print(f"{len(alive_features)} alive features discovered at position {tok_position}.")
            print(f"Modifying feature {alive_features[feature].item()}")
        
        original, decoded, modified = intervene.intervene_feature(
            activations=acts,
            feature=alive_features[feature].item(),
            alpha=alpha, 
            token_positions=tok_position
        )

        if original is None or decoded is None or modified is None:
            raise RuntimeError("Feature intervention returned 'None'")

        feature_versions = [original, decoded, modified]
        outputs = []
        for i, feature_acts in enumerate(feature_versions):
            try:
                out = reinject.reinject_and_generate(
                    text=text,
                    modified_activations=feature_acts,
                    layer=layer,
                    generate=generate,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                outputs.append(out)
            except Exception as e:
                raise RuntimeError(f"Failed to reinject feature version {i}: {e}")      
              
        return tuple(outputs)

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}")