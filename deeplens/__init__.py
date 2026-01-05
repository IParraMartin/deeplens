"""DeepLens: A library for mechanistic interpretability using Sparse Autoencoders.

DeepLens provides tools for training and analyzing Sparse Autoencoders (SAEs)
for interpreting neural network activations. It includes feature extraction,
training utilities, and intervention capabilities.
"""

__version__ = "0.1.0"

from deeplens.sae import SparseAutoencoder
from deeplens.train import SAETrainer
from deeplens.extractor import FromHuggingFace, ExtractSingleSample
from deeplens.pipeline import pipeline
from deeplens.intervene import InterveneFeatures, ReinjectSingleSample

__all__ = [
    "SparseAutoencoder",
    "SAETrainer",
    "FromHuggingFace",
    "ExtractSingleSample",
    "InterveneFeatures",
    "ReinjectSingleSample",
    "pipeline"
]