# Installation

## Creating a Virtual Environment

Before installing any dependencies, I recommend creating a new virtual environment to avoid library conflicts.

```bash
conda create -n deeplens python=3.11
conda activate deeplens
```

The library should work with Python 3.11+, but I recommend using Python 3.11 specifically, as it was used during development. This will help prevent version mismatches and dependency conflicts.

## Standard Installation

To install DeepLens and its dependencies, run the following commands:

```bash
pip install -e .
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Alternative Installation

If you encounter any errors with the standard installation, try the manual installation method:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Common Version Issues

During installation, you may encounter version conflicts. The most common issues involve `numpy` and `torch` versions.

### NumPy Version

We recommend installing `numpy<2`, as this version was used during development and has shown the fewest conflicts with other libraries (e.g., `scipy`).

### PyTorch Installation

**For CUDA support:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For MPS/CPU usage:**

```bash
pip install torch torchvision torchaudio
```

## Future Updates

PyPI installation is not yet available. Future versions will support easier installation via pip.

---

Ready to dive deeper? Check out the [Quickstart Guide](quickstart.md) for complete examples!