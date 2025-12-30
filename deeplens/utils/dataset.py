from torch.utils.data import(
    Dataset, random_split, DataLoader
)
import torchaudio
import torch
import pandas as pd
import os


__all__ = [
    "AudioDatasetBuilder",
    "GetDataLoaders",
    "ActivationsDatasetBuilder"
]


class AudioDatasetBuilder(Dataset):
    """PyTorch Dataset for loading and preprocessing audio files with optional mel spectrogram transformation.

    This dataset handles audio files in various formats (WAV, MP3, FLAC), performs preprocessing
    operations like resampling, mono conversion, and padding, and optionally converts waveforms
    to mel spectrograms. Supports both labeled and unlabeled datasets.
    """
    def __init__(
            self, 
            audio_dir: str = None, 
            annotations_file: str = None, 
            target_sample_rate: int = 22050, 
            num_samples: int = 22050, 
            device: str = "auto",
            transformation_args: dict = {"n_fft": 1024, "hop_length": 512, "n_mels": 64}
        ) -> None:
        """Initialize the AudioDatasetBuilder with audio preprocessing parameters.

        Args:
            audio_dir (str, optional): Path to directory containing audio files. The directory
                should contain files with extensions .wav, .mp3, or .flac. Defaults to None.
            annotations_file (str, optional): Path to CSV file containing annotations/labels.
                If None, dataset returns only audio without labels. Defaults to None.
            target_sample_rate (int, optional): Target sampling rate in Hz for resampling.
                All audio will be resampled to this rate. Defaults to 22050.
            num_samples (int, optional): Target number of samples per audio clip. Audio will
                be truncated or zero-padded to this length. Defaults to 22050.
            device (str, optional): Device for tensor operations. Can be "auto" for automatic
                selection, "cuda", "mps", or "cpu". Defaults to "auto".
            transformation_args (dict, optional): Dictionary containing parameters for mel
                spectrogram transformation. Must include keys: "n_fft", "hop_length", "n_mels".
                If None, raw waveforms are returned without transformation. Defaults to
                {"n_fft": 1024, "hop_length": 512, "n_mels": 64}.
        """
        super().__init__()
        self.audio_dir = audio_dir 
        self.file_list = [
            f for f in os.listdir(self.audio_dir) 
            if f.endswith(('.wav', '.mp3', '.flac'))
        ]
        if annotations_file is not None:
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = None

        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )     
        else:
            self.device = torch.device(device)

        self.transformation_args = transformation_args
        if transformation_args is not None:
            assert {"n_fft", "hop_length", "n_mels"}.issubset(transformation_args.keys()), \
            "Missing arguments. Please provide n_fft, hop_length, and n_mels."
            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft = transformation_args["n_fft"],
                hop_length=transformation_args["hop_length"],
                n_mels=transformation_args["n_mels"]
            ).to(self.device)
        else:
            self.mel_spectrogram = None

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of audio samples. Returns length of annotations if provided,
                otherwise returns number of audio files in the directory.
        """
        if self.annotations is not None:
            return len(self.annotations)
        else:
            return len(self.file_list)

    def __getitem__(self, index) -> torch.Tensor | tuple[torch.Tensor, int]:
        """Retrieve and preprocess an audio sample at the specified index.

        Loads the audio file, applies preprocessing (resampling, mono conversion, padding/truncation),
        optionally transforms to mel spectrogram, and returns with label if annotations are available.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor | tuple[torch.Tensor, int]: If annotations are provided, returns a tuple
                of (processed_audio, label). Otherwise, returns only the processed audio tensor.
                Audio shape depends on transformation: (1, num_samples) for waveform or
                (1, n_mels, time_steps) for mel spectrogram.
        """
        audio_sample = self._audio_sample_path(index)    
        signal, sr = torchaudio.load(audio_sample)
        signal = signal.to(self.device)                   
        signal = self._resample_if_necessary(signal, sr)  
        signal = self._mix_down_if_necessary(signal)  
        signal = self._truncate_if_necessary(signal)    
        signal = self._pad_if_necessary(signal)
        if self.transformation_args is not None:
            signal = self._apply_transformation(signal)
        if self.annotations is not None:
            label = self._audio_sample_label(index)        
            return signal, label
        else:
            return signal

    def _audio_sample_path(self, index) -> str:
        """Construct the file path for an audio sample at the given index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            str: Full path to the audio file. If annotations are provided, constructs path
                using fold structure; otherwise, uses direct file listing.
        """
        if self.annotations is not None:
            fold = f"fold{self.annotations.iloc[index, 5]}"
            path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        else:
            path = os.path.join(self.audio_dir, self.file_list[index])
        return path

    def _audio_sample_label(self, index) -> int:
        """Retrieve the label for an audio sample at the given index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            int: Label value from the annotations file (column 6).
        """
        return self.annotations.iloc[index, 6]

    @torch.no_grad()
    def _resample_if_necessary(self, signal, sr) -> torch.Tensor:
        """Resample audio signal to target sample rate if necessary.

        Args:
            signal (torch.Tensor): Input audio waveform.
            sr (int): Current sample rate of the audio signal.

        Returns:
            torch.Tensor: Resampled audio at target_sample_rate, or original signal
                if already at the target rate.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    @torch.no_grad()
    def _mix_down_if_necessary(self, signal) -> torch.Tensor:
        """Convert stereo audio to mono by averaging channels if necessary.

        Args:
            signal (torch.Tensor): Input audio with shape (channels, samples).

        Returns:
            torch.Tensor: Mono audio with shape (1, samples). If input is already mono,
                returns unchanged.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _truncate_if_necessary(self, signal) -> torch.Tensor:
        """Truncate audio signal to target length if it exceeds num_samples.

        Args:
            signal (torch.Tensor): Input audio waveform.

        Returns:
            torch.Tensor: Truncated audio limited to num_samples length, or original
                signal if already shorter.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    @torch.no_grad()
    def _pad_if_necessary(self, signal) -> torch.Tensor:
        """Zero-pad audio signal to target length if it's shorter than num_samples.

        Args:
            signal (torch.Tensor): Input audio waveform.

        Returns:
            torch.Tensor: Zero-padded audio extended to num_samples length, or original
                signal if already long enough.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            n_padding = self.num_samples - length_signal
            r_pad_dim = (0, n_padding)
            signal = torch.nn.functional.pad(signal, r_pad_dim)
        return signal
    
    def _apply_transformation(self, signal) -> torch.Tensor:
        """Transform audio waveform to mel spectrogram representation.

        Args:
            signal (torch.Tensor): Input audio waveform with shape (1, num_samples).

        Returns:
            torch.Tensor: Mel spectrogram with shape (1, n_mels, time_steps), where
                time_steps depends on n_fft and hop_length parameters.
        """
        spectrogram = self.mel_spectrogram(signal)
        return spectrogram


class GetDataLoaders():
    """Utility class for creating train and test DataLoaders from a PyTorch Dataset.

    Handles dataset splitting and DataLoader creation with consistent parameters.
    """
    def __init__(
            self, 
            dataset: Dataset = None,
            splits: list = [0.8, 0.2],
            batch_size: int = 16
        ) -> None:
        """Initialize the DataLoader factory.

        Args:
            dataset (Dataset, optional): PyTorch Dataset to split and load. Defaults to None.
            splits (list, optional): List of two floats representing train and test split
                proportions. Must sum to 1.0. Defaults to [0.8, 0.2].
            batch_size (int, optional): Number of samples per batch. Defaults to 16.
        """
        self.dataset = dataset
        self.splits = splits
        self.batch_size = batch_size
        
    def _prepare_loader(self) -> tuple[DataLoader, DataLoader]:
        """Create train and test DataLoaders with the specified configuration.

        Splits the dataset according to the split proportions and creates two DataLoaders
        with appropriate settings for training and testing.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader).
                Training loader has shuffle=True, test loader has shuffle=False. Both use
                pin_memory=True for faster data transfer to GPU.
        """
        train, test = random_split(self.dataset, self.splits)
        train_loader = DataLoader(train, self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test, self.batch_size, shuffle=False, pin_memory=True)
        return train_loader, test_loader


class ActivationsDataset(Dataset):
    """Lightweight PyTorch Dataset wrapper for pre-computed activation tensors.

    Provides a simple Dataset interface for tensors of neural network activations,
    enabling use with PyTorch DataLoader for batching and iteration.
    """
    def __init__(self, activations: torch.Tensor):
        """Initialize the dataset with activation tensors.

        Args:
            activations (torch.Tensor): Tensor containing pre-computed activations
                with shape (num_samples, feature_dim).
        """
        super().__init__()
        self.activations = activations
    
    def __len__(self) -> int:
        """Get the number of activation samples.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.activations)
    
    def __getitem__(self, idx) -> torch.Tensor:
        """Retrieve activation tensor at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Activation tensor at the given index.
        """
        return self.activations[idx]
    

class ActivationsDatasetBuilder():
    """Builder class for creating DataLoaders from saved activation tensors.

    Loads pre-computed activations from disk, applies optional normalization, and creates
    train/validation DataLoaders for training sparse autoencoders or other downstream models.
    """
    def __init__(
            self, 
            activations: torch.Tensor = None, 
            splits: list = [0.8, 0.2],
            batch_size: int = 16,
            norm: bool = True
        ):
        """Initialize the builder and load activations from disk.

        Args:
            activations (torch.Tensor | str, optional): Path to a .pt file containing
                saved activation tensors, or a tensor directly. Defaults to None.
            splits (list, optional): List of two floats representing train and validation
                split proportions. Must sum to 1.0. Defaults to [0.8, 0.2].
            batch_size (int, optional): Number of samples per batch for DataLoaders.
                Defaults to 16.
            norm (bool, optional): Whether to apply z-score normalization (standardization)
                to the activations. Defaults to True.
        """
        self.activations = torch.load(activations, weights_only=True)
        self.splits = splits
        self.batch_size = batch_size
        self.norm = norm
        self.normalize()

    def set_tensor_dataset(self) -> Dataset:
        """Create a PyTorch Dataset from the loaded activations.

        Returns:
            Dataset: ActivationsDataset instance wrapping the activation tensors.
        """
        return ActivationsDataset(self.activations)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders from the activations.

        Splits the dataset according to the specified proportions and creates two DataLoaders
        with appropriate settings for training and evaluation.

        Returns:
            tuple: A tuple containing (train_loader, eval_loader).
                Training loader has shuffle=True for randomized batching, evaluation loader
                has shuffle=False for consistent evaluation.
        """
        data = self.set_tensor_dataset()
        train, eval = random_split(data, lengths=self.splits)
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval, batch_size=self.batch_size, shuffle=False)
        return train_loader, eval_loader

    @torch.no_grad()
    def normalize(self) -> None:
        """Apply z-score normalization to the activations in-place.

        Standardizes the activations by subtracting the mean and dividing by the standard
        deviation (computed along the batch dimension). Adds small epsilon (1e-8) to prevent
        division by zero. Only applies if norm=True was set during initialization.

        Returns:
            None: Modifies self.activations in-place.
        """
        if self.norm:
            mean = self.activations.mean(dim=0, keepdim=True)
            std = self.activations.std(dim=0, keepdim=True)
            self.activations = (self.activations - mean) / (std + 1e-8)
