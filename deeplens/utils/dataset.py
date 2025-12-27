from torch.utils.data import(
    Dataset, random_split, DataLoader
)
import torchaudio
import torch
import pandas as pd
import os


class AudioDatasetBuilder(Dataset):
    def __init__(
            self, 
            audio_dir: str = None, 
            annotations_file: str = None, 
            target_sample_rate: int = 22050, 
            num_samples: int = 22050, 
            device: str = "auto",
            transformation_args: dict = {"n_fft": 1024, "hop_length": 512, "n_mels": 64}
        ) -> None:
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
        """Get dataset length for Pytorch
        """
        if self.annotations is not None:
            return len(self.annotations)
        else:
            return len(self.file_list)

    def __getitem__(self, index) -> torch.Tensor:
        """Retrieves the item to be fed to a PyTorch nn.Module
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
        """Function to get the audio path used in __getitem__
        """
        if self.annotations is not None:
            fold = f"fold{self.annotations.iloc[index, 5]}"
            path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        else:
            path = os.path.join(self.audio_dir, self.file_list[index])
        return path

    def _audio_sample_label(self, index):
        """Function to get the labels from the file used in __getitem__
        """
        return self.annotations.iloc[index, 6]

    @torch.no_grad()
    def _resample_if_necessary(self, signal, sr) -> torch.Tensor:
        """Function to resample the sound used in __getitem__"""
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    @torch.no_grad()
    def _mix_down_if_necessary(self, signal) -> torch.Tensor:
        """Function to turn mono only if the sound is originally stereo used in __getitem__
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _truncate_if_necessary(self, signal) -> torch.Tensor:
        """_summary_
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    @torch.no_grad()
    def _pad_if_necessary(self, signal) -> torch.Tensor:
        """Function to pad the audio waveform if the sound is originally stereo
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            n_padding = self.num_samples - length_signal
            r_pad_dim = (0, n_padding)
            signal = torch.nn.functional.pad(signal, r_pad_dim)
        return signal
    
    def _apply_transformation(self, signal) -> torch.Tensor:
        """Converts waveform into a mel spectrogram
        """
        spectrogram = self.mel_spectrogram(signal)
        return spectrogram


class GetDataLoaders():
    def __init__(
            self, 
            dataset: Dataset = None,
            splits: list = [0.8, 0.2],
            batch_size: int = 16
        ) -> None:
        self.dataset = dataset
        self.splits = splits
        self.batch_size = batch_size
        
    def _prepare_loader(self) -> tuple[DataLoader, DataLoader]:
        """Returns the prepared dataloaders for a PyTorch model
        """
        train, test = random_split(self.dataset, self.splits)
        train_loader = DataLoader(train, self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test, self.batch_size, shuffle=False, pin_memory=True)
        return train_loader, test_loader


class ActivationsDataset(Dataset):
    """Custom dataset for activations that returns tensors directly
    """
    def __init__(self, activations: torch.Tensor):
        super().__init__()
        self.activations = activations
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]
    

class ActivationsDatasetBuilder():
    def __init__(
            self, 
            activations: torch.Tensor = None, 
            splits: list = [0.8, 0.2],
            batch_size: int = 16,
            norm: bool = True
        ):
        self.activations = torch.load(activations, weights_only=True)
        self.splits = splits
        self.batch_size = batch_size
        self.norm = norm
        self.normalize()

    def set_tensor_dataset(self) -> Dataset:
        """Converts the saved features into a tensor dataset
        """
        return ActivationsDataset(self.activations)

    def get_dataloaders(self) -> tuple:
        """Returns the dtaloaders for training
        """
        data = self.set_tensor_dataset()
        train, eval = random_split(data, lengths=self.splits)
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval, batch_size=self.batch_size, shuffle=False)
        return train_loader, eval_loader

    @torch.no_grad()
    def normalize(self):
        """Normalizes the activations dataset
        """
        if self.norm:
            mean = self.activations.mean(dim=0, keepdim=True)
            std = self.activations.std(dim=0, keepdim=True)
            self.activations = (self.activations - mean) / (std + 1e-8)


if __name__ == "__main__":
    AUDIO_DIR = "/Users/inigoparra/Desktop/Projects/TIMITPhones/timit-phones/phonemes"
    data = AudioDatasetBuilder(
        audio_dir=AUDIO_DIR,
        num_samples=16000,
        target_sample_rate=16000,
        transformation_args=None
    )
    loader_pipeline = GetDataLoaders(dataset=data)
    train, test = loader_pipeline._prepare_loader()
    example = next(iter(train)).squeeze()
