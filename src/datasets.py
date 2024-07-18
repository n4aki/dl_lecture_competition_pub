import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from scipy.signal import butter, lfilter

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess: bool = False) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        self.preprocess = preprocess

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path)).float()
        
        if self.preprocess:
            X = self.preprocess_data(X)
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path)).long()
            
            return X, y, subject_idx
        else:
            return X, subject_idx

    def preprocess_data(self, X):
        X = self.bandpass_filter(X)
        X = self.normalize(X)
        X = self.baseline_correction(X)
        return X
    
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, X, lowcut=0.5, highcut=30.0, fs=200.0, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, X.numpy(), axis=-1)
        return torch.from_numpy(y).float()

    def normalize(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        return (X - mean) / std

    def baseline_correction(self, X, baseline_window=(0, 50)):
        baseline = X[:, baseline_window[0]:baseline_window[1]].mean(dim=-1, keepdim=True)
        return X - baseline
    
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
