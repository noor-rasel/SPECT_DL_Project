import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

class SPECT2CTDataset(Dataset):
    """
    Dataset class for loading SPECT photopeak, scatter, and CT-derived attnmap.
    Expects (H, W, D) input and returns (C, D, H, W) for PyTorch Conv3D.
    """
    def __init__(self, data_dir, photo_key="photopeak", scatter_key="scatter", target_key="attnmap"):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".mat")
        ])
        if not self.files:
            raise FileNotFoundError(f"No .mat files in {data_dir}")

        self.photo_key = photo_key
        self.scatter_key = scatter_key
        self.target_key = target_key

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the .mat file
        d = loadmat(self.files[idx])

        # Convert to numpy arrays with float32 precision
        photo = np.asarray(d[self.photo_key], dtype=np.float32)
        scat  = np.asarray(d[self.scatter_key], dtype=np.float32)
        targ  = np.asarray(d[self.target_key], dtype=np.float32)

        # Sanity check for matching dimensions
        if photo.shape != scat.shape or photo.shape != targ.shape:
            raise ValueError(f"Shape mismatch in {self.files[idx]}: "
                             f"photo {photo.shape}, scat {scat.shape}, target {targ.shape}")

        # x: Combine photopeak and scatter into 2 channels -> (2, H, W, D)
        x = np.stack([photo, scat], axis=0)
        # y: Add channel dimension to target -> (1, H, W, D)
        y = targ[np.newaxis, ...]

        # Convert to Tensor and Permute to (C, D, H, W) for PyTorch Conv3D
        # From (2, 128, 128, 22) -> (2, 22, 128, 128)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        y = torch.from_numpy(y).permute(0, 3, 1, 2)

        return x, y