import os,cupy as cp
import cv2
import torch
from torch.utils.data import Dataset

class DRDatasetGPU(Dataset):
    def __init__(self, image_dir, dataframe, test_mode=False, transform=None, exts=(".jpg", ".jpeg", ".png")):
        self.entries = []
        self.test_mode = test_mode
        # ImageNet mean/std in 0-255 scale (as float32 CuPy arrays)
        self.mean = cp.array([123.675, 116.28, 103.53], dtype=cp.float32)
        self.std  = cp.array([58.395, 57.12, 57.375], dtype=cp.float32)
        for _, row in dataframe.iterrows():
            image_id = row["image"]
            label = -1 if self.test_mode else int(row["level"])
            for ext in exts:
                path = os.path.join(image_dir, f"{image_id}{ext}")
                if os.path.isfile(path):
                    self.entries.append((path, label))
                    break

        print(f"[PyTorch Dataset] Valid images found: {len(self.entries)}")
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        image_id = os.path.splitext(os.path.basename(path))[0]

        # Load image on CPU
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize using OpenCV
        img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_AREA)

        # Move to GPU
        img_cp = cp.asarray(img, dtype=cp.float32)

        # Transpose to (C, H, W) BEFORE normalization
        img_cp = cp.transpose(img_cp, (2, 0, 1))  # now (3, 380, 380)

        # Reshape mean and std for broadcasting
        mean = self.mean.reshape((3, 1, 1))
        std = self.std.reshape((3, 1, 1))

        # Normalize: (img - mean) / std
        img_cp = (img_cp - mean) / std

        # Convert to PyTorch CUDA tensor
        img_tensor = torch.from_dlpack(cp.asarray(img_cp).toDlpack()).float()

        if self.test_mode:
            return img_tensor, image_id
        else:
            return img_tensor, torch.tensor(label, dtype=torch.long).to(img_tensor.device)