
import os
import gc
import cv2
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

# 1. Preprocessing Class for DICOM Series

class DICOMPreprocessor:
    def __init__(self, target_shape=(32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_series(self, series_path: str) -> List[pydicom.Dataset]:
        datasets = []
        for root, _, files in os.walk(series_path):
            for f in files:
                if f.endswith(".dcm"):
                    try:
                        ds = pydicom.dcmread(os.path.join(root, f))
                        datasets.append(ds)
                    except:
                        continue
        return datasets

    def extract_pixel_array(self, ds):
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:  # if RGB or multi-slice single file
            img = img[img.shape[0] // 2]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return img

    def normalize(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        p1, p99 = 0, 500  # fixed range for CT
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-6)
        return (img * 255).astype(np.uint8)

    def resize_volume(self, volume):
        zoom_factors = [self.target_depth / volume.shape[0],
                        self.target_height / volume.shape[1],
                        self.target_width / volume.shape[2]]
        volume = ndimage.zoom(volume, zoom_factors, order=1)
        return volume.astype(np.uint8)

    def process(self, series_path: str) -> np.ndarray:
        datasets = self.load_series(series_path)
        slices = []
        for ds in datasets:
            img = self.extract_pixel_array(ds)
            img = self.normalize(img)
            img = cv2.resize(img, (self.target_width, self.target_height))
            slices.append(img)
        if not slices:
            return np.zeros((self.target_depth, self.target_height, self.target_width), dtype=np.uint8)
        volume = np.stack(slices, axis=0)
        volume = self.resize_volume(volume)
        return volume


# 2. Dataset Class

class RSNADataset(Dataset):
    def __init__(self, volume_paths, labels, transform=None):
        self.volume_paths = volume_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = DICOMPreprocessor()

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        volume_path = self.volume_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        volume = self.preprocessor.process(volume_path)  # (D,H,W)
        volume = volume.transpose(1, 2, 0)  # (H,W,D)

        if self.transform:
            volume = self.transform(image=volume)["image"]

        return volume, label


# 3. Model (Vision Transformer)

def get_model(num_classes, in_chans=32, model_name="vit_base_patch16_384"):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_chans
    )
    return model


# 4. Training 

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, gts = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating", leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds.append(torch.sigmoid(outputs).cpu())
            gts.append(targets.cpu())

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    return running_loss / len(loader.dataset), preds, gts



def run_training(volume_paths, labels, num_classes, n_folds=5, epochs=10, lr=1e-4, batch_size=2, save_dir="./checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Transform
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2(),
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(volume_paths, np.argmax(labels, axis=1))):
        print(f"\n===== Fold {fold+1}/{n_folds} =====")

        train_ds = RSNADataset([volume_paths[i] for i in train_idx],
                               [labels[i] for i in train_idx],
                               transform=transform)
        val_ds = RSNADataset([volume_paths[i] for i in val_idx],
                             [labels[i] for i in val_idx],
                             transform=transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        model = get_model(num_classes=num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler()

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, preds, gts = validate_one_epoch(model, val_loader, criterion, device)

            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(save_dir, f"vit_fold{fold}_best.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
                print(f"✅ Saved best model at {ckpt_path}")

        del model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 5. Entry Point

if __name__ == "__main__":
    # Example usage (replace with real data paths & labels)
    volume_paths = ["/path/to/series1", "/path/to/series2", "/path/to/series3"]
    labels = np.array([[1,0,0,1,0,0,0,1,0,0,0,0,0,0,0],  # one-hot multi-labels
                       [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0],
                       [0,0,1,0,0,0,0,0,0,1,0,0,0,0,1]])

    num_classes = labels.shape[1]

    run_training(volume_paths, labels, num_classes=num_classes, n_folds=5, epochs=10, lr=1e-4, batch_size=2)
