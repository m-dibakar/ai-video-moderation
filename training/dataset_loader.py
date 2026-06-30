"""
Dataset Loader for deepghs/nsfw_detect dataset.

Label mapping:
    Safe (0):  neutral, drawings
    NSFW (1):  sexy, porn, hentai

Real OTT imbalance simulation:
    We undersample NSFW to create 95/5, 80/20, 50/50 splits
    for the loss function comparison experiment.
"""

import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Label mapping ──────────────────────────────────────────────────────────
SAFE_CLASSES  = ["neutral", "drawings"]
NSFW_CLASSES  = ["sexy", "porn", "hentai"]

def load_all_samples(dataset_dir: str) -> tuple[list, list]:
    """
    Scan dataset folder and return (safe_paths, nsfw_paths).
    
    Returns:
        safe_paths: list of (image_path, 0)
        nsfw_paths: list of (image_path, 1)
    """
    dataset_dir = Path(dataset_dir)
    safe_paths = []
    nsfw_paths = []
    
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    for class_name in SAFE_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                safe_paths.append((str(img_path), 0))
    
    for class_name in NSFW_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                nsfw_paths.append((str(img_path), 1))
    
    print(f"Found {len(safe_paths)} safe images")
    print(f"Found {len(nsfw_paths)} NSFW images")
    return safe_paths, nsfw_paths


def build_imbalanced_split(
    safe_paths: list,
    nsfw_paths: list,
    imbalance_ratio: float = 0.05,
    total_samples: int = 6000,
    val_fraction: float = 0.2,
    seed: int = 42
) -> tuple[list, list]:
    """
    Build train/val split with controlled class imbalance.
    
    Args:
        imbalance_ratio: Fraction of NSFW in final dataset
                         0.05 = 5% NSFW (realistic OTT)
                         0.20 = 20% NSFW (mild imbalance)
                         0.50 = 50% NSFW (balanced baseline)
        total_samples:   Total dataset size
        val_fraction:    Fraction held out for validation
    
    Returns:
        train_samples, val_samples — each is list of (path, label)
    """
    random.seed(seed)
    
    n_nsfw = int(total_samples * imbalance_ratio)
    n_safe = total_samples - n_nsfw
    
    # Clamp to available samples
    n_nsfw = min(n_nsfw, len(nsfw_paths))
    n_safe = min(n_safe, len(safe_paths))
    
    selected_safe = random.sample(safe_paths, n_safe)
    selected_nsfw = random.sample(nsfw_paths, n_nsfw)
    
    all_samples = selected_safe + selected_nsfw
    random.shuffle(all_samples)
    
    n_val = int(len(all_samples) * val_fraction)
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    
    # Stats
    train_pos = sum(1 for _, l in train_samples if l == 1)
    train_neg = sum(1 for _, l in train_samples if l == 0)
    val_pos   = sum(1 for _, l in val_samples if l == 1)
    val_neg   = sum(1 for _, l in val_samples if l == 0)
    
    print(f"\n{'='*55}")
    print(f"DATASET SPLIT (imbalance_ratio={imbalance_ratio})")
    print(f"  Train: {train_neg} safe + {train_pos} NSFW = {len(train_samples)} total")
    print(f"  Val:   {val_neg} safe + {val_pos} NSFW = {len(val_samples)} total")
    print(f"  Train positive rate: {train_pos/len(train_samples)*100:.1f}%")
    print(f"{'='*55}")
    
    return train_samples, val_samples


class NSFWDataset(Dataset):
    """
    PyTorch Dataset for NSFW classification.
    
    Takes a list of (image_path, label) tuples.
    Applies standard ViT preprocessing transforms.
    """
    def __init__(self, samples: list, augment: bool = False):
        self.samples = samples
        
        if augment:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation transforms — no augmentation
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            # Corrupt image — return a black image instead of crashing
            print(f"Warning: Could not load {img_path}: {e}")
            img = self.transform(Image.new("RGB", (224, 224), (0, 0, 0)))
        
        import torch
        return img, torch.tensor(label, dtype=torch.float32)


# ── Quick verification ─────────────────────────────────────────────────────
if __name__ == "__main__":
    DATASET_DIR = "nsfw_dataset_v1"
    
    safe_paths, nsfw_paths = load_all_samples(DATASET_DIR)
    
    # Test all three imbalance ratios we'll use in the experiment
    for ratio in [0.50, 0.20, 0.05]:
        train, val = build_imbalanced_split(
            safe_paths, nsfw_paths,
            imbalance_ratio=ratio,
            total_samples=6000
        )
    
    # Test dataset loads correctly
    print("\nTesting dataset loading...")
    train_ds = NSFWDataset(train, augment=True)
    img, label = train_ds[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Label: {label.item()}")
    print(f"Pixel range: [{img.min():.2f}, {img.max():.2f}]")
    print("\nDataset loader ready.")
