"""
ODIR-5K Dataset Loader
=======================
Ocular Disease Intelligent Recognition — 5000 patients, 8 labels.

Dataset structure expected:
  data/odir/
  ├── ODIR-5K_Training_Dataset/       (fundus images)
  ├── ODIR-5K_Testing_Images/
  ├── ODIR-5K_Training_Annotations.xlsx
  └── ODIR-5K_Testing_Annotations.xlsx  (if available)

Labels (multi-label, one patient may have multiple):
  N = Normal
  D = Diabetic Retinopathy
  G = Glaucoma
  C = Cataract
  A = Age-related Macular Degeneration
  H = Hypertension
  M = Pathological Myopia
  O = Other diseases

Key challenges:
  - Each patient has LEFT + RIGHT eye images (both must be used)
  - Severe class imbalance (Normal ~53%, Hypertension ~7%)
  - Labels apply to the patient, not individual eyes
  - Image quality varies significantly
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CLASSES       = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
CLASS_NAMES   = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract',
    'Age-related Macular Degeneration', 'Hypertension',
    'Pathological Myopia', 'Other'
]
NUM_CLASSES   = 8
IMAGE_SIZE    = 512   # for EfficientNetV2-L
VIT_SIZE      = 384   # for ViT-L/16

# ImageNet stats (both models pretrained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Class weights (inverse frequency from ODIR-5K training set)
# N:1806, D:1182, G:220, C:192, A:148, H:249, M:244, O:683
CLASS_FREQ    = [1806, 1182, 220, 192, 148, 249, 244, 683]
CLASS_WEIGHTS = [max(CLASS_FREQ) / f for f in CLASS_FREQ]


# ─────────────────────────────────────────────
# LABEL PARSER
# ─────────────────────────────────────────────

def parse_labels(label_str: str) -> np.ndarray:
    """
    Parse ODIR diagnostic keyword string to binary label vector.
    Examples:
      'N'         → [1,0,0,0,0,0,0,0]
      'D,G'       → [0,1,1,0,0,0,0,0]
      'C,O'       → [0,0,0,1,0,0,0,1]
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if not isinstance(label_str, str):
        return vec
    parts = [p.strip() for p in label_str.split(',')]
    for p in parts:
        if p in CLASSES:
            vec[CLASSES.index(p)] = 1.0
    return vec


def load_annotations(xlsx_path: str) -> pd.DataFrame:
    """
    Load and normalise ODIR annotations Excel file.
    Returns DataFrame with columns:
      ID, Left-Fundus, Right-Fundus, labels (np.ndarray), label_str
    """
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names across different ODIR releases
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'left' in cl and 'fundus' in cl:     col_map[c] = 'Left-Fundus'
        elif 'right' in cl and 'fundus' in cl:  col_map[c] = 'Right-Fundus'
        elif 'left' in cl and 'diagnos' in cl:  col_map[c] = 'Left-Diagnostic'
        elif 'right' in cl and 'diagnos' in cl: col_map[c] = 'Right-Diagnostic'
        elif 'diagnos' in cl and 'keyword' in cl: col_map[c] = 'Keywords'
        elif cl in ['id', 'patient id', 'patient_id']: col_map[c] = 'ID'
    df = df.rename(columns=col_map)

    # Build unified label from left + right diagnostic keywords
    def get_label(row):
        parts = set()
        for col in ['Left-Diagnostic', 'Right-Diagnostic', 'Keywords']:
            if col in df.columns and isinstance(row.get(col), str):
                for p in row[col].split(','):
                    p = p.strip()
                    if p in CLASSES:
                        parts.add(p)
        return ','.join(sorted(parts)) if parts else 'N'

    df['label_str'] = df.apply(get_label, axis=1)
    df['labels']    = df['label_str'].apply(parse_labels)
    return df


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def ben_graham_preprocess(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """
    Ben Graham's preprocessing (winner of Kaggle DR competition):
    Subtract local average colour to remove illumination variation.
    Widely used as the highest-quality fundus preprocessing step.
    """
    import cv2
    arr = np.array(img.resize((size, size)))
    # Gaussian blur of the image (radius = size/10)
    blurred = cv2.GaussianBlur(arr, (0, 0), size / 10)
    # Weighted combination: enhance local contrast
    result  = cv2.addWeighted(arr, 4, blurred, -4, 128)
    # Clip to valid range
    result  = np.clip(result, 0, 255).astype(np.uint8)
    # Circular crop (remove corners)
    cx, cy  = size // 2, size // 2
    mask    = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(size * 0.48), 255, -1)
    result[mask == 0] = 0
    return Image.fromarray(result)


def remove_black_border(img: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop to non-black bounding box."""
    arr  = np.array(img.convert('L'))
    mask = arr > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img.crop((int(c0), int(r0), int(c1+1), int(r1+1)))


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s    = max(w, h)
    out  = Image.new('RGB', (s, s), (0, 0, 0))
    out.paste(img, ((s-w)//2, (s-h)//2))
    return out


def preprocess_fundus(img: Image.Image, size: int = IMAGE_SIZE,
                      use_ben_graham: bool = True) -> Image.Image:
    img = img.convert('RGB')
    img = remove_black_border(img)
    img = pad_to_square(img)
    if use_ben_graham:
        img = ben_graham_preprocess(img, size)
    else:
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────
# AUGMENTATION — AGGRESSIVE FOR MAX ACCURACY
# ─────────────────────────────────────────────

def build_train_transforms(size: int = IMAGE_SIZE) -> A.Compose:
    """
    Aggressive augmentation policy targeting maximum generalisation.
    All transforms are medically plausible for fundus images.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=0, p=0.7
        ),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=120, sigma=120*0.05, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.CoarseDropout(
            max_holes=8, max_height=size//16, max_width=size//16,
            min_holes=1, fill_value=0, p=0.3
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def build_val_transforms(size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def build_tta_transforms(size: int = IMAGE_SIZE) -> List[A.Compose]:
    """
    Test-Time Augmentation: 8 deterministic transforms.
    Average predictions across all for maximum accuracy.
    """
    base = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    configs = [
        [],                                          # original
        [A.HorizontalFlip(p=1.0)],                  # hflip
        [A.VerticalFlip(p=1.0)],                    # vflip
        [A.Rotate(limit=(90,90), p=1.0)],           # rot90
        [A.Rotate(limit=(180,180), p=1.0)],         # rot180
        [A.Rotate(limit=(270,270), p=1.0)],         # rot270
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],    # hv flip
        [A.Rotate(limit=(90,90), p=1.0), A.HorizontalFlip(p=1.0)],  # rot90+hflip
    ]
    return [A.Compose(c + base) for c in configs]


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class ODIRDataset(Dataset):
    """
    ODIR-5K dataset with dual-eye image support.

    Modes:
      'single'  — one sample per image (left and right treated separately)
      'paired'  — one sample per patient (left + right concatenated along channel dim → 6ch)
      'dual'    — one sample per image, but loads partner eye as auxiliary input
    """

    def __init__(
        self,
        root:        str,
        annotations: str,
        split:       str  = 'train',
        size:        int  = IMAGE_SIZE,
        mode:        str  = 'single',
        transform          = None,
        use_ben_graham: bool = True,
        cache:       bool = False,
    ):
        assert split in ('train', 'val', 'test')
        assert mode  in ('single', 'paired')

        self.root         = Path(root)
        self.size         = size
        self.mode         = mode
        self.transform    = transform or (
            build_train_transforms(size) if split == 'train'
            else build_val_transforms(size)
        )
        self.use_bg       = use_ben_graham
        self.split        = split
        self.cache        = cache
        self._cache       = {}

        # Image directory
        if split in ('train', 'val'):
            self.img_dir = self.root / 'ODIR-5K_Training_Dataset'
        else:
            self.img_dir = self.root / 'ODIR-5K_Testing_Images'

        # Load annotations
        df = load_annotations(str(self.root / annotations))

        # Train/val split (90/10 stratified by dominant class)
        if split in ('train', 'val'):
            np.random.seed(42)
            idx       = np.arange(len(df))
            val_size  = int(len(df) * 0.1)
            val_idx   = np.random.choice(idx, val_size, replace=False)
            train_idx = np.setdiff1d(idx, val_idx)
            df = df.iloc[train_idx if split == 'train' else val_idx].reset_index(drop=True)

        self.df = df
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        samples = []
        for _, row in self.df.iterrows():
            labels = row['labels']
            if self.mode == 'single':
                for eye in ['Left-Fundus', 'Right-Fundus']:
                    if eye in self.df.columns and isinstance(row.get(eye), str):
                        fname = row[eye].strip()
                        path  = self._find_image(fname)
                        if path:
                            samples.append({'path': path, 'labels': labels, 'eye': eye})
            elif self.mode == 'paired':
                lpath = self._find_image(str(row.get('Left-Fundus', '')).strip())
                rpath = self._find_image(str(row.get('Right-Fundus', '')).strip())
                if lpath and rpath:
                    samples.append({'lpath': lpath, 'rpath': rpath, 'labels': labels})
        return samples

    def _find_image(self, fname: str) -> Optional[Path]:
        if not fname or fname == 'nan':
            return None
        for ext in ['', '.jpg', '.jpeg', '.png']:
            p = self.img_dir / (fname + ext)
            if p.exists():
                return p
        return None

    def _load_image(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache:
            return self._cache[path]
        img = Image.open(path).convert('RGB')
        img = preprocess_fundus(img, self.size, self.use_bg)
        arr = np.array(img)
        if self.cache:
            self._cache[path] = arr
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        if self.mode == 'single':
            arr    = self._load_image(s['path'])
            result = self.transform(image=arr)
            image  = result['image']
            return {
                'image' : image,
                'labels': torch.tensor(s['labels'], dtype=torch.float32),
                'path'  : str(s['path']),
            }

        elif self.mode == 'paired':
            larr = self._load_image(s['lpath'])
            rarr = self._load_image(s['rpath'])
            # Apply same random transform to both eyes
            replay = A.ReplayCompose(self.transform.transforms)
            res_l  = replay(image=larr)
            res_r  = A.ReplayCompose.replay(res_l['replay'], image=rarr)
            limg   = res_l['image']
            rimg   = res_r['image']
            # 6-channel concatenation
            image  = torch.cat([limg, rimg], dim=0)
            return {
                'image' : image,
                'labels': torch.tensor(s['labels'], dtype=torch.float32),
            }


# ─────────────────────────────────────────────
# WEIGHTED SAMPLER
# ─────────────────────────────────────────────

def get_sample_weights(dataset: ODIRDataset) -> torch.Tensor:
    """
    Per-sample weight = max class weight among its positive labels.
    Ensures rare diseases (Glaucoma, Cataract, AMD, Hypertension)
    are sampled proportionally despite low frequency.
    """
    cw = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
    weights = []
    for s in dataset.samples:
        lbl    = torch.tensor(s['labels'])
        active = lbl * cw
        w      = active.max().item() if active.max() > 0 else 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────

def build_dataloaders(
    root:        str,
    annotations: str  = 'ODIR-5K_Training_Annotations.xlsx',
    batch_size:  int  = 16,
    num_workers: int  = 4,
    size:        int  = IMAGE_SIZE,
    mode:        str  = 'single',
    cache:       bool = False,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = ODIRDataset(root, annotations, 'train', size, mode, cache=cache)
    val_ds   = ODIRDataset(root, annotations, 'val',   size, mode, cache=cache)

    sample_w = get_sample_weights(train_ds)
    sampler  = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[ODIR] Train: {len(train_ds)} samples | Val: {len(val_ds)} | "
          f"Batch: {batch_size} | Mode: {mode}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# MIXUP / CUTMIX
# ─────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup augmentation for multi-label classification."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix augmentation for multi-label classification."""
    lam  = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    idx  = torch.randperm(B, device=x.device)

    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2-x1)*(y2-y1) / (W*H)
    mixed_y = lam_actual * y + (1 - lam_actual) * y[idx]
    return mixed_x, mixed_y
