"""
IDRiD Dataset Loader, Preprocessing and Augmentation
=====================================================
Handles all three IDRiD sub-tasks:
  A. Disease Grading  (516 images, DR grade 0-4, DME grade 0-2)
  B. Segmentation     (81 images, pixel-level lesion masks)
  C. Localisation     (516 images, OD + fovea coordinates)

Folder layout expected:
  data/idrid/
  ├── Disease Grading/
  │   ├── Original Images/
  │   │   ├── a. Training Set/   (413 .jpg)
  │   │   └── b. Testing Set/    (103 .jpg)
  │   └── Groundtruths/
  │       ├── a. IDRiD_Disease Grading_Training Labels.csv
  │       └── b. IDRiD_Disease Grading_Testing Labels.csv
  ├── A. Segmentation/
  │   ├── Original Images/
  │   │   ├── Training Set/      (54 .jpg)
  │   │   └── Testing Set/       (27 .jpg)
  │   └── Ground Truths/
  │       ├── Training Set/
  │       │   ├── Microaneurysms/    (*_MA.tif)
  │       │   ├── Haemorrhages/      (*_HE.tif)
  │       │   ├── Hard Exudates/     (*_EX.tif)
  │       │   ├── Soft Exudates/     (*_SE.tif)
  │       │   └── Optic Disc/        (*_OD.tif)
  │       └── Testing Set/   (same structure)
  └── C. Localization/
      └── Groundtruths/
          ├── Training Set/  (OD + Fovea CSVs)
          └── Testing Set/
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

IMAGE_SIZE    = 512
MEAN          = [0.4154, 0.2168, 0.0703]   # IDRiD-specific channel stats
STD           = [0.2979, 0.1667, 0.0878]

LESION_NAMES  = ['MA', 'HE', 'EX', 'SE', 'NV']
LESION_COLS   = ['Microaneurysms', 'Haemorrhages',
                 'Hard Exudates', 'Soft Exudates', 'Neovascularization']

# Original capture resolution (for normalising coordinates)
ORIG_W, ORIG_H = 4288, 2848

DR_GRADE_WEIGHTS = [1.0, 2.5, 2.0, 4.0, 6.0]   # inverse freq weights


# ─────────────────────────────────────────────
# FUNDUS-SPECIFIC PREPROCESSING
# ─────────────────────────────────────────────

def green_channel_clahe(img: Image.Image) -> Image.Image:
    """
    Apply CLAHE (approximated via PIL) to the green channel —
    the highest-contrast channel for DR lesion detection.
    Green channel enhancement is standard in retinal image analysis.
    """
    r, g, b = img.split()
    g_enhanced = ImageEnhance.Contrast(g).enhance(2.0)
    g_enhanced = g_enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=120))
    return Image.merge('RGB', (r, g_enhanced, b))


def remove_black_border(img: Image.Image,
                        threshold: int = 10) -> Tuple[Image.Image, Tuple]:
    """
    Fundus images have large black borders from the camera aperture.
    Crop to the bounding box of non-black pixels.
    Returns cropped image and (x0, y0, x1, y1) crop box.
    """
    arr  = np.array(img.convert('L'))
    mask = arr > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img, (0, 0, img.width, img.height)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    box = (int(c0), int(r0), int(c1 + 1), int(r1 + 1))
    return img.crop(box), box


def square_pad(img: Image.Image) -> Image.Image:
    """Pad image to square with black pixels (preserves aspect ratio)."""
    w, h = img.size
    s    = max(w, h)
    pad  = Image.new('RGB', (s, s), (0, 0, 0))
    pad.paste(img, ((s - w) // 2, (s - h) // 2))
    return pad


def preprocess_fundus(img: Image.Image,
                      size: int = IMAGE_SIZE,
                      enhance: bool = True) -> Image.Image:
    """Full preprocessing pipeline for a single fundus image."""
    img, _ = remove_black_border(img)
    img     = square_pad(img)
    if enhance:
        img = green_channel_clahe(img)
    img = img.resize((size, size), Image.LANCZOS)
    return img


def normalise_coord(x_px: float, y_px: float,
                    orig_w: int = ORIG_W,
                    orig_h: int = ORIG_H) -> Tuple[float, float]:
    """Convert pixel coordinates to [0, 1] normalised space."""
    return x_px / orig_w, y_px / orig_h


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────

class FundusAugmentation:
    """
    Retina-specific augmentation policy.
    Applies consistent geometric transforms to both image and masks.
    """

    def __init__(self, mode: str = 'train'):
        assert mode in ('train', 'val', 'test')
        self.mode = mode

        self.normalise = T.Normalize(mean=MEAN, std=STD)

        # Colour jitter (conservative — preserve diagnostic colour)
        self.colour_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        # Random Gaussian blur (simulate camera focus variation)
        self.blur = T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2)

    def __call__(
        self,
        image: Image.Image,
        masks: Optional[List[Image.Image]] = None,
        od_xy: Optional[Tuple] = None,
        fovea_xy: Optional[Tuple] = None,
    ):
        """
        Returns:
          image_tensor : (3, H, W) float32
          masks_tensor : (K, H, W) float32 or None
          od_xy        : (2,) normalised, updated for flips
          fovea_xy     : (2,) normalised, updated for flips
        """
        flip_h = flip_v = False
        angle  = 0.0
        scale  = 1.0
        crop_params = None

        if self.mode == 'train':
            # Random horizontal flip (left/right eye)
            if torch.rand(1) < 0.5:
                flip_h = True
                image  = TF.hflip(image)

            # Random vertical flip
            if torch.rand(1) < 0.3:
                flip_v = True
                image  = TF.vflip(image)

            # Random rotation (±15°)
            angle = float(torch.FloatTensor(1).uniform_(-15, 15))
            image = TF.rotate(image, angle, fill=0)

            # Random scale + crop (zoom in 0.9–1.1×)
            scale = float(torch.FloatTensor(1).uniform_(0.9, 1.1))
            new_s = int(IMAGE_SIZE * scale)
            image = image.resize((new_s, new_s), Image.LANCZOS)
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(IMAGE_SIZE, IMAGE_SIZE))
            image = TF.crop(image, i, j, h, w)
            crop_params = (i, j, new_s)

            # Colour augmentation (only image, not masks)
            image = self.colour_jitter(image)
            image = self.blur(image)

            # Apply same geometric transforms to masks
            if masks is not None:
                new_masks = []
                for m in masks:
                    if flip_h: m = TF.hflip(m)
                    if flip_v: m = TF.vflip(m)
                    m = TF.rotate(m, angle, fill=0)
                    m = m.resize((new_s, new_s), Image.NEAREST)
                    m = TF.crop(m, i, j, h, w)
                    new_masks.append(m)
                masks = new_masks

            # Update landmark coordinates for flips
            if od_xy is not None:
                ox, oy = od_xy
                fx, fy = fovea_xy
                if flip_h:
                    ox, fx = 1.0 - ox, 1.0 - fx
                if flip_v:
                    oy, fy = 1.0 - oy, 1.0 - fy
                od_xy, fovea_xy = (ox, oy), (fx, fy)

        # To tensor + normalise
        image_tensor = TF.to_tensor(image)               # (3, H, W) float [0,1]
        image_tensor = self.normalise(image_tensor)

        masks_tensor = None
        if masks is not None:
            masks_tensor = torch.stack([
                torch.from_numpy(np.array(m, dtype=np.float32) / 255.0)
                for m in masks
            ])   # (K, H, W)

        return image_tensor, masks_tensor, od_xy, fovea_xy


# ─────────────────────────────────────────────
# GRADING DATASET
# ─────────────────────────────────────────────

class IDRiDGradingDataset(Dataset):
    """
    IDRiD Disease Grading subset.
    Returns per-image: image tensor, DR grade, DME grade,
                       OD coords, fovea coords, image ID.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = IMAGE_SIZE,
        augment: bool = True,
    ):
        assert split in ('train', 'test')
        self.root      = Path(root)
        self.split     = split
        self.aug       = FundusAugmentation('train' if augment else 'val')

        split_folder   = 'a. Training Set' if split == 'train' else 'b. Testing Set'
        self.image_dir = self.root / 'Disease Grading' / 'Original Images' / split_folder

        # Load grading labels
        label_file = (
            'a. IDRiD_Disease Grading_Training Labels.csv'
            if split == 'train' else
            'b. IDRiD_Disease Grading_Testing Labels.csv'
        )
        label_path = (self.root / 'Disease Grading' / 'Groundtruths' / label_file)
        self.labels = pd.read_csv(label_path)
        self.labels.columns = [c.strip() for c in self.labels.columns]

        # Load localisation coordinates
        loc_split  = 'Training Set' if split == 'train' else 'Testing Set'
        loc_dir    = self.root / 'C. Localization' / 'Groundtruths' / loc_split

        self.od_coords    = self._load_loc_csv(loc_dir / 'OD_Location.csv')
        self.fovea_coords = self._load_loc_csv(loc_dir / 'Fovea_Location.csv')

        self.image_ids = self.labels['Image name'].tolist()

    @staticmethod
    def _load_loc_csv(path: Path) -> Dict[str, Tuple[float, float]]:
        """Returns {image_id: (x_norm, y_norm)} dict."""
        coords = {}
        if not path.exists():
            return coords
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            img_id = str(row.iloc[0]).strip()
            x_px   = float(row.iloc[1])
            y_px   = float(row.iloc[2])
            coords[img_id] = normalise_coord(x_px, y_px)
        return coords

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        row    = self.labels[self.labels['Image name'] == img_id].iloc[0]

        # Load and preprocess image
        img_path = self.image_dir / f"{img_id}.jpg"
        image    = Image.open(img_path).convert('RGB')
        image    = preprocess_fundus(image)

        # Labels
        dr_grade  = int(row['Retinopathy grade'])
        dme_grade = int(row['Risk of macular edema '])

        # Localisation coordinates (use defaults if missing)
        od_xy    = self.od_coords.get(img_id,    (0.62, 0.50))
        fovea_xy = self.fovea_coords.get(img_id, (0.50, 0.50))

        # Augment
        image_t, _, od_xy, fovea_xy = self.aug(image, None, od_xy, fovea_xy)

        return {
            'image'    : image_t,
            'dr_grade' : torch.tensor(dr_grade,  dtype=torch.long),
            'dme_grade': torch.tensor(dme_grade, dtype=torch.long),
            'od_xy'    : torch.tensor(od_xy,     dtype=torch.float32),
            'fovea_xy' : torch.tensor(fovea_xy,  dtype=torch.float32),
            'image_id' : img_id,
        }


# ─────────────────────────────────────────────
# SEGMENTATION DATASET
# ─────────────────────────────────────────────

class IDRiDSegmentationDataset(Dataset):
    """
    IDRiD Segmentation subset (81 images).
    Returns per-image: image tensor, 5 binary lesion masks (MA/HE/EX/SE/NV),
                       OD mask, OD coords, fovea coords.

    Used for anatomical pre-training of the AnatomicalConstraintModule.
    """

    LESION_DIRS = {
        'MA': '1. Microaneurysms',
        'HE': '2. Haemorrhages',
        'EX': '3. Hard Exudates',
        'SE': '4. Soft Exudates',
        'OD': '5. Optic Disc',
    }
    LESION_SUFFIXES = {
        'MA': '_MA', 'HE': '_HE', 'EX': '_EX', 'SE': '_SE', 'OD': '_OD'
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = IMAGE_SIZE,
        augment: bool = True,
    ):
        assert split in ('train', 'test')
        self.root = Path(root)
        self.size = image_size
        self.aug  = FundusAugmentation('train' if augment else 'val')

        split_dir = 'Training Set' if split == 'train' else 'Testing Set'
        self.image_dir = self.root / 'A. Segmentation' / 'Original Images' / split_dir
        self.mask_base = self.root / 'A. Segmentation' / 'Ground Truths' / split_dir

        self.image_ids = sorted([
            p.stem for p in self.image_dir.glob('*.jpg')
        ])

    def __len__(self):
        return len(self.image_ids)

    def _load_mask(self, img_id: str, lesion_key: str) -> Image.Image:
        """Load a .tif mask or return a black image if missing."""
        lesion_dir = self.LESION_DIRS[lesion_key]
        suffix     = self.LESION_SUFFIXES[lesion_key]
        path       = self.mask_base / lesion_dir / f"{img_id}{suffix}.tif"
        if path.exists():
            mask = Image.open(path).convert('L')
        else:
            mask = Image.new('L', (ORIG_W, ORIG_H), 0)
        mask = square_pad_mask(mask)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return mask

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        image  = Image.open(self.image_dir / f"{img_id}.jpg").convert('RGB')
        image  = preprocess_fundus(image, self.size)

        masks  = [self._load_mask(img_id, k) for k in ['MA','HE','EX','SE','OD']]

        # Default coords (no localisation file for segmentation split)
        od_xy    = (0.62, 0.50)
        fovea_xy = (0.50, 0.50)

        image_t, masks_t, od_xy, fovea_xy = self.aug(image, masks, od_xy, fovea_xy)

        return {
            'image'      : image_t,
            'masks'      : masks_t,          # (5, H, W): MA,HE,EX,SE,OD
            'od_xy'      : torch.tensor(od_xy,    dtype=torch.float32),
            'fovea_xy'   : torch.tensor(fovea_xy, dtype=torch.float32),
            'image_id'   : img_id,
        }


def square_pad_mask(mask: Image.Image) -> Image.Image:
    """Black-pad a mask image to square."""
    w, h = mask.size
    s    = max(w, h)
    pad  = Image.new('L', (s, s), 0)
    pad.paste(mask, ((s - w) // 2, (s - h) // 2))
    return pad


# ─────────────────────────────────────────────
# COMBINED DATASET (grading + lesion labels)
# ─────────────────────────────────────────────

class IDRiDCombinedDataset(Dataset):
    """
    Combines grading labels with lesion presence labels
    (derived from segmentation masks — a pixel is present if
    any mask pixel > 0).

    Used for the full multi-task training objective.
    """

    def __init__(
        self,
        root: str,
        split: str  = 'train',
        image_size: int = IMAGE_SIZE,
        augment: bool   = True,
    ):
        self.grading = IDRiDGradingDataset(root, split, image_size, augment)
        self.root    = Path(root)
        self.split   = split

        # Build lesion presence map from segmentation subset (81 images)
        self.lesion_presence = self._build_lesion_presence()

    def _build_lesion_presence(self) -> Dict[str, np.ndarray]:
        """
        For each image in the grading set, check if a corresponding
        segmentation mask exists; derive binary lesion presence.
        If no mask exists, use DR grade heuristics as a fallback.
        """
        presence = {}
        seg_split = 'Training Set' if self.split == 'train' else 'Testing Set'
        mask_base = self.root / 'A. Segmentation' / 'Ground Truths' / seg_split

        lesion_dirs = {
            'MA': '1. Microaneurysms',
            'HE': '2. Haemorrhages',
            'EX': '3. Hard Exudates',
            'SE': '4. Soft Exudates',
        }
        suffixes = {'MA': '_MA', 'HE': '_HE', 'EX': '_EX', 'SE': '_SE'}

        for img_id in self.grading.image_ids:
            vec = np.zeros(NUM_LESIONS := 5, dtype=np.float32)
            for i, (key, d) in enumerate(lesion_dirs.items()):
                mask_path = mask_base / d / f"{img_id}{suffixes[key]}.tif"
                if mask_path.exists():
                    m = np.array(Image.open(mask_path).convert('L'))
                    vec[i] = float(m.max() > 0)
                else:
                    # Fallback: grade-based heuristic
                    row = self.grading.labels[
                        self.grading.labels['Image name'] == img_id
                    ].iloc[0]
                    g = int(row['Retinopathy grade'])
                    vec[i] = float(g >= 1)  # any lesion at grade ≥ 1
            # NV (neovascularisation) from grade ≥ 3
            row = self.grading.labels[
                self.grading.labels['Image name'] == img_id
            ].iloc[0]
            vec[4] = float(int(row['Retinopathy grade']) >= 3)
            presence[img_id] = vec

        return presence

    def __len__(self):
        return len(self.grading)

    def __getitem__(self, idx: int):
        sample   = self.grading[idx]
        img_id   = sample['image_id']
        lesion_v = self.lesion_presence.get(img_id, np.zeros(5, dtype=np.float32))

        sample['lesion_labels'] = torch.tensor(lesion_v, dtype=torch.float32)
        return sample


# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────

def get_class_weights(dataset: IDRiDCombinedDataset) -> torch.Tensor:
    """Compute per-sample weights for WeightedRandomSampler (DR grade balance)."""
    grades  = [int(dataset.grading.labels[
                   dataset.grading.labels['Image name'] == img_id
               ]['Retinopathy grade'].values[0])
               for img_id in dataset.grading.image_ids]
    weights = [DR_GRADE_WEIGHTS[g] for g in grades]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    root: str,
    batch_size: int  = 8,
    num_workers: int = 4,
    image_size: int  = IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    Training loader uses WeightedRandomSampler to address class imbalance.
    """
    train_ds = IDRiDCombinedDataset(root, 'train', image_size, augment=True)
    val_ds   = IDRiDCombinedDataset(root, 'test',  image_size, augment=False)

    # Weighted sampler for DR grade imbalance
    sample_weights = get_class_weights(train_ds)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Dataset] Train: {len(train_ds)} samples | "
          f"Val: {len(val_ds)} samples | "
          f"Batch size: {batch_size}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# DATASET STATISTICS (run once)
# ─────────────────────────────────────────────

def compute_channel_stats(root: str, n_samples: int = 100) -> None:
    """
    Compute per-channel mean and std for normalisation.
    Run once on the training set and update MEAN/STD constants.
    """
    ds   = IDRiDGradingDataset(root, 'train', augment=False)
    aug  = FundusAugmentation('val')
    means, stds = [], []

    indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    for i in indices:
        sample = ds[i]
        img    = sample['image']     # already normalised tensor — undo for stats
        means.append(img.mean(dim=[1, 2]).numpy())
        stds.append(img.std(dim=[1, 2]).numpy())

    print(f"Channel mean: {np.mean(means, axis=0).tolist()}")
    print(f"Channel std : {np.mean(stds,  axis=0).tolist()}")


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/idrid'

    ds = IDRiDCombinedDataset(root, split='train', augment=True)
    sample = ds[0]
    print("Image shape    :", sample['image'].shape)
    print("DR grade       :", sample['dr_grade'].item())
    print("DME grade      :", sample['dme_grade'].item())
    print("Lesion labels  :", sample['lesion_labels'])
    print("OD xy          :", sample['od_xy'])
    print("Fovea xy       :", sample['fovea_xy'])
    print("Dataset sanity check PASSED")
