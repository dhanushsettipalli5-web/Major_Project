"""
Inference Script — AC-ViT-POG
==============================
Single-image inference with:
  - DR grade prediction (0–4) + confidence
  - DME grade prediction (0–2) + confidence
  - Per-lesion presence probability
  - GradCAM++ attention heatmap over the fundus image
  - POG adjacency matrix export (for UI visualisation)

Usage (CLI):
    python src/inference.py \
        --image path/to/fundus.jpg \
        --checkpoint checkpoints/best.pth \
        --od_x 2660 --od_y 1440 \
        --fovea_x 2144 --fovea_y 1424 \
        --output results/

Usage (Python API):
    from inference import Predictor
    pred = Predictor('checkpoints/best.pth')
    result = pred.predict('fundus.jpg', od_xy=(0.62, 0.50), fovea_xy=(0.50, 0.50))
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from dataset import preprocess_fundus, normalise_coord, MEAN, STD, ORIG_W, ORIG_H
from model   import build_acvit_pog, IMAGE_SIZE


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DR_GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'PDR (Proliferative)',
}

DME_GRADE_LABELS = {
    0: 'No DME',
    1: 'Mild DME',
    2: 'Severe DME',
}

LESION_NAMES = [
    'Microaneurysms',
    'Haemorrhages',
    'Hard Exudates',
    'Soft Exudates',
    'Neovascularisation',
]

REFERRAL_RULES = {
    # (dr_grade, dme_grade) → referral urgency
    (0, 0): ('No referral needed', 'green'),
    (1, 0): ('Annual review', 'green'),
    (1, 1): ('6-month review', 'yellow'),
    (2, 0): ('6-month review', 'yellow'),
    (2, 1): ('3-month review', 'orange'),
    (2, 2): ('Urgent referral', 'red'),
    (3, 0): ('Urgent referral', 'red'),
    (3, 1): ('Urgent referral', 'red'),
    (3, 2): ('Same-day referral', 'red'),
    (4, 0): ('Urgent referral', 'red'),
    (4, 1): ('Same-day referral', 'red'),
    (4, 2): ('Same-day referral', 'red'),
}


# ─────────────────────────────────────────────
# GRADCAM++
# ─────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    GradCAM++ adapted for ViT:
    Hooks the last transformer block's attention output to compute
    a spatial saliency map highlighting lesion-relevant patches.

    Returns a (H, W) heatmap resized to match the input image.
    """

    def __init__(self, model, target_layer_idx: int = -1):
        self.model        = model
        self.target_block = model.transformer[target_layer_idx]
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output is (tokens, attn_weights); tokens: (B, N+1, D)
            self.activations = output[0].detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_block.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_block.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        od_xy:        torch.Tensor,
        fovea_xy:     torch.Tensor,
        target:       str = 'dr',   # 'dr', 'dme', or lesion index 0-4
        target_class: int = None,
    ) -> np.ndarray:
        """
        Returns: (image_size, image_size) float32 heatmap in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        # Forward
        outputs = self.model(image_tensor, od_xy.unsqueeze(0), fovea_xy.unsqueeze(0))

        # Select score to backprop
        if target == 'dr':
            logits = outputs['dr_logits']
            cls    = target_class if target_class is not None else logits.argmax(1).item()
            score  = logits[0, cls]
        elif target == 'dme':
            logits = outputs['dme_logits']
            cls    = target_class if target_class is not None else logits.argmax(1).item()
            score  = logits[0, cls]
        else:   # lesion index
            score = outputs['lesion_logits'][0, int(target)]

        self.model.zero_grad()
        score.backward()

        # GradCAM++ weights
        grads = self.gradients[:, 1:, :]     # remove CLS, (B, N, D)
        acts  = self.activations[:, 1:, :]   # (B, N, D)

        # Alpha weights: second-order gradient approximation
        grads_sq   = grads ** 2
        grads_cb   = grads ** 3
        denom      = 2 * grads_sq + acts * grads_cb
        denom      = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha      = grads_sq / denom
        weights    = (alpha * F.relu(grads)).sum(dim=-1)  # (B, N)

        # Spatial map
        n = int(IMAGE_SIZE // 16)  # 32
        cam = weights[0].reshape(n, n).cpu().numpy()
        cam = np.maximum(cam, 0)

        # Resize to image size
        cam_img = Image.fromarray(cam).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        cam_np  = np.array(cam_img, dtype=np.float32)

        # Normalise to [0, 1]
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        return cam_np


# ─────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────

class Predictor:
    """
    High-level inference API.

    Example:
        pred   = Predictor('checkpoints/best.pth')
        result = pred.predict(
            image_path = 'IDRiD_001.jpg',
            od_xy      = (0.62, 0.50),
            fovea_xy   = (0.50, 0.50),
        )
        print(result['dr_grade_label'])   # 'Moderate NPDR'
        print(result['lesion_probs'])     # {'Microaneurysms': 0.91, ...}
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = build_acvit_pog().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.gradcam = GradCAMPlusPlus(self.model, target_layer_idx=-1)
        print(f"[Predictor] Loaded from {checkpoint_path} on {self.device}")

    def _load_and_preprocess(
        self, image_input
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Accepts file path (str/Path) or PIL Image.
        Returns (tensor (3,H,W), preprocessed PIL image).
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError("image_input must be a file path or PIL.Image")

        img_processed = preprocess_fundus(img, IMAGE_SIZE, enhance=True)

        # To tensor + normalise
        tensor = TF.to_tensor(img_processed)
        for c, (m, s) in enumerate(zip(MEAN, STD)):
            tensor[c] = (tensor[c] - m) / s

        return tensor, img_processed

    @torch.no_grad()
    def predict(
        self,
        image_input,
        od_xy:    Tuple[float, float] = (0.62, 0.50),
        fovea_xy: Tuple[float, float] = (0.50, 0.50),
        generate_heatmap: bool = True,
    ) -> Dict:
        """
        Run full inference on a single fundus image.

        Args:
            image_input      : file path or PIL.Image
            od_xy            : optic disc centre, normalised [0,1]
            fovea_xy         : fovea centre, normalised [0,1]
            generate_heatmap : if True, compute GradCAM++ heatmap

        Returns:
            dict with keys:
              dr_grade, dr_grade_label, dr_probs,
              dme_grade, dme_grade_label, dme_probs,
              lesion_probs,
              referral, referral_colour,
              heatmap (optional, np.ndarray),
              pog_adj (np.ndarray 5×5),
        """
        tensor, pil_img = self._load_and_preprocess(image_input)
        t = tensor.to(self.device)

        od_t    = torch.tensor(od_xy,    dtype=torch.float32, device=self.device)
        fovea_t = torch.tensor(fovea_xy, dtype=torch.float32, device=self.device)

        outputs = self.model(t.unsqueeze(0), od_t.unsqueeze(0), fovea_t.unsqueeze(0))

        # DR grade
        dr_probs    = F.softmax(outputs['dr_logits'][0], dim=0).cpu().numpy()
        dr_grade    = int(dr_probs.argmax())

        # DME grade
        dme_probs   = F.softmax(outputs['dme_logits'][0], dim=0).cpu().numpy()
        dme_grade   = int(dme_probs.argmax())

        # Lesion probabilities
        les_probs   = torch.sigmoid(outputs['lesion_logits'][0]).cpu().numpy()

        # Referral
        ref_key     = (min(dr_grade, 4), min(dme_grade, 2))
        referral, ref_colour = REFERRAL_RULES.get(ref_key, ('Urgent referral', 'red'))

        # POG adjacency
        pog_adj = outputs['adj_matrix'].cpu().numpy()

        result = {
            'dr_grade'       : dr_grade,
            'dr_grade_label' : DR_GRADE_LABELS[dr_grade],
            'dr_probs'       : {f'Grade {i}': round(float(p), 4) for i, p in enumerate(dr_probs)},
            'dr_confidence'  : round(float(dr_probs.max()), 4),
            'dme_grade'      : dme_grade,
            'dme_grade_label': DME_GRADE_LABELS[dme_grade],
            'dme_probs'      : {f'Grade {i}': round(float(p), 4) for i, p in enumerate(dme_probs)},
            'dme_confidence' : round(float(dme_probs.max()), 4),
            'lesion_probs'   : {
                name: round(float(p), 4)
                for name, p in zip(LESION_NAMES, les_probs)
            },
            'referral'       : referral,
            'referral_colour': ref_colour,
            'pog_adj'        : pog_adj.round(3).tolist(),
            'pog_adj_labels' : ['MA','HE','EX','SE','NV'],
            'heatmap'        : None,
            'preprocessed_image': pil_img,
        }

        # GradCAM++ heatmap (requires grad)
        if generate_heatmap:
            t_grad = tensor.to(self.device)
            heatmap = self.gradcam.generate(t_grad, od_t, fovea_t, target='dr')
            result['heatmap'] = heatmap

        return result

    def predict_from_array(
        self,
        image_array: np.ndarray,
        od_xy:    Tuple[float, float] = (0.62, 0.50),
        fovea_xy: Tuple[float, float] = (0.50, 0.50),
    ) -> Dict:
        """Convenience wrapper: accepts (H, W, 3) uint8 numpy array."""
        pil_img = Image.fromarray(image_array.astype(np.uint8))
        return self.predict(pil_img, od_xy, fovea_xy)

    def predict_batch(
        self,
        image_paths: list,
        od_xys:      Optional[list] = None,
        fovea_xys:   Optional[list] = None,
    ) -> list:
        """Run inference on a list of image paths."""
        results = []
        for i, path in enumerate(image_paths):
            od    = od_xys[i]    if od_xys    else (0.62, 0.50)
            fovea = fovea_xys[i] if fovea_xys else (0.50, 0.50)
            results.append(self.predict(path, od, fovea, generate_heatmap=False))
        return results

    def heatmap_overlay(
        self,
        pil_img:  Image.Image,
        heatmap:  np.ndarray,
        alpha:    float = 0.45,
    ) -> Image.Image:
        """
        Overlay a GradCAM heatmap onto the fundus image using a
        jet-like colourmap. Returns a PIL RGB image.
        """
        import colorsys

        h, w = heatmap.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                val = float(heatmap[y, x])
                # Jet colormap approximation
                r = min(1.0, max(0.0, 1.5 - abs(4 * val - 3)))
                g = min(1.0, max(0.0, 1.5 - abs(4 * val - 2)))
                b = min(1.0, max(0.0, 1.5 - abs(4 * val - 1)))
                overlay[y, x] = [int(r*255), int(g*255), int(b*255)]

        # Vectorised version (much faster)
        val = heatmap
        r = np.clip(1.5 - np.abs(4 * val - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * val - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * val - 1), 0, 1)
        overlay = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        base   = np.array(pil_img.resize((w, h)))
        blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
        return Image.fromarray(blended)


# ─────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Run inference on a single fundus image')
    p.add_argument('--image',      type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--od_x',       type=float, default=None,
                   help='Optic disc x in pixels (original resolution)')
    p.add_argument('--od_y',       type=float, default=None)
    p.add_argument('--fovea_x',    type=float, default=None)
    p.add_argument('--fovea_y',    type=float, default=None)
    p.add_argument('--output',     type=str, default='results/')
    p.add_argument('--no_heatmap', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Normalise coordinates
    if args.od_x is not None:
        od_xy    = normalise_coord(args.od_x,    args.od_y)
        fovea_xy = normalise_coord(args.fovea_x, args.fovea_y)
    else:
        od_xy    = (0.62, 0.50)
        fovea_xy = (0.50, 0.50)
        print("[Inference] No OD/fovea coords supplied — using defaults")

    pred   = Predictor(args.checkpoint)
    result = pred.predict(
        args.image,
        od_xy    = od_xy,
        fovea_xy = fovea_xy,
        generate_heatmap = not args.no_heatmap,
    )

    # Save heatmap overlay
    if result['heatmap'] is not None:
        overlay = pred.heatmap_overlay(result['preprocessed_image'], result['heatmap'])
        overlay_path = Path(args.output) / 'heatmap_overlay.png'
        overlay.save(overlay_path)
        print(f"Heatmap saved to {overlay_path}")

    # Save JSON results (exclude non-serialisable fields)
    out = {k: v for k, v in result.items()
           if k not in ('heatmap', 'preprocessed_image')}
    json_path = Path(args.output) / 'prediction.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)

    # Print summary
    print("\n═══ PREDICTION SUMMARY ═══")
    print(f"  DR Grade  : {result['dr_grade']} — {result['dr_grade_label']} "
          f"(confidence {result['dr_confidence']:.1%})")
    print(f"  DME Grade : {result['dme_grade']} — {result['dme_grade_label']} "
          f"(confidence {result['dme_confidence']:.1%})")
    print(f"  Referral  : {result['referral']}")
    print("\n  Lesion probabilities:")
    for name, prob in result['lesion_probs'].items():
        bar = '█' * int(prob * 20)
        print(f"    {name:<22} {prob:.3f}  {bar}")
    print(f"\nFull results saved to {json_path}")


if __name__ == '__main__':
    main()
