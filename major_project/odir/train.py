"""
Maximum-Accuracy Training Script — ODIR-5K
===========================================
Every technique here is chosen specifically to push past the SOTA
(AUC 97.27%, F1 94.28%) on ODIR-5K.

Training strategy:
  Stage 1 (epochs 1–5):    Freeze backbone, train head only (warm-up)
  Stage 2 (epochs 6–20):   Unfreeze top-N layers, lower LR on backbone
  Stage 3 (epochs 21–end): Full fine-tune with cosine decay + SWA

Techniques for maximum accuracy:
  ✓ Asymmetric Loss (ASL) — handles severe class imbalance
  ✓ Label smoothing (0.05) — prevents overconfident predictions
  ✓ Mixup + CutMix (50/50 random) — strongest augmentation pair
  ✓ Layer-wise LR decay (backbone 10× lower than head)
  ✓ Stochastic Weight Averaging (SWA) — last 20% of training
  ✓ EMA (Exponential Moving Average) of weights
  ✓ Gradient clipping + gradient accumulation (effective batch 64)
  ✓ Cosine annealing with warm restarts
  ✓ Progressive resizing (384→448→512)
  ✓ Mixed precision (FP16)
  ✓ Early stopping on val AUC (patience 15)
  ✓ Best checkpoint saved per metric (AUC, F1, kappa)

Usage:
  # Stage: train EfficientNet alone first
  python odir/train.py --data data/odir --backbone efficientnet --epochs 30

  # Stage: train ViT alone
  python odir/train.py --data data/odir --backbone vit --epochs 30 --size 384

  # Stage: train full ensemble (requires GPU with ≥24GB VRAM)
  python odir/train.py --data data/odir --backbone ensemble --epochs 40 --batch_size 8
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, average_precision_score

from dataset import build_dataloaders, mixup_data, cutmix_data, NUM_CLASSES, CLASSES, CLASS_WEIGHTS
from model   import build_ensemble, build_single


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Train ODIR-5K for maximum accuracy')
    p.add_argument('--data',        type=str,   default='data/odir')
    p.add_argument('--annotations', type=str,   default='ODIR-5K_Training_Annotations.xlsx')
    p.add_argument('--backbone',    type=str,   default='efficientnet',
                   choices=['efficientnet', 'vit', 'ensemble'])
    p.add_argument('--fusion',      type=str,   default='concat',
                   choices=['concat', 'mean', 'learned'])
    p.add_argument('--epochs',      type=int,   default=40)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--accum_steps', type=int,   default=4,
                   help='Gradient accumulation steps (effective batch = batch*accum)')
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--backbone_lr', type=float, default=3e-5)
    p.add_argument('--weight_decay',type=float, default=0.01)
    p.add_argument('--size',        type=int,   default=512)
    p.add_argument('--warmup_epochs',type=int,  default=5)
    p.add_argument('--swa_start',   type=float, default=0.7,
                   help='Fraction of training to start SWA')
    p.add_argument('--mixup_alpha', type=float, default=0.4)
    p.add_argument('--cutmix_alpha',type=float, default=1.0)
    p.add_argument('--label_smooth',type=float, default=0.05)
    p.add_argument('--drop_rate',   type=float, default=0.35)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/odir')
    p.add_argument('--pretrained',  type=str,   default=None)
    p.add_argument('--resume',      type=str,   default=None)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--no_swa',      action='store_true')
    p.add_argument('--no_ema',      action='store_true')
    return p.parse_args()


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────

class AsymmetricLossOptimised(nn.Module):
    """
    ASL (Ben-Baruch et al., ICCV 2021) — best loss for multi-label
    classification with class imbalance.
    gamma_neg=4, gamma_pos=1 recommended for medical imaging.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05,
                 eps=1e-8, disable_torch_grad=True):
        super().__init__()
        self.gamma_neg           = gamma_neg
        self.gamma_pos           = gamma_pos
        self.clip                = clip
        self.eps                 = eps
        self.disable_torch_grad  = disable_torch_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(x)
        xs_pos    = x_sigmoid
        xs_neg    = 1 - x_sigmoid

        # Asymmetric clip
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        lo_pos = y       * torch.log(xs_pos.clamp(min=self.eps))
        lo_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss   = lo_pos + lo_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad:
                torch.set_grad_enabled(False)
            pt0   = xs_pos * (1 - y)
            pt1   = xs_neg * y
            pt    = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w     = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad:
                torch.set_grad_enabled(True)
            loss = loss * one_sided_w

        return -loss.sum() / x.size(0)


class CombinedLoss(nn.Module):
    """
    ASL + weighted BCE with label smoothing + auxiliary head losses.
    """
    def __init__(self, label_smooth: float = 0.05, aux_weight: float = 0.3):
        super().__init__()
        self.asl         = AsymmetricLossOptimised(gamma_neg=4, gamma_pos=1, clip=0.05)
        self.label_smooth= label_smooth
        self.aux_weight  = aux_weight
        cw = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
        self.register_buffer('class_weights', cw / cw.mean())

    def _smooth(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to soft targets."""
        return y * (1 - self.label_smooth) + self.label_smooth / NUM_CLASSES

    def forward(self, outputs: Dict, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        smoothed = self._smooth(targets)

        # Main ASL loss
        l_asl = self.asl(outputs['logits'], smoothed)

        # Weighted BCE (complements ASL on hard positives)
        l_wbce = F.binary_cross_entropy_with_logits(
            outputs['logits'], smoothed,
            weight=self.class_weights.unsqueeze(0).expand_as(smoothed),
            reduction='mean',
        )

        total = 0.6 * l_asl + 0.4 * l_wbce

        # Auxiliary losses
        l_aux = torch.tensor(0.0, device=targets.device)
        for key in ['eff_aux', 'vit_aux', 'aux_logits']:
            if key in outputs:
                l_aux = l_aux + F.binary_cross_entropy_with_logits(
                    outputs[key], smoothed, reduction='mean'
                )
        total = total + self.aux_weight * l_aux

        breakdown = {
            'loss_asl' : l_asl.item(),
            'loss_wbce': l_wbce.item(),
            'loss_aux' : l_aux.item() if isinstance(l_aux, torch.Tensor) else 0.0,
            'loss_total': total.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class MetricAccumulator:
    def __init__(self):
        self.probs   = []
        self.targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        p = np.concatenate(self.probs,   axis=0)  # (N, K)
        t = np.concatenate(self.targets, axis=0)  # (N, K)
        b = (p >= 0.5).astype(int)

        # Per-class AUC
        aucs = []
        for i in range(NUM_CLASSES):
            if t[:, i].sum() > 0 and (1 - t[:, i]).sum() > 0:
                aucs.append(roc_auc_score(t[:, i], p[:, i]))
        mean_auc = float(np.mean(aucs)) if aucs else 0.0

        # F1 (macro)
        f1 = float(f1_score(t, b, average='macro', zero_division=0))

        # mAP
        aps = []
        for i in range(NUM_CLASSES):
            if t[:, i].sum() > 0:
                aps.append(average_precision_score(t[:, i], p[:, i]))
        mean_ap = float(np.mean(aps)) if aps else 0.0

        # Kappa (convert multi-label to dominant class for kappa)
        pred_cls = b.argmax(axis=1)
        true_cls = t.argmax(axis=1)
        kappa = float(cohen_kappa_score(true_cls, pred_cls, weights='quadratic'))

        # Exact match accuracy
        exact_match = float((b == t).all(axis=1).mean())

        per_class = {}
        for i, cls in enumerate(CLASSES):
            if t[:, i].sum() > 0:
                per_class[cls] = {
                    'auc': round(float(roc_auc_score(t[:, i], p[:, i]))
                                 if (1 - t[:, i]).sum() > 0 else 0.0, 4),
                    'f1' : round(float(f1_score(t[:, i], b[:, i], zero_division=0)), 4),
                }

        return {
            'auc'        : round(mean_auc,    4),
            'f1'         : round(f1,          4),
            'map'        : round(mean_ap,     4),
            'kappa'      : round(kappa,       4),
            'exact_match': round(exact_match, 4),
            'per_class'  : per_class,
        }


# ─────────────────────────────────────────────
# LR SCHEDULE
# ─────────────────────────────────────────────

def build_optimizer(model, lr: float, backbone_lr: float, weight_decay: float):
    """Layer-wise LR: backbone gets 10× lower LR."""
    backbone_params, head_params = [], []
    backbone_names = {'backbone', 'effnet', 'vit', 'efficientnet'}

    for name, param in model.named_parameters():
        if any(name.startswith(b) for b in backbone_names):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'base_lr': backbone_lr},
        {'params': head_params,     'lr': lr,          'base_lr': lr},
    ], weight_decay=weight_decay, betas=(0.9, 0.999))


def cosine_warmup_lr(optimizer, epoch: int, warmup: int, total: int):
    if epoch < warmup:
        mult = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        mult = 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = pg['base_lr'] * mult


# ─────────────────────────────────────────────
# PROGRESSIVE RESIZING
# ─────────────────────────────────────────────

PROG_SIZES = {0: 384, 10: 448, 20: 512}  # epoch → image size


def get_progressive_size(epoch: int) -> int:
    size = 384
    for e, s in sorted(PROG_SIZES.items()):
        if epoch >= e:
            size = s
    return size


# ─────────────────────────────────────────────
# FREEZE / UNFREEZE SCHEDULE
# ─────────────────────────────────────────────

def set_trainable(model, stage: int):
    """
    Stage 0: freeze all backbone, train head only
    Stage 1: unfreeze top 1/3 of backbone
    Stage 2: unfreeze all
    """
    if stage == 0:
        for name, param in model.named_parameters():
            if any(name.startswith(b) for b in ['effnet', 'vit']):
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("[Train] Stage 0: backbone frozen")

    elif stage == 1:
        # Unfreeze last N blocks
        for name, param in model.named_parameters():
            param.requires_grad = True
            # Keep early layers frozen for EfficientNet
            if 'effnet.backbone.blocks.0' in name or \
               'effnet.backbone.blocks.1' in name:
                param.requires_grad = False
        print("[Train] Stage 1: top backbone layers unfrozen")

    else:
        for param in model.parameters():
            param.requires_grad = True
        print("[Train] Stage 2: all parameters trainable")


# ─────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device,
                epoch, accum_steps, mixup_alpha, cutmix_alpha) -> Tuple[float, Dict]:
    model.train()
    total_loss  = 0.0
    breakdown_sum = {}
    metrics     = MetricAccumulator()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        image   = batch['image'].to(device, non_blocking=True)
        labels  = batch['labels'].to(device, non_blocking=True)

        # Mixup / CutMix (alternate randomly)
        if np.random.rand() < 0.5 and mixup_alpha > 0:
            image, labels = mixup_data(image, labels, mixup_alpha)
        elif cutmix_alpha > 0:
            image, labels = cutmix_data(image, labels, cutmix_alpha)

        with autocast():
            outputs = model(image)
            loss, bd = criterion(outputs, labels)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        for k, v in bd.items():
            breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v

        # Accumulate metrics (use hard labels for metrics even if mixed)
        hard_labels = (labels >= 0.5).float()
        metrics.update(outputs['logits'].detach(), hard_labels)

    n = len(loader)
    avg_loss = total_loss / n
    for k in breakdown_sum:
        breakdown_sum[k] /= n

    m = metrics.compute()
    return avg_loss, {**breakdown_sum, **m}


# ─────────────────────────────────────────────
# VAL EPOCH
# ─────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    metrics    = MetricAccumulator()

    for batch in loader:
        image  = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast():
            outputs = model(image)
            loss, _ = criterion(outputs, labels)

        total_loss += loss.item()
        metrics.update(outputs['logits'], labels)

    return total_loss / len(loader), metrics.compute()


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self):
        """Load EMA weights into model (for eval)."""
        self.model.load_state_dict(
            {k: v.to(next(self.model.parameters()).device)
             for k, v in self.shadow.items()}
        )

    def restore(self, original_state: Dict):
        self.model.load_state_dict(original_state)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}  |  backbone: {args.backbone}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Build model ──────────────────────────
    if args.backbone == 'ensemble':
        model = build_ensemble(pretrained=True, drop_rate=args.drop_rate,
                               fusion=args.fusion).to(device)
    else:
        model = build_single(backbone=args.backbone, pretrained=True,
                             drop_rate=args.drop_rate).to(device)

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded pretrained: {args.pretrained}")

    # ── Optimiser + loss ─────────────────────
    optimizer = build_optimizer(model, args.lr, args.backbone_lr, args.weight_decay)
    criterion = CombinedLoss(label_smooth=args.label_smooth, aux_weight=0.3)
    scaler    = GradScaler()

    # ── SWA ──────────────────────────────────
    swa_model, swa_scheduler = None, None
    if not args.no_swa:
        swa_model     = AveragedModel(model)
        swa_start_ep  = int(args.epochs * args.swa_start)
        swa_scheduler = SWALR(optimizer,
                              swa_lr=args.lr * 0.1,
                              anneal_epochs=max(1, args.epochs - swa_start_ep))
        print(f"[SWA] Starts at epoch {swa_start_ep}")
    else:
        swa_start_ep = args.epochs + 1

    # ── EMA ───────────────────────────────────
    ema = EMA(model, decay=0.9997) if not args.no_ema else None

    # ── Resume ────────────────────────────────
    start_epoch = 0
    best        = {'auc': 0.0, 'f1': 0.0, 'kappa': 0.0}
    patience_ctr= 0
    history     = []

    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best        = ckpt.get('best', best)
        print(f"[Resume] Epoch {start_epoch}")

    # ── Stage schedule ────────────────────────
    set_trainable(model, stage=0)  # freeze backbone initially

    # ── Training loop ─────────────────────────
    size = get_progressive_size(0)
    train_loader, val_loader = build_dataloaders(
        args.data, args.annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=size,
    )

    for epoch in range(start_epoch, args.epochs):
        t0   = time.time()

        # Progressive resizing
        new_size = get_progressive_size(epoch)
        if new_size != size:
            size = new_size
            train_loader, val_loader = build_dataloaders(
                args.data, args.annotations,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                size=size,
            )
            print(f"  [ProgResize] Switched to {size}px")

        # Unfreeze schedule
        if epoch == args.warmup_epochs:
            set_trainable(model, stage=1)
        if epoch == args.warmup_epochs + 5:
            set_trainable(model, stage=2)

        # LR schedule (cosine warmup, unless SWA active)
        if epoch < swa_start_ep:
            cosine_warmup_lr(optimizer, epoch, args.warmup_epochs, swa_start_ep)

        current_lr = optimizer.param_groups[1]['lr']

        # Train
        train_loss, train_m = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, args.accum_steps, args.mixup_alpha, args.cutmix_alpha,
        )

        # Update EMA
        if ema:
            ema.update()

        # SWA update
        if swa_model and epoch >= swa_start_ep:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validate (use EMA model if available)
        if ema:
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply()
            val_loss, val_m = val_epoch(model, val_loader, criterion, device)
            ema.restore(orig_state)
        else:
            val_loss, val_m = val_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1:3d}/{args.epochs} | {elapsed:.0f}s | LR={current_lr:.2e}")
        print(f"  Train: loss={train_loss:.4f} | AUC={train_m['auc']:.4f} | "
              f"F1={train_m['f1']:.4f} | mAP={train_m['map']:.4f}")
        print(f"  Val:   loss={val_loss:.4f}   | AUC={val_m['auc']:.4f} | "
              f"F1={val_m['f1']:.4f} | mAP={val_m['map']:.4f} | κ={val_m['kappa']:.4f}")
        print(f"  Per-class AUC: " +
              " | ".join([f"{c}={val_m['per_class'].get(c,{}).get('auc','—')}"
                          for c in ['N','D','G','C','A','H','M','O']]))

        history.append({
            'epoch': epoch + 1, 'lr': current_lr, 'size': size,
            'train': {k: v for k, v in train_m.items() if k != 'per_class'},
            'val'  : {k: v for k, v in val_m.items()   if k != 'per_class'},
        })
        with open(Path(args.checkpoint_dir) / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save checkpoint
        ckpt = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_metrics': val_m, 'best': best,
        }
        torch.save(ckpt, Path(args.checkpoint_dir) / 'last.pth')

        # Best checkpoints per metric
        improved = False
        for metric in ['auc', 'f1', 'kappa']:
            if val_m[metric] > best[metric]:
                best[metric] = val_m[metric]
                torch.save(ckpt, Path(args.checkpoint_dir) / f'best_{metric}.pth')
                print(f"  ✓ New best {metric.upper()}: {best[metric]:.4f}")
                improved = True

        if not improved:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/15)")
            if patience_ctr >= 15:
                print("[Train] Early stopping triggered.")
                break
        else:
            patience_ctr = 0

    # ── SWA final BN update ───────────────────
    if swa_model:
        print("\n[SWA] Updating batch norm statistics...")
        update_bn(train_loader, swa_model, device=device)
        _, swa_m = val_epoch(swa_model, val_loader, criterion, device)
        print(f"[SWA] AUC={swa_m['auc']:.4f} | F1={swa_m['f1']:.4f}")
        torch.save({'model': swa_model.module.state_dict(), 'val_metrics': swa_m},
                   Path(args.checkpoint_dir) / 'swa_model.pth')

    print(f"\n[Done] Best AUC={best['auc']:.4f} | F1={best['f1']:.4f} | κ={best['kappa']:.4f}")


if __name__ == '__main__':
    main()
