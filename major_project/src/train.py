"""
Training Script — AC-ViT-POG
============================
Trains the Anatomically-Constrained ViT + POG model on IDRiD.

Usage:
    python src/train.py --data data/idrid --epochs 50 --batch_size 8

Key training decisions:
  - Layer-wise learning rate decay (ViT backbone gets 10× lower LR)
  - Cosine LR schedule with linear warm-up
  - Mixed precision (FP16) via torch.amp
  - Gradient clipping at 1.0
  - Early stopping on val DR kappa (patience=10)
  - Best checkpoint saved to checkpoints/best.pth
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score

from dataset import build_dataloaders
from model   import build_acvit_pog
from losses  import CombinedLoss
from graph   import GraphStats


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Train AC-ViT-POG on IDRiD')
    p.add_argument('--data',          type=str,   default='data/idrid')
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--backbone_lr',   type=float, default=1e-5,
                   help='LR for ViT backbone (10× lower)')
    p.add_argument('--weight_decay',  type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--image_size',    type=int,   default=512)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--checkpoint_dir',type=str,   default='checkpoints')
    p.add_argument('--pretrained',    type=str,   default=None,
                   help='Path to pretrained ViT-B/16 weights')
    p.add_argument('--resume',        type=str,   default=None,
                   help='Resume from checkpoint')
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class MetricsAccumulator:
    """Accumulates predictions across batches for epoch-level metrics."""

    def __init__(self):
        self.dr_preds   = []
        self.dr_targets = []
        self.dme_preds  = []
        self.dme_targets= []
        self.lesion_preds   = []
        self.lesion_targets = []

    def update(self, outputs: dict, batch: dict):
        self.dr_preds.append(outputs['dr_logits'].argmax(1).cpu().numpy())
        self.dr_targets.append(batch['dr_grade'].cpu().numpy())
        self.dme_preds.append(outputs['dme_logits'].argmax(1).cpu().numpy())
        self.dme_targets.append(batch['dme_grade'].cpu().numpy())
        self.lesion_preds.append(torch.sigmoid(outputs['lesion_logits']).cpu().numpy())
        self.lesion_targets.append(batch['lesion_labels'].cpu().numpy())

    def compute(self) -> Dict[str, float]:
        dr_p  = np.concatenate(self.dr_preds)
        dr_t  = np.concatenate(self.dr_targets)
        dme_p = np.concatenate(self.dme_preds)
        dme_t = np.concatenate(self.dme_targets)
        les_p = np.concatenate(self.lesion_preds, axis=0)  # (N, 5)
        les_t = np.concatenate(self.lesion_targets, axis=0)

        dr_kappa  = cohen_kappa_score(dr_t,  dr_p,  weights='quadratic')
        dme_kappa = cohen_kappa_score(dme_t, dme_p, weights='quadratic')

        # Per-lesion AUC (skip if only one class present)
        auc_scores = []
        for i in range(les_t.shape[1]):
            if les_t[:, i].sum() > 0 and (1 - les_t[:, i]).sum() > 0:
                auc_scores.append(roc_auc_score(les_t[:, i], les_p[:, i]))
        mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

        # F1 for lesion detection (threshold at 0.5)
        les_bin = (les_p >= 0.5).astype(int)
        f1 = f1_score(les_t, les_bin, average='macro', zero_division=0)

        return {
            'dr_kappa'  : round(float(dr_kappa),  4),
            'dme_kappa' : round(float(dme_kappa), 4),
            'lesion_auc': round(mean_auc, 4),
            'lesion_f1' : round(float(f1), 4),
        }


# ─────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    epoch:         int,
) -> float:
    """Returns the LR multiplier for the current epoch."""
    if epoch < warmup_epochs:
        lr_mult = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr_mult  = 0.5 * (1.0 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = pg['base_lr'] * lr_mult
    return lr_mult


def build_optimizer(model, lr: float, backbone_lr: float, weight_decay: float):
    """
    Layer-wise LR decay:
      - ViT backbone (patch_embed + transformer): backbone_lr
      - New modules (anat_pos_embed, anat_constraint, lesion_extractor,
                     pog, heads):               lr
    """
    backbone_params = []
    new_params      = []

    backbone_names = {'patch_embed', 'transformer', 'norm'}

    for name, param in model.named_parameters():
        if any(name.startswith(bn) for bn in backbone_names):
            backbone_params.append(param)
        else:
            new_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'base_lr': backbone_lr},
        {'params': new_params,      'lr': lr,          'base_lr': lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
) -> Tuple[float, Dict]:
    model.train()
    total_loss = 0.0
    loss_breakdown = {k: 0.0 for k in ['loss_dr','loss_dme','loss_lesion','loss_anat']}
    metrics = MetricsAccumulator()
    n_batches = len(loader)

    for step, batch in enumerate(loader):
        image    = batch['image'].to(device)
        od_xy    = batch['od_xy'].to(device)
        fovea_xy = batch['fovea_xy'].to(device)
        dr_g     = batch['dr_grade'].to(device)
        dme_g    = batch['dme_grade'].to(device)
        les_l    = batch['lesion_labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(image, od_xy, fovea_xy)
            loss, breakdown = criterion(
                outputs, dr_g, dme_g, les_l,
                anat_module=model.anat_constraint,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_breakdown:
            loss_breakdown[k] += breakdown.get(k, 0.0)

        metrics.update(outputs, batch)

        if (step + 1) % max(1, n_batches // 5) == 0:
            print(f"  Epoch {epoch+1} [{step+1}/{n_batches}] "
                  f"loss={loss.item():.4f} "
                  f"dr={breakdown['loss_dr']:.4f} "
                  f"dme={breakdown['loss_dme']:.4f} "
                  f"lesion={breakdown['loss_lesion']:.4f}")

    avg_loss = total_loss / n_batches
    for k in loss_breakdown:
        loss_breakdown[k] /= n_batches
    epoch_metrics = metrics.compute()

    return avg_loss, {**loss_breakdown, **epoch_metrics}


# ─────────────────────────────────────────────
# VALIDATE ONE EPOCH
# ─────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    metrics    = MetricsAccumulator()

    for batch in loader:
        image    = batch['image'].to(device)
        od_xy    = batch['od_xy'].to(device)
        fovea_xy = batch['fovea_xy'].to(device)
        dr_g     = batch['dr_grade'].to(device)
        dme_g    = batch['dme_grade'].to(device)
        les_l    = batch['lesion_labels'].to(device)

        with autocast():
            outputs = model(image, od_xy, fovea_xy)
            loss, _ = criterion(outputs, dr_g, dme_g, les_l)

        total_loss += loss.item()
        metrics.update(outputs, batch)

    avg_loss     = total_loss / len(loader)
    epoch_metrics = metrics.compute()
    return avg_loss, epoch_metrics


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        root        = args.data,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        image_size  = args.image_size,
    )

    # ── Model ────────────────────────────────
    model = build_acvit_pog(pretrained_vit_path=args.pretrained).to(device)

    # ── Optimiser + Loss ─────────────────────
    optimizer = build_optimizer(model, args.lr, args.backbone_lr, args.weight_decay)
    criterion = CombinedLoss(w_dr=1.0, w_dme=0.8, w_lesion=0.5)
    scaler    = GradScaler()

    # ── Resume ───────────────────────────────
    start_epoch   = 0
    best_kappa    = -1.0
    patience_ctr  = 0
    history       = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_kappa  = ckpt.get('best_kappa', -1.0)
        print(f"[Train] Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Adjust LR
        cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs, epoch)
        current_lr = optimizer.param_groups[1]['lr']

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_metrics = val_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # POG graph stats
        adj = torch.sigmoid(model.pog.adj_logits).detach()
        gs  = GraphStats(adj)

        print(f"\nEpoch {epoch+1:3d}/{args.epochs} | "
              f"LR={current_lr:.2e} | {elapsed:.0f}s")
        print(f"  Train loss={train_loss:.4f} | "
              f"DR κ={train_metrics['dr_kappa']:.4f} | "
              f"DME κ={train_metrics['dme_kappa']:.4f} | "
              f"Lesion AUC={train_metrics['lesion_auc']:.4f}")
        print(f"  Val   loss={val_loss:.4f}   | "
              f"DR κ={val_metrics['dr_kappa']:.4f} | "
              f"DME κ={val_metrics['dme_kappa']:.4f} | "
              f"Lesion AUC={val_metrics['lesion_auc']:.4f}")
        print(f"  {gs}")

        # History
        history.append({
            'epoch': epoch + 1,
            'lr'   : current_lr,
            'train': {**train_metrics, 'loss': train_loss},
            'val'  : {**val_metrics,   'loss': val_loss},
        })
        with open(Path(args.checkpoint_dir) / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Checkpoint
        ckpt = {
            'epoch'      : epoch,
            'model'      : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_kappa' : best_kappa,
        }
        torch.save(ckpt, Path(args.checkpoint_dir) / 'last.pth')

        # Best model
        val_kappa = val_metrics['dr_kappa']
        if val_kappa > best_kappa:
            best_kappa   = val_kappa
            patience_ctr = 0
            torch.save(ckpt, Path(args.checkpoint_dir) / 'best.pth')
            print(f"  ✓ New best DR kappa: {best_kappa:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/10)")
            if patience_ctr >= 10:
                print("[Train] Early stopping triggered.")
                break

    print(f"\n[Train] Complete. Best DR kappa: {best_kappa:.4f}")


if __name__ == '__main__':
    main()
