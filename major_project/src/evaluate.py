"""
Evaluation Script — AC-ViT-POG
================================
Full test-set evaluation with:
  - DR / DME quadratic weighted kappa
  - Per-lesion AUC, F1, sensitivity, specificity
  - Confusion matrix
  - Per-grade accuracy breakdown
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report,
)

from dataset import IDRiDCombinedDataset, build_dataloaders
from model   import build_acvit_pog

LESION_NAMES = ['Microaneurysms','Haemorrhages','Hard Exudates','Soft Exudates','Neovascularisation']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str, default='data/idrid')
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--output',     type=str, default='results.json')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dr_preds, dr_targets    = [], []
    dme_preds, dme_targets  = [], []
    les_preds, les_targets  = [], []

    for batch in loader:
        image    = batch['image'].to(device)
        od_xy    = batch['od_xy'].to(device)
        fovea_xy = batch['fovea_xy'].to(device)

        with autocast():
            outputs = model(image, od_xy, fovea_xy)

        dr_preds.append(outputs['dr_logits'].argmax(1).cpu().numpy())
        dr_targets.append(batch['dr_grade'].cpu().numpy())
        dme_preds.append(outputs['dme_logits'].argmax(1).cpu().numpy())
        dme_targets.append(batch['dme_grade'].cpu().numpy())
        les_preds.append(torch.sigmoid(outputs['lesion_logits']).cpu().numpy())
        les_targets.append(batch['lesion_labels'].cpu().numpy())

    dr_p  = np.concatenate(dr_preds)
    dr_t  = np.concatenate(dr_targets)
    dme_p = np.concatenate(dme_preds)
    dme_t = np.concatenate(dme_targets)
    les_p = np.concatenate(les_preds,   axis=0)
    les_t = np.concatenate(les_targets, axis=0)

    results = {}

    # DR grading
    results['dr_kappa'] = float(cohen_kappa_score(dr_t, dr_p, weights='quadratic'))
    results['dr_accuracy'] = float((dr_p == dr_t).mean())
    results['dr_confusion'] = confusion_matrix(dr_t, dr_p).tolist()
    results['dr_per_grade'] = {
        str(g): float((dr_p[dr_t == g] == g).mean())
        for g in range(5) if (dr_t == g).sum() > 0
    }

    # DME grading
    results['dme_kappa']    = float(cohen_kappa_score(dme_t, dme_p, weights='quadratic'))
    results['dme_accuracy'] = float((dme_p == dme_t).mean())
    results['dme_confusion'] = confusion_matrix(dme_t, dme_p).tolist()

    # Lesion detection
    les_bin = (les_p >= 0.5).astype(int)
    lesion_results = {}
    for i, name in enumerate(LESION_NAMES):
        col_p, col_t = les_p[:, i], les_t[:, i]
        col_b        = les_bin[:, i]
        has_pos = col_t.sum() > 0
        has_neg = (1 - col_t).sum() > 0
        r = {
            'auc'        : float(roc_auc_score(col_t, col_p)) if has_pos and has_neg else None,
            'f1'         : float(f1_score(col_t, col_b, zero_division=0)),
            'sensitivity': float(col_b[col_t == 1].mean()) if has_pos else None,
            'specificity': float((1 - col_b)[col_t == 0].mean()) if has_neg else None,
            'prevalence' : float(col_t.mean()),
        }
        lesion_results[name] = r

    results['lesion'] = lesion_results
    results['mean_lesion_auc'] = float(np.mean([
        v['auc'] for v in lesion_results.values() if v['auc'] is not None
    ]))

    return results


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader = build_dataloaders(
        root=args.data, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=2,
    )

    model = build_acvit_pog().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    results = evaluate(model, val_loader, device)

    print("\n═══ RESULTS ═══")
    print(f"DR  Quadratic Kappa : {results['dr_kappa']:.4f}")
    print(f"DME Quadratic Kappa : {results['dme_kappa']:.4f}")
    print(f"Mean Lesion AUC     : {results['mean_lesion_auc']:.4f}")
    print("\nPer-lesion:")
    for name, r in results['lesion'].items():
        print(f"  {name:<22} AUC={r['auc'] or 'N/A':.3f}  "
              f"F1={r['f1']:.3f}  "
              f"Sens={r['sensitivity'] or 'N/A':.3f}  "
              f"Spec={r['specificity'] or 'N/A':.3f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
