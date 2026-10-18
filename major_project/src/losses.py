"""
Loss Functions for AC-ViT-POG
==============================
1. AsymmetricLoss       — handles positive/negative imbalance in multi-label lesion detection
2. OrdinalLoss          — smooth ordinal loss for DR (0-4) and DME (0-2) grading
3. AnatomicalRegLoss    — regularises anatomical constraint module zone radii
4. CombinedLoss         — weighted sum of all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1.  ASYMMETRIC LOSS  (ASL)
# ─────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Reference: Ben-Baruch et al., ICCV 2021.

    Key idea: down-weight easy negatives by shifting the probability
    for negatives (gamma_neg) more aggressively than positives (gamma_pos).
    Clip negatives below probability_margin to hard-ignore very easy negatives.

    Handles IDRiD's severe positive/negative imbalance
    (e.g., NV present in only ~8% of images).
    """

    def __init__(self,
                 gamma_neg: float = 4.0,
                 gamma_pos: float = 0.0,
                 clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, K) raw (un-sigmoid'd)
        targets : (B, K) binary float {0, 1}
        """
        probs    = torch.sigmoid(logits)
        probs_m  = probs

        # Shift negatives down by clip margin
        if self.clip > 0:
            probs_m = (probs + self.clip).clamp(max=1)

        # Positive loss
        loss_pos = targets       * torch.log(probs.clamp(min=self.eps))
        # Negative loss (using shifted probabilities)
        loss_neg = (1 - targets) * torch.log((1 - probs_m).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            p_t    = probs * targets + (1 - probs) * (1 - targets)
            modulation = (
                (1 - p_t) ** (self.gamma_pos * targets + self.gamma_neg * (1 - targets))
            )
            loss = loss * modulation

        return -loss.mean()


# ─────────────────────────────────────────────
# 2.  ORDINAL LOSS
# ─────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    """
    Soft ordinal loss: combines cross-entropy with Earth Mover's Distance
    penalty to respect grade ordering (e.g., predicting grade 1 when
    truth is grade 2 is less wrong than predicting grade 4).

    loss = α * CE(logits, target) + (1-α) * EMD(softmax(logits), one_hot(target))
    """

    def __init__(self, num_classes: int, alpha: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha       = alpha
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)
        targets : (B,) long
        """
        ce_loss = self.ce(logits, targets)

        # EMD: cumulative sum of softmax vs one-hot CDF
        probs = F.softmax(logits, dim=-1)              # (B, C)
        one_hot = F.one_hot(targets, self.num_classes).float()
        cdf_pred   = probs.cumsum(dim=-1)
        cdf_target = one_hot.cumsum(dim=-1)
        emd_loss   = (cdf_pred - cdf_target).abs().mean()

        return self.alpha * ce_loss + (1 - self.alpha) * emd_loss


# ─────────────────────────────────────────────
# 3.  ANATOMICAL REGULARISATION LOSS
# ─────────────────────────────────────────────

class AnatomicalRegLoss(nn.Module):
    """
    Keeps the learnable zone radii in the AnatomicalConstraintModule
    within clinically plausible ranges and penalises collapse or explosion.

    macula_r       should stay ~[0.08, 0.18]  (≈ 1 OD diameter)
    peripapillary_r should stay ~[0.10, 0.22]
    """

    def __init__(self,
                 macula_target: float        = 0.12,
                 peripapillary_target: float = 0.15,
                 weight: float               = 0.01):
        super().__init__()
        self.macula_target        = macula_target
        self.peripapillary_target = peripapillary_target
        self.weight               = weight

    def forward(self, anat_module) -> torch.Tensor:
        loss = (
            (anat_module.macula_r.abs()        - self.macula_target) ** 2 +
            (anat_module.peripapillary_r.abs() - self.peripapillary_target) ** 2
        )
        return self.weight * loss


# ─────────────────────────────────────────────
# 4.  COMBINED LOSS
# ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted combination of:
      - OrdinalLoss for DR grading         (weight: w_dr)
      - OrdinalLoss for DME grading        (weight: w_dme)
      - AsymmetricLoss for lesion labels   (weight: w_lesion)
      - AnatomicalRegLoss                  (weight: built-in)

    Weights tuned to IDRiD task difficulty:
      DR grading is the hardest (highest weight).
    """

    def __init__(
        self,
        w_dr:     float = 1.0,
        w_dme:    float = 0.8,
        w_lesion: float = 0.5,
    ):
        super().__init__()
        self.w_dr     = w_dr
        self.w_dme    = w_dme
        self.w_lesion = w_lesion

        self.dr_loss     = OrdinalLoss(num_classes=5, alpha=0.5)
        self.dme_loss    = OrdinalLoss(num_classes=3, alpha=0.5)
        self.lesion_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        self.anat_reg    = AnatomicalRegLoss()

    def forward(
        self,
        outputs: dict,
        dr_targets:     torch.Tensor,
        dme_targets:    torch.Tensor,
        lesion_targets: torch.Tensor,
        anat_module=None,
    ) -> Tuple[torch.Tensor, dict]:
        from typing import Tuple   # local import avoids circular

        l_dr     = self.dr_loss(outputs['dr_logits'],      dr_targets)
        l_dme    = self.dme_loss(outputs['dme_logits'],    dme_targets)
        l_lesion = self.lesion_loss(outputs['lesion_logits'], lesion_targets)

        total = self.w_dr * l_dr + self.w_dme * l_dme + self.w_lesion * l_lesion

        if anat_module is not None:
            l_anat = self.anat_reg(anat_module)
            total  = total + l_anat
        else:
            l_anat = torch.tensor(0.0)

        breakdown = {
            'loss_dr'    : l_dr.item(),
            'loss_dme'   : l_dme.item(),
            'loss_lesion': l_lesion.item(),
            'loss_anat'  : l_anat.item() if isinstance(l_anat, torch.Tensor) else 0.0,
            'loss_total' : total.item(),
        }
        return total, breakdown
