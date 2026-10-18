"""
Pathological Occurrence Graph (POG) — standalone utilities.

Provides:
  - build_cooccurrence_matrix()  : compute empirical co-occurrence from labels
  - visualise_graph()            : matplotlib adjacency heatmap
  - GraphStats                   : summary statistics for the POG adjacency
"""

import torch
import numpy as np


LESION_NAMES = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates',
                'Soft Exudates', 'Neovascularisation']


def build_cooccurrence_matrix(lesion_labels: np.ndarray,
                               smooth: float = 1.0) -> np.ndarray:
    """
    Compute lesion co-occurrence probability matrix from binary label matrix.

    Args:
        lesion_labels : (N, K) binary array — N samples, K lesion types
        smooth        : Laplace smoothing to avoid zero entries

    Returns:
        adj : (K, K) symmetric matrix with values in (0, 1)
              adj[i,j] = P(lesion_j present | lesion_i present)
    """
    N, K  = lesion_labels.shape
    adj   = np.zeros((K, K), dtype=np.float32)

    for i in range(K):
        mask_i = lesion_labels[:, i] == 1
        count_i = mask_i.sum() + smooth
        for j in range(K):
            count_ij = (mask_i & (lesion_labels[:, j] == 1)).sum() + smooth
            adj[i, j] = count_ij / count_i

    # Symmetrise: use geometric mean
    adj = np.sqrt(adj * adj.T)
    np.fill_diagonal(adj, 1.0)
    return adj


def adj_to_init_logits(adj: np.ndarray) -> torch.Tensor:
    """
    Convert a (K, K) probability matrix to logit-space for use as
    the initial value of POG.adj_logits (before sigmoid).
    """
    adj_clipped = np.clip(adj, 1e-4, 1 - 1e-4)
    logits = np.log(adj_clipped / (1 - adj_clipped))
    return torch.tensor(logits, dtype=torch.float32)


class GraphStats:
    """Simple container for POG statistics printed during training."""

    def __init__(self, adj: torch.Tensor):
        adj_np = adj.cpu().numpy()
        self.mean_weight = float(adj_np[~np.eye(len(adj_np), dtype=bool)].mean())
        self.max_pair    = self._top_pair(adj_np)
        self.min_pair    = self._min_pair(adj_np)

    @staticmethod
    def _top_pair(adj):
        K = adj.shape[0]
        best, best_pair = -1, (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                if adj[i, j] > best:
                    best, best_pair = adj[i, j], (i, j)
        i, j = best_pair
        return f"{LESION_NAMES[i]} ↔ {LESION_NAMES[j]} ({adj[i,j]:.3f})"

    @staticmethod
    def _min_pair(adj):
        K = adj.shape[0]
        worst, worst_pair = 2.0, (0, 1)
        for i in range(K):
            for j in range(i + 1, K):
                if adj[i, j] < worst:
                    worst, worst_pair = adj[i, j], (i, j)
        i, j = worst_pair
        return f"{LESION_NAMES[i]} ↔ {LESION_NAMES[j]} ({adj[i,j]:.3f})"

    def __str__(self):
        return (f"POG | mean edge weight: {self.mean_weight:.3f} | "
                f"strongest: {self.max_pair} | "
                f"weakest:   {self.min_pair}")
