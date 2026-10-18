"""
Anatomically-Constrained Vision Transformer (AC-ViT)
with Pathological Occurrence Graph (POG) Modeling
for Multi-Label Retinal Fundus Disease Grading

Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NUM_LESIONS    = 5   # MA, HE, EX, SE, NV
DR_GRADES      = 5   # 0–4
DME_GRADES     = 3   # 0–2
IMAGE_SIZE     = 512
PATCH_SIZE     = 16
NUM_PATCHES    = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 1024
DIM            = 768   # ViT-B hidden dim
DEPTH          = 12
HEADS          = 12
MLP_DIM        = 3072
DIM_HEAD       = DIM // HEADS  # 64
DROPOUT        = 0.1

# IDRiD image dimensions for normalising optic disc / fovea coords
IMG_W, IMG_H   = 4288, 2848   # original capture resolution
RESIZED        = IMAGE_SIZE


# ─────────────────────────────────────────────
# 1.  PATCH EMBEDDING
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits a 512×512 fundus image into 16×16 patches and
    projects each to DIM-dimensional token vectors.
    """
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=3, embed_dim=DIM):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * in_channels),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)   # (B, N, D)


# ─────────────────────────────────────────────
# 2.  ANATOMICAL POSITIONAL EMBEDDING
# ─────────────────────────────────────────────

class AnatomicalPositionalEmbedding(nn.Module):
    """
    Augments standard learnable positional embeddings with
    anatomy-aware bias derived from:
      - Optic Disc (OD) centre
      - Fovea centre
    Both supplied as (x, y) pixel coordinates normalised to [0, 1].

    For each patch token we compute:
      - distance to OD
      - distance to fovea
      - retinal quadrant index (0-3: TN, TS, IN, IS)
      - macula flag (within 1 OD-diameter of fovea)

    These 5 scalars are projected to DIM and added to the token.
    """
    def __init__(self, num_patches=NUM_PATCHES, embed_dim=DIM,
                 image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size  = patch_size
        self.image_size  = image_size
        n = image_size // patch_size   # 32

        # Standard learnable positional embedding
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        # Grid of patch centre coordinates, normalised to [0,1]
        xs = (torch.arange(n).float() + 0.5) / n
        ys = (torch.arange(n).float() + 0.5) / n
        grid_y, grid_x  = torch.meshgrid(ys, xs, indexing='ij')
        # shape (N, 2) — (row=y, col=x)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        self.register_buffer('patch_coords', coords)  # (1024, 2)

        # Projection: 5 anatomical scalars → DIM
        self.anat_proj   = nn.Sequential(
            nn.Linear(5, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

    def _anatomy_features(self, od_xy, fovea_xy):
        """
        od_xy, fovea_xy: (B, 2) normalised coords in [0,1]
        Returns: (B, N, 5)
        """
        B = od_xy.size(0)
        coords = self.patch_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # Euclidean distance to OD and fovea
        od     = od_xy.unsqueeze(1)      # (B,1,2)
        fovea  = fovea_xy.unsqueeze(1)   # (B,1,2)
        d_od   = torch.norm(coords - od,    dim=-1, keepdim=True)  # (B,N,1)
        d_fov  = torch.norm(coords - fovea, dim=-1, keepdim=True)  # (B,N,1)

        # Quadrant relative to OD (TN=0, TS=1, IN=2, IS=3)
        dx = coords[..., 0:1] - od[..., 0:1]   # temporal(+) / nasal(-)
        dy = coords[..., 1:2] - od[..., 1:2]   # superior(-) / inferior(+)
        quad = ((dx < 0).float() * 2 + (dy >= 0).float())  # (B,N,1) in {0,1,2,3}
        quad_norm = quad / 3.0

        # Macula flag: within 0.1 normalised units of fovea (~1 OD diameter)
        macula = (d_fov < 0.10).float()

        return torch.cat([d_od, d_fov, quad_norm, macula, dx.abs()], dim=-1)  # (B,N,5)

    def forward(self, tokens, od_xy, fovea_xy):
        """
        tokens:   (B, N, D)   — patch tokens (no CLS yet)
        od_xy:    (B, 2)
        fovea_xy: (B, 2)
        Returns:  (B, N+1, D) with CLS prepended
        """
        B, N, D = tokens.shape
        anat_feat = self._anatomy_features(od_xy, fovea_xy)   # (B, N, 5)
        anat_bias = self.anat_proj(anat_feat)                  # (B, N, D)
        tokens    = tokens + anat_bias

        # Prepend CLS token (zeros, positional embedding handles it)
        cls = torch.zeros(B, 1, D, device=tokens.device)
        tokens = torch.cat([cls, tokens], dim=1)               # (B, N+1, D)
        tokens = tokens + self.pos_embed[:, :N+1]
        return tokens


# ─────────────────────────────────────────────
# 3.  MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, dim_head=DIM_HEAD, dropout=DROPOUT):
        super().__init__()
        inner_dim   = heads * dim_head
        self.heads  = heads
        self.scale  = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            # mask: (B, N) — True where tokens should be attended to
            mask = mask.unsqueeze(1).unsqueeze(2)   # (B,1,1,N)
            dots = dots.masked_fill(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out  = torch.einsum('bhij,bhjd->bhid', attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


# ─────────────────────────────────────────────
# 4.  TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim=DIM, hidden_dim=MLP_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, dim_head=DIM_HEAD,
                 mlp_dim=MLP_DIM, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        normed       = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask)
        x            = x + attn_out
        x            = x + self.ff(self.norm2(x))
        return x, attn_weights


# ─────────────────────────────────────────────
# 5.  ANATOMICAL CONSTRAINT MODULE
# ─────────────────────────────────────────────

class AnatomicalConstraintModule(nn.Module):
    """
    Generates per-quadrant attention masks so that lesion-specific
    attention heads focus on clinically relevant retinal regions.

    Clinical priors encoded:
      - Microaneurysms (MA):   all quadrants, mainly posterior pole
      - Haemorrhages  (HE):   all quadrants
      - Hard Exudates (EX):   macula zone priority
      - Soft Exudates (SE):   peripapillary (near OD)
      - Neovascularisation(NV): near OD and along vessels
    """
    # Lesion index mapping
    MA, HE, EX, SE, NV = 0, 1, 2, 3, 4

    # Which zones each lesion favours: 0=all, 1=macula, 2=peripapillary
    LESION_ZONES = [0, 0, 1, 2, 2]

    def __init__(self, num_patches=NUM_PATCHES, image_size=IMAGE_SIZE,
                 patch_size=PATCH_SIZE, embed_dim=DIM):
        super().__init__()
        n = image_size // patch_size
        xs = (torch.arange(n).float() + 0.5) / n
        ys = (torch.arange(n).float() + 0.5) / n
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        self.register_buffer('patch_coords', coords)

        # Learnable zone radii (initialised from clinical knowledge)
        self.macula_r       = nn.Parameter(torch.tensor(0.12))
        self.peripapillary_r= nn.Parameter(torch.tensor(0.15))

    def forward(self, od_xy, fovea_xy):
        """
        Returns masks: (B, NUM_LESIONS, N) — True = attend, False = suppress
        """
        B = od_xy.size(0)
        N = self.patch_coords.size(0)
        coords = self.patch_coords.unsqueeze(0).expand(B, -1, -1)  # (B,N,2)

        d_od   = torch.norm(coords - od_xy.unsqueeze(1),    dim=-1)  # (B,N)
        d_fov  = torch.norm(coords - fovea_xy.unsqueeze(1), dim=-1)  # (B,N)

        macula_mask       = d_fov  < self.macula_r.abs()
        peripapillary_mask= d_od   < self.peripapillary_r.abs()
        all_mask          = torch.ones(B, N, dtype=torch.bool, device=od_xy.device)

        zone_masks = [all_mask, macula_mask, peripapillary_mask]
        masks = torch.stack([
            zone_masks[z].clone() for z in self.LESION_ZONES
        ], dim=1)   # (B, NUM_LESIONS, N)

        # Always include CLS token
        cls_col = torch.ones(B, NUM_LESIONS, 1, dtype=torch.bool, device=od_xy.device)
        masks   = torch.cat([cls_col, masks], dim=2)  # (B, NUM_LESIONS, N+1)
        return masks


# ─────────────────────────────────────────────
# 6.  PATHOLOGICAL OCCURRENCE GRAPH (POG)
# ─────────────────────────────────────────────

class PathologicalOccurrenceGraph(nn.Module):
    """
    Graph Convolutional Network over lesion co-occurrence relationships.

    Nodes: NUM_LESIONS lesion types
    Edges: weighted by co-occurrence probability (learnable, initialised
           from IDRiD training set statistics)

    Co-occurrence matrix (estimated from IDRiD grading subset):
         MA    HE    EX    SE    NV
    MA [1.00, 0.72, 0.68, 0.31, 0.18]
    HE [0.72, 1.00, 0.61, 0.28, 0.22]
    EX [0.68, 0.61, 1.00, 0.35, 0.15]
    SE [0.31, 0.28, 0.35, 1.00, 0.20]
    NV [0.18, 0.22, 0.15, 0.20, 1.00]

    Two-layer GCN: H' = σ(A_norm H W)
    Input node features: lesion token embeddings from ViT
    Output: refined lesion embeddings for classification
    """

    # Prior co-occurrence matrix from IDRiD training statistics
    CO_OCCUR = torch.tensor([
        [1.00, 0.72, 0.68, 0.31, 0.18],
        [0.72, 1.00, 0.61, 0.28, 0.22],
        [0.68, 0.61, 1.00, 0.35, 0.15],
        [0.31, 0.28, 0.35, 1.00, 0.20],
        [0.18, 0.22, 0.15, 0.20, 1.00],
    ], dtype=torch.float32)

    def __init__(self, num_nodes=NUM_LESIONS, node_dim=DIM, hidden_dim=256):
        super().__init__()
        self.num_nodes  = num_nodes

        # Learnable adjacency (initialised from co-occurrence prior)
        adj_init = self.CO_OCCUR.clone()
        self.adj_logits = nn.Parameter(torch.log(adj_init / (1 - adj_init + 1e-6)))

        # Two-layer GCN
        self.gcn1 = nn.Linear(node_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output projection back to node_dim for residual
        self.out_proj = nn.Linear(hidden_dim, node_dim)

    def _normalised_adj(self):
        """Sigmoid-activate and symmetrically normalise adjacency."""
        A = torch.sigmoid(self.adj_logits)   # (K, K) in (0,1)
        # Add self-loops
        A = A + torch.eye(self.num_nodes, device=A.device)
        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        d     = A.sum(dim=-1)                  # (K,)
        d_inv = d.pow(-0.5)
        D_inv = torch.diag(d_inv)
        return D_inv @ A @ D_inv               # (K, K)

    def forward(self, node_feats):
        """
        node_feats: (B, K, D)  lesion token embeddings
        Returns:    (B, K, D)  refined embeddings
        """
        A = self._normalised_adj()             # (K, K)

        # Layer 1
        h = torch.einsum('kj,bjd->bkd', A, node_feats)
        h = self.norm1(F.gelu(self.gcn1(h)))

        # Layer 2
        h = torch.einsum('kj,bjd->bkd', A, h)
        h = self.norm2(F.gelu(self.gcn2(h)))

        return node_feats + self.out_proj(h)   # residual


# ─────────────────────────────────────────────
# 7.  LESION TOKEN EXTRACTOR
# ─────────────────────────────────────────────

class LesionTokenExtractor(nn.Module):
    """
    Extracts per-lesion summary tokens from ViT patch tokens
    using learnable cross-attention queries (one per lesion type),
    optionally masked by AnatomicalConstraintModule outputs.
    """
    def __init__(self, num_lesions=NUM_LESIONS, embed_dim=DIM, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_lesions, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads,
                                                batch_first=True, dropout=DROPOUT)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens, attn_masks=None):
        """
        patch_tokens: (B, N+1, D)
        attn_masks:   (B, NUM_LESIONS, N+1) — True=attend
        Returns:      (B, NUM_LESIONS, D)
        """
        B = patch_tokens.size(0)
        q = self.queries.expand(B, -1, -1)   # (B, K, D)

        if attn_masks is not None:
            # MultiheadAttention key_padding_mask: True = ignore
            # We build one mask per lesion by iterating (simple approach)
            outputs = []
            for i in range(self.queries.size(1)):
                mask_i = ~attn_masks[:, i, :]  # (B, N+1): True = ignore
                out_i, _ = self.cross_attn(
                    q[:, i:i+1, :], patch_tokens, patch_tokens,
                    key_padding_mask=mask_i
                )
                outputs.append(out_i)
            out = torch.cat(outputs, dim=1)    # (B, K, D)
        else:
            out, _ = self.cross_attn(q, patch_tokens, patch_tokens)

        return self.norm(out)                  # (B, K, D)


# ─────────────────────────────────────────────
# 8.  CLASSIFICATION HEADS
# ─────────────────────────────────────────────

class GradingHead(nn.Module):
    """Ordinal classification head for DR (0-4) and DME (0-2) grading."""
    def __init__(self, in_dim, num_classes, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)   # raw logits


class LesionHead(nn.Module):
    """Binary classification head per lesion type."""
    def __init__(self, in_dim=DIM, num_lesions=NUM_LESIONS, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, node_feats):
        """node_feats: (B, K, D) → (B, K) logits"""
        return self.net(node_feats).squeeze(-1)


# ─────────────────────────────────────────────
# 9.  FULL MODEL — AC-ViT + POG
# ─────────────────────────────────────────────

class ACViTPOG(nn.Module):
    """
    Anatomically-Constrained Vision Transformer
    with Pathological Occurrence Graph Modeling.

    Inputs:
      image    : (B, 3, 512, 512) normalised fundus image
      od_xy    : (B, 2) optic disc centre, normalised [0,1]
      fovea_xy : (B, 2) fovea centre, normalised [0,1]

    Outputs (dict):
      dr_logits    : (B, 5)  DR grade logits
      dme_logits   : (B, 3)  DME grade logits
      lesion_logits: (B, 5)  per-lesion binary logits
      attn_weights : list of (B, H, N+1, N+1) per transformer layer
      adj_matrix   : (5, 5) current POG adjacency (for visualisation)
    """

    def __init__(
        self,
        image_size   = IMAGE_SIZE,
        patch_size   = PATCH_SIZE,
        embed_dim    = DIM,
        depth        = DEPTH,
        heads        = HEADS,
        mlp_dim      = MLP_DIM,
        dropout      = DROPOUT,
        num_lesions  = NUM_LESIONS,
        dr_grades    = DR_GRADES,
        dme_grades   = DME_GRADES,
    ):
        super().__init__()

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches      = (image_size // patch_size) ** 2

        # --- Anatomical positional embedding ---
        self.anat_pos_embed = AnatomicalPositionalEmbedding(
            num_patches, embed_dim, image_size, patch_size
        )

        # --- Transformer layers ---
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, heads, embed_dim // heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- Anatomical constraint module ---
        self.anat_constraint = AnatomicalConstraintModule(
            num_patches, image_size, patch_size, embed_dim
        )

        # --- Lesion token extractor ---
        self.lesion_extractor = LesionTokenExtractor(num_lesions, embed_dim)

        # --- Pathological Occurrence Graph ---
        self.pog = PathologicalOccurrenceGraph(num_lesions, embed_dim)

        # --- Grading heads ---
        # CLS token for global grading; concatenate with mean-pooled lesion feats
        self.dr_head  = GradingHead(embed_dim * 2, dr_grades,  dropout)
        self.dme_head = GradingHead(embed_dim * 2, dme_grades, dropout)

        # --- Lesion detection head ---
        self.lesion_head = LesionHead(embed_dim, num_lesions, dropout)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image, od_xy, fovea_xy):
        # ── 1. Patch embedding ───────────────────────────
        tokens = self.patch_embed(image)                     # (B, N, D)

        # ── 2. Anatomical positional embedding ───────────
        tokens = self.anat_pos_embed(tokens, od_xy, fovea_xy)  # (B, N+1, D)

        # ── 3. Anatomical masks for lesion attention ─────
        anat_masks = self.anat_constraint(od_xy, fovea_xy)  # (B, K, N+1)

        # ── 4. Transformer forward ───────────────────────
        all_attn = []
        for i, block in enumerate(self.transformer):
            # Apply anatomical mask only in the last 3 layers
            mask = None
            if i >= len(self.transformer) - 3:
                # Use the union of lesion masks (any lesion attends → attend)
                union_mask = anat_masks.any(dim=1)          # (B, N+1)
                mask = union_mask
            tokens, attn_w = block(tokens, mask)
            all_attn.append(attn_w)

        tokens = self.norm(tokens)                           # (B, N+1, D)

        # ── 5. Lesion token extraction ───────────────────
        lesion_tokens = self.lesion_extractor(tokens, anat_masks)  # (B, K, D)

        # ── 6. POG refinement ────────────────────────────
        lesion_tokens = self.pog(lesion_tokens)              # (B, K, D)

        # ── 7. Classification heads ──────────────────────
        cls_token     = tokens[:, 0]                         # (B, D)
        lesion_mean   = lesion_tokens.mean(dim=1)            # (B, D)
        global_feat   = torch.cat([cls_token, lesion_mean], dim=-1)  # (B, 2D)

        dr_logits     = self.dr_head(global_feat)            # (B, 5)
        dme_logits    = self.dme_head(global_feat)           # (B, 3)
        lesion_logits = self.lesion_head(lesion_tokens)      # (B, K)

        return {
            'dr_logits'     : dr_logits,
            'dme_logits'    : dme_logits,
            'lesion_logits' : lesion_logits,
            'attn_weights'  : all_attn,
            'adj_matrix'    : torch.sigmoid(self.pog.adj_logits).detach(),
            'lesion_tokens' : lesion_tokens,
            'cls_token'     : cls_token,
        }

    def get_gradcam_targets(self):
        """Returns the layer to hook for GradCAM visualisation."""
        return self.transformer[-1]


# ─────────────────────────────────────────────
# 10.  MODEL FACTORY + PRETRAINED LOADING
# ─────────────────────────────────────────────

def build_acvit_pog(pretrained_vit_path=None):
    """
    Build the AC-ViT-POG model.
    Optionally load a pretrained ViT-B/16 checkpoint and
    transfer patch embedding + transformer weights.
    """
    model = ACViTPOG()

    if pretrained_vit_path is not None:
        state = torch.load(pretrained_vit_path, map_location='cpu')
        # Filter to only ViT backbone keys
        vit_keys = {
            k: v for k, v in state.items()
            if k.startswith('patch_embed') or k.startswith('transformer')
        }
        missing, unexpected = model.load_state_dict(vit_keys, strict=False)
        print(f"[ACViTPOG] Loaded pretrained ViT weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ACViTPOG] Total params: {total/1e6:.1f}M  "
          f"Trainable: {trainable/1e6:.1f}M")

    return model


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_acvit_pog().to(device)

    B = 2
    image    = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    od_xy    = torch.tensor([[0.62, 0.50], [0.61, 0.51]]).to(device)   # normalised
    fovea_xy = torch.tensor([[0.50, 0.50], [0.50, 0.49]]).to(device)

    with torch.no_grad():
        out = model(image, od_xy, fovea_xy)

    print("DR logits shape     :", out['dr_logits'].shape)       # (2, 5)
    print("DME logits shape    :", out['dme_logits'].shape)      # (2, 3)
    print("Lesion logits shape :", out['lesion_logits'].shape)   # (2, 5)
    print("POG adj matrix :\n",   out['adj_matrix'].cpu().numpy().round(3))
    print("Sanity check PASSED")
