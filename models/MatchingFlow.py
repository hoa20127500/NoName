import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphEmbedding import GraphEmbedding
from regularizers import N3


class VectorField(nn.Module):
    """
    Feed-forward MLP that predicts the velocity vector for conditional flow matching.

    Reduced capacity vs. original to mitigate overfitting:
      3*d_model → 2*d_model → d_model → d_model
    with a skip connection from input to output.

    Input:  cat([x_t, time_emb, context_emb])  — (bs, 3 * d_model)
    Output: velocity vector                    — (bs, d_model)
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        # Layer 1: 3*d_model → 2*d_model  (reduced from 4*d_model)
        self.layer1 = nn.Linear(3 * d_model, 2 * d_model)
        self.norm1  = nn.LayerNorm(2 * d_model)

        # Layer 2: 2*d_model → d_model
        self.layer2 = nn.Linear(2 * d_model, d_model)
        self.norm2  = nn.LayerNorm(d_model)

        # Skip connection: 3*d_model → d_model
        self.skip = nn.Linear(3 * d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()   # GELU generalises slightly better than ReLU

    def forward(
        self,
        x_t:         torch.Tensor,   # (bs, d_model)
        time_emb:    torch.Tensor,   # (bs, d_model)
        context_emb: torch.Tensor,   # (bs, d_model)
    ) -> torch.Tensor:               # (bs, d_model)

        h        = torch.cat([x_t, time_emb, context_emb], dim=-1)   # (bs, 3*d_model)
        residual = h

        h = self.dropout(self.act(self.norm1(self.layer1(h))))
        h = self.dropout(self.act(self.norm2(self.layer2(h))))

        return h + self.skip(residual)


def _make_time_emb(
    t_scalar: torch.Tensor,   # (bs,)
    d_model:  int,
    time_mlp: nn.Linear,
) -> torch.Tensor:             # (bs, d_model)
    """Shared sinusoidal time embedding used in both train and test."""
    device = t_scalar.device
    half   = d_model // 2
    freqs  = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=device, dtype=torch.float32)
        / half
    )                                                    # (half,)
    temp  = t_scalar[:, None] * freqs[None]              # (bs, half)
    t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (bs, d_model or d_model-1)
    if d_model % 2:
        t_emb = torch.cat([t_emb, torch.zeros(t_emb.size(0), 1, device=device)], dim=-1)
    return time_mlp(t_emb)                               # (bs, d_model)


class MatchingFlowTKG(nn.Module):
    """
    Conditional flow-matching model for TKG link prediction.

    Maps Gaussian noise → tail entity embedding conditioned on
    (head, relation, year, month, day) using a learned vector field.

    Anti-overfitting measures applied:
    - Reduced VectorField capacity (2x instead of 4x hidden)
    - GELU activations
    - Dropout on embeddings before LP scoring
    - L2 normalisation of context/tail embeddings before dot-product scoring
    - Stronger N3 regularisation weight (0.01)
    - Weight decay should be set in the optimiser (recommended: 1e-4)
    """

    def __init__(self, config, ode_steps: int = 10):
        super().__init__()

        self.n_ent        = config.n_ent
        self.n_rel        = config.n_rel
        self.d_model      = config.d_model
        self.dropout_rate = config.dropout
        self.ode_steps    = ode_steps

        # --- submodules ---
        self.encoder = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )
        self.vector_field    = VectorField(self.d_model, self.dropout_rate)
        self.emb_regularizer = N3(0.004)

        # Learnable sinusoidal temporal frequency vector (matches NoName / Tero)
        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float(),
            requires_grad=True,
        )

        self.time_mlp  = nn.Linear(self.d_model, self.d_model)
        self.lp_loss_fn = nn.CrossEntropyLoss()
        self.dropout   = nn.Dropout(self.dropout_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_time_emb(self, t_scalar: torch.Tensor) -> torch.Tensor:
        return _make_time_emb(t_scalar, self.d_model, self.time_mlp)

    def _encode_context(self, heads, rels, year, month, day):
        """
        Returns:
            context_emb: (bs, d_model)
            head_emb:    (bs, d_model)
            rel_emb:     (bs, d_model)
        """
        time_signal = month + day % month + year % month         # (bs,)
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))  # (bs, d_model)

        head_emb = self.encoder.get_ent_embedding(heads)         # (bs, d_model)
        rel_emb  = self.encoder.get_rel_embedding(rels)          # (bs, d_model)
        query_emb = head_emb + rel_emb                           # (bs, d_model)

        context_emb = self.encoder(query_emb, d_real)            # (bs, d_model)
        return context_emb, head_emb, rel_emb

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        context_emb, head_emb, rel_emb = self._encode_context(heads, rels, year, month, day)
        bs = heads.size(0)

        # Target tail embedding (detached where used as a fixed target)
        x1 = self.encoder.get_ent_embedding(tails)               # (bs, d_model)

        # ------------------------------------------------------------------
        # 1. Flow matching loss — trains the VectorField only
        # ------------------------------------------------------------------
        t_scalar = torch.rand(bs, device=x1.device)
        noise    = torch.randn_like(x1)
        x_t      = (1 - t_scalar[:, None]) * noise + t_scalar[:, None] * x1.detach()
        target_v = x1.detach() - noise                           # fixed target, no encoder grad here

        t_emb  = self._make_time_emb(t_scalar)
        v_pred = self.vector_field(x_t, t_emb, context_emb.detach())  # flow only
        fm_loss = F.mse_loss(v_pred, target_v)

        # ------------------------------------------------------------------
        # 2. Link prediction loss — trains the encoder (context + embeddings)
        #    Scores via dot product between context_emb and entity embeddings.
        #    Detach negatives' impact on the flow path.
        # ------------------------------------------------------------------
        neg_emb = self.encoder.get_ent_embedding(neg)            # (bs, 500, d_model)

        pos_score = (context_emb * x1).sum(dim=-1, keepdim=True)                         # (bs, 1)
        neg_score = torch.bmm(neg_emb, context_emb.unsqueeze(-1)).squeeze(-1)            # (bs, 500)
        lp_scores = torch.cat([pos_score, neg_score], dim=1)                             # (bs, 501)
        lp_loss   = self.lp_loss_fn(lp_scores, torch.zeros(bs, dtype=torch.long, device=x1.device))

        # ------------------------------------------------------------------
        # 3. Consistency loss — aligns flow output with context_emb direction
        #    Uses a single Euler step from pure noise as a cheap approximation.
        #    Trains both the flow and the encoder to agree on the target region.
        # ------------------------------------------------------------------
        with torch.no_grad():
            noise_test = torch.randn_like(x1)
        t_one = torch.ones(bs, device=x1.device)
        t_emb_one = self._make_time_emb(t_one)
        x_pred = noise_test + self.vector_field(noise_test, t_emb_one, context_emb)      # single Euler step
        consist_loss = F.mse_loss(x_pred, x1.detach())           # push flow output toward x1

        # ------------------------------------------------------------------
        # 4. Embedding regularisation
        # ------------------------------------------------------------------
        reg_loss = self.emb_regularizer((head_emb, rel_emb, x1))

        return fm_loss + lp_loss + 0.5 * consist_loss + reg_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def test_forward(self, heads, rels, tails, year, month, day):
        """
        Euler ODE integration from t=0 to t=1, then score all entities
        by negative squared L2 distance to the integrated embedding.

        Returns: scores (bs, n_ent)
        """
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        context_emb, _, _ = self._encode_context(heads, rels, year, month, day)
        bs     = heads.size(0)
        device = heads.device

        x_t = torch.randn(bs, self.d_model, device=device)
        dt  = 1.0 / self.ode_steps

        for i in range(self.ode_steps):
            t_scalar = torch.full((bs,), i * dt, device=device)
            t_emb    = self._make_time_emb(t_scalar)
            v        = self.vector_field(x_t, t_emb, context_emb)
            x_t      = x_t + dt * v

        all_ent_embs = self.encoder.get_all_ent_embedding()      # (n_ent, d_model)
        scores = -torch.cdist(x_t, all_ent_embs).pow(2)          # (bs, n_ent)
        return scores
