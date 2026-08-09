"""
MatchingFlowTKG — Conditional Flow Matching for Temporal Knowledge Graphs
==========================================================================

v3 — correctly mirrors NoName's scoring function at test time.

The critical insight from NoName's test_forward:
    scores = softplus(
        dropout(context * x_denoised).sum(-1, keepdim=True)  -- per-query alignment
      - context @ all_ent.T                                   -- global contrast
    )
This difference-based score is what drives accuracy. Previous versions used
plain dot-product context @ all_ent.T, which is a fundamentally different
(weaker) scoring function.

Here, the flow's denoised output x_hat plays exactly the role that x_t1/x_t2
play in NoName — the per-query target estimate that sharpens discrimination.

Architecture
------------
- Dual complex-valued encoders (real + imaginary), identical to NoName.
- Temporal encoding: same formula as NoName (month + day%month + year%month).
- Relation rotation: TComplEx-style (d_real*r1 - d_img*r2).
- Flow: Euler ODE integration from noise → x_hat at test time.
- Scoring: softplus(dropout(context*x_hat).sum - context@all_ent.T) × 2 branches.
- Training loss: CE on difference scores + contrastive(×5) + FM(×0.1) + N3(×5).
"""

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


# ---------------------------------------------------------------------------
# VectorField
# ---------------------------------------------------------------------------

class VectorField(nn.Module):
    """
    2-layer MLP with skip connection predicting flow velocity.
    Input:  cat([x_t, time_emb, context_emb])  (bs, 3*d_model)
    Output: velocity                            (bs, d_model)
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1  = nn.Linear(3 * d_model, 2 * d_model)
        self.norm1 = nn.LayerNorm(2 * d_model)
        self.fc2  = nn.Linear(2 * d_model, d_model)
        self.skip = nn.Linear(3 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t, time_emb, context_emb):
        h   = torch.cat([x_t, time_emb, context_emb], dim=-1)
        out = self.drop(F.gelu(self.norm1(self.fc1(h))))
        out = self.fc2(out)
        return out + self.skip(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sinusoidal_time_emb(t_scalar, d_model, time_mlp):
    """Sinusoidal embedding of scalar t, projected through time_mlp."""
    device = t_scalar.device
    half   = d_model // 2
    freqs  = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    temp  = t_scalar[:, None].float() * freqs[None]
    t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
    if d_model % 2:
        t_emb = torch.cat([t_emb, torch.zeros(t_emb.size(0), 1, device=device)], dim=-1)
    return time_mlp(t_emb)


def _contrastive_loss(scores, labels, pos_margin=20.0, neg_margin=-20.0):
    """Margin-based contrastive loss — identical to NoName's contrastive_loss."""
    loss_pos = labels       * torch.pow(F.relu(scores - pos_margin),  2)
    loss_neg = (1 - labels) * torch.pow(F.relu(neg_margin - scores), 2)
    return torch.mean(loss_pos + loss_neg)


# ---------------------------------------------------------------------------
# MatchingFlowTKG
# ---------------------------------------------------------------------------

class MatchingFlowTKG(nn.Module):
    """
    Scoring at test time (mirrors NoName exactly):

        x_hat1, x_hat2 = Euler ODE integration of flow from noise

        scores = F.softplus(
            dropout(context1 * x_hat1).sum(-1, keepdim=True) - context1 @ all_real.T
          + dropout(context2 * x_hat2).sum(-1, keepdim=True) - context2 @ all_img.T
        )

    The per-query denoised estimate (x_hat) acts as a personalized anchor that
    makes the difference-based score more discriminative than plain dot-product.
    """

    def __init__(self, config, ode_steps: int = 10):
        super().__init__()

        self.n_ent        = config.n_ent
        self.n_rel        = config.n_rel
        self.d_model      = config.d_model
        self.dropout_rate = config.dropout
        self.ode_steps    = ode_steps

        # Dual encoders — same as NoName
        self.encoder1 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )
        self.encoder2 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )

        # Flow vector fields (one per encoder branch)
        self.vector_field1 = VectorField(self.d_model, self.dropout_rate)
        self.vector_field2 = VectorField(self.d_model, self.dropout_rate)

        # Regulariser & losses
        self.emb_regularizer = N3(0.004)
        self.lp_loss_fn      = nn.CrossEntropyLoss()

        # Learnable temporal frequency — same initialisation as NoName / Tero
        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float(),
            requires_grad=True,
        )

        # Time MLP for flow scalar embedding
        self.time_mlp = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(self.dropout_rate)

    # -----------------------------------------------------------------------
    # Temporal encoding — matches NoName train_forward exactly
    # -----------------------------------------------------------------------

    def _temporal_encoding(self, year, month, day):
        """
        Returns d_real (sin) and d_img (cos), each (bs, d_model).
        Uses the same time_signal formula as NoName.
        """
        month = month.float()
        day   = day.float()
        year  = year.float()
        safe_month  = month.clamp(min=1.0)
        time_signal = month + day % safe_month + year % safe_month
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))
        d_img  = torch.cos(self.w.view(1, -1) * time_signal.unsqueeze(1))
        return d_real, d_img

    # -----------------------------------------------------------------------
    # Context encoder — mirrors NoName.train_forward
    # -----------------------------------------------------------------------

    def _encode_context(self, heads, rels, year, month, day):
        """
        Returns:
            context_real, context_img  — (bs, d_model) CNN-encoded conditions
            head_real, head_img        — (bs, d_model) raw head embeddings
            rel_real, rel_img          — (bs, d_model) temporally-rotated relations
        """
        d_real, d_img = self._temporal_encoding(year, month, day)

        head_real = self.encoder1.get_ent_embedding(heads)
        head_img  = self.encoder2.get_ent_embedding(heads)

        # Temporal rotation of relations — identical to NoName.forward()
        r1 = self.encoder1.get_rel_embedding(rels)
        r2 = self.encoder2.get_rel_embedding(rels)
        rel_real = d_real * r1 - d_img * r2
        rel_img  = d_real * r2 + d_img * r1

        # CNN encoder: real branch uses (head+rel, d_real), img uses (head-rel, d_img)
        context_real = self.encoder1(head_real + rel_real, d_real)
        context_img  = self.encoder2(head_img  - rel_img,  d_img)

        return context_real, context_img, head_real, head_img, rel_real, rel_img

    # -----------------------------------------------------------------------
    # Euler ODE integration (flow → denoised estimate)
    # -----------------------------------------------------------------------

    def _euler_integrate(self, context_emb, vector_field):
        """
        Integrate vector field from t=0 (noise) to t=1 (entity space).
        Returns x_hat of shape (bs, d_model).
        """
        bs    = context_emb.size(0)
        x_t   = torch.randn_like(context_emb)
        dt    = 1.0 / self.ode_steps
        steps = torch.linspace(0.0, 1.0 - dt, self.ode_steps, device=context_emb.device)

        for t_val in steps:
            t_scalar = torch.full((bs,), t_val.item(), device=context_emb.device)
            t_emb    = _sinusoidal_time_emb(t_scalar, self.d_model, self.time_mlp)
            v        = vector_field(x_t, t_emb, context_emb)
            x_t      = x_t + dt * v

        return x_t   # (bs, d_model)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        context_real, context_img, \
        head_real, head_img, \
        rel_real, rel_img = self._encode_context(heads, rels, year, month, day)

        bs = heads.size(0)

        # Tail and negative embeddings
        tail_real = self.encoder1.get_ent_embedding(tails)    # (bs, d_model)
        tail_img  = self.encoder2.get_ent_embedding(tails)    # (bs, d_model)
        neg_real  = self.encoder1.get_ent_embedding(neg)      # (bs, num_neg, d_model)
        neg_img   = self.encoder2.get_ent_embedding(neg)      # (bs, num_neg, d_model)

        # --- Flow matching losses ---
        # Branch 1: flow grounds context_real in encoder1's entity space
        t1       = torch.rand(bs, device=tails.device)
        noise1   = torch.randn_like(tail_real)
        x_t1     = (1 - t1[:, None]) * noise1 + t1[:, None] * tail_real
        t_emb1   = _sinusoidal_time_emb(t1, self.d_model, self.time_mlp)
        v_pred1  = self.vector_field1(x_t1, t_emb1, context_real)
        fm_loss1 = F.mse_loss(v_pred1, tail_real - noise1)

        # Branch 2: flow grounds context_img in encoder2's entity space
        t2       = torch.rand(bs, device=tails.device)
        noise2   = torch.randn_like(tail_img)
        x_t2     = (1 - t2[:, None]) * noise2 + t2[:, None] * tail_img
        t_emb2   = _sinusoidal_time_emb(t2, self.d_model, self.time_mlp)
        v_pred2  = self.vector_field2(x_t2, t_emb2, context_img)
        fm_loss2 = F.mse_loss(v_pred2, tail_img - noise2)

        # Use the denoised estimates as per-query anchors for scoring
        # (same role as x_t1/x_t2 in NoName's train_forward after q_sample)
        # Detach so FM and LP gradients don't interfere
        x_hat1 = v_pred1.detach() * (1 - t1[:, None]) + x_t1.detach()
        x_hat2 = v_pred2.detach() * (1 - t2[:, None]) + x_t2.detach()

        # --- Difference-based scores (mirrors NoName's type_intes formula) ---
        # pos: context * tail  vs  context * x_hat  for each candidate
        ent_embs1 = torch.cat([tail_real.unsqueeze(1), neg_real], dim=1)  # (bs, 1+neg, d)
        ent_embs2 = torch.cat([tail_img.unsqueeze(1),  neg_img],  dim=1)

        type_intes = (
            self.dropout(
                context_real.multiply(x_hat1).unsqueeze(1)
              - context_real.unsqueeze(1).multiply(ent_embs1)
            ).sum(dim=-1)
          + self.dropout(
                context_img.multiply(x_hat2).unsqueeze(1)
              - context_img.unsqueeze(1).multiply(ent_embs2)
            ).sum(dim=-1)
        )  # (bs, 1+num_neg)

        labels = torch.cat([
            torch.ones_like(tails).unsqueeze(1),
            torch.zeros_like(neg),
        ], dim=1).float()  # (bs, 1+num_neg)

        # 1. CE loss
        lp_loss = self.lp_loss_fn(
            type_intes,
            torch.zeros(bs, dtype=torch.long, device=tails.device),
        )

        # 2. Contrastive loss — same weight as NoName (×5)
        contra_loss = _contrastive_loss(type_intes, labels)

        # 3. Flow matching (×0.1 — auxiliary)
        fm_loss = fm_loss1 + fm_loss2

        # 4. N3 regularisation — same 4 factors as NoName (×5)
        reg_loss = self.emb_regularizer((head_real, head_img, rel_real, rel_img))

        return lp_loss + 5.0 * contra_loss + 0.1 * fm_loss + 5.0 * reg_loss

    # -----------------------------------------------------------------------
    # Evaluation — mirrors NoName's test_forward scoring exactly
    # -----------------------------------------------------------------------

    def test_forward(self, heads, rels, tails, year, month, day):
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        context_real, context_img, _, _, _, _ = self._encode_context(
            heads, rels, year, month, day
        )

        # Euler integrate to get per-query denoised estimates
        x_hat1 = self._euler_integrate(context_real, self.vector_field1)  # (bs, d_model)
        x_hat2 = self._euler_integrate(context_img,  self.vector_field2)  # (bs, d_model)

        all_real = self.encoder1.get_all_ent_embedding()   # (n_ent, d_model)
        all_img  = self.encoder2.get_all_ent_embedding()   # (n_ent, d_model)

        # Difference-based scoring — identical structure to NoName.test_forward:
        #   softplus( dropout(context*x_hat).sum - context @ all_ent.T )
        scores = F.softplus(
            self.dropout(context_real.multiply(x_hat1)).sum(dim=-1, keepdim=True)
          - context_real.mm(all_real.t())
          + self.dropout(context_img.multiply(x_hat2)).sum(dim=-1, keepdim=True)
          - context_img.mm(all_img.t())
        )  # (bs, n_ent)

        return scores
