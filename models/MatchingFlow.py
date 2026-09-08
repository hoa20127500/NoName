"""
MatchingFlowTKG — Conditional Flow Matching for Temporal Knowledge Graphs
==========================================================================

v7 — 1-step OT-CFM by default (train/test aligned)

Speed:
  OT-CFM has a constant target velocity, so one Euler step from t=0 is
  exact when the vector field is well trained. Default ode_steps=1.
  Dual-branch integration shares one time embedding per step.

Accuracy:
  Ranking loss now trains the vector field (x_hat is not detached).
  FM weight raised to 1.0. VectorField has two hidden layers.
  Raise --ode_steps (e.g. 4) for extra Euler refinement at test.

Loss: CE + contrastive(×5) + FM(×1.0) + N3(×5) + w_smooth(×0.01)
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
# FourierTransformerEncoder
# ---------------------------------------------------------------------------

class FourierTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float, ff_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        rfft_out_dim = (d_model // 2 + 1) * 2
        self.freq_proj = nn.Linear(rfft_out_dim, d_model, bias=False)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta  = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1   = nn.Linear(d_model, d_model * ff_mult)
        self.ff2   = nn.Linear(d_model * ff_mult, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, query_emb, time_emb):
        freq    = torch.fft.rfft(query_emb, dim=-1)
        freq_ri = torch.cat([freq.real, freq.imag], dim=-1)
        mixed   = self.freq_proj(freq_ri)
        gamma   = self.film_gamma(time_emb)
        beta    = self.film_beta(time_emb)
        mixed   = gamma * mixed + beta
        x = self.norm1(query_emb + mixed)
        ff_out = self.drop(F.gelu(self.ff1(x)))
        ff_out = self.ff2(ff_out)
        return self.norm2(x + self.drop(ff_out))


# ---------------------------------------------------------------------------
# VectorField
# ---------------------------------------------------------------------------

class VectorField(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1   = nn.Linear(3 * d_model, 2 * d_model)
        self.norm1 = nn.LayerNorm(2 * d_model)
        self.fc2   = nn.Linear(2 * d_model, 2 * d_model)
        self.norm2 = nn.LayerNorm(2 * d_model)
        self.fc3   = nn.Linear(2 * d_model, d_model)
        self.skip  = nn.Linear(3 * d_model, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x_t, time_emb, context_emb):
        h   = torch.cat([x_t, time_emb, context_emb], dim=-1)
        out = self.drop(F.gelu(self.norm1(self.fc1(h))))
        out = self.drop(F.gelu(self.norm2(self.fc2(out))))
        return self.fc3(out) + self.skip(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contrastive_loss(scores, labels, pos_margin=20.0, neg_margin=-20.0):
    loss_pos = labels       * torch.pow(F.relu(scores - pos_margin),  2)
    loss_neg = (1 - labels) * torch.pow(F.relu(neg_margin - scores), 2)
    return torch.mean(loss_pos + loss_neg)


# ---------------------------------------------------------------------------
# MatchingFlowTKG
# ---------------------------------------------------------------------------

class MatchingFlowTKG(nn.Module):

    def __init__(self, config, ode_steps: int = 1):
        super().__init__()

        self.n_ent        = config.n_ent
        self.n_rel        = config.n_rel
        self.d_model      = config.d_model
        self.dropout_rate = config.dropout
        self.ode_steps    = ode_steps

        self.encoder1 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )
        self.encoder2 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )
        # Alias used by older tests / callers that expect a single encoder.
        self.encoder = self.encoder1

        self.fourier_enc1 = FourierTransformerEncoder(self.d_model, self.dropout_rate)
        self.fourier_enc2 = FourierTransformerEncoder(self.d_model, self.dropout_rate)

        self.vector_field1 = VectorField(self.d_model, self.dropout_rate)
        self.vector_field2 = VectorField(self.d_model, self.dropout_rate)

        self.emb_regularizer = N3(0.004)
        self.lp_loss_fn      = nn.CrossEntropyLoss()

        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float(),
            requires_grad=True,
        )

        half = self.d_model // 2
        self.register_buffer(
            "time_freqs",
            torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=torch.float32) / half
            ),
            persistent=False,
        )

        self.temporal_mlp = nn.Linear(self.d_model, self.d_model)
        self.time_mlp     = nn.Linear(self.d_model, self.d_model)
        self.dropout      = nn.Dropout(self.dropout_rate)

    # -----------------------------------------------------------------------

    def _embed_flow_time(self, t_scalar):
        temp  = t_scalar[:, None].float() * self.time_freqs[None]
        t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.d_model % 2:
            t_emb = torch.cat(
                [t_emb, t_emb.new_zeros(t_emb.size(0), 1)], dim=-1
            )
        return self.time_mlp(t_emb)

    def _temporal_encoding(self, year, month, day):
        month = month.float()
        day   = day.float()
        year  = year.float()
        safe_month  = month.clamp(min=1.0)
        time_signal = month + day % safe_month + year % safe_month
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))
        d_img  = torch.cos(self.w.view(1, -1) * time_signal.unsqueeze(1))
        return d_real, d_img

    def _temporal_emb(self, d_real):
        return self.temporal_mlp(d_real)

    def _encode_context(self, heads, rels, year, month, day):
        d_real, d_img = self._temporal_encoding(year, month, day)
        temp_emb_real = self._temporal_emb(d_real)
        temp_emb_img  = self._temporal_emb(d_img)

        head_real = self.encoder1.get_ent_embedding(heads)
        head_img  = self.encoder2.get_ent_embedding(heads)

        r1 = self.encoder1.get_rel_embedding(rels)
        r2 = self.encoder2.get_rel_embedding(rels)
        rel_real = d_real * r1 - d_img * r2
        rel_img  = d_real * r2 + d_img * r1

        context_real = self.fourier_enc1(head_real + rel_real, temp_emb_real)
        context_img  = self.fourier_enc2(head_img  - rel_img,  temp_emb_img)

        return context_real, context_img, head_real, head_img, rel_real, rel_img

    def _euler_integrate_pair(self, context_real, context_img, x1=None, x2=None):
        """Integrate both branches in one loop, sharing time embeddings."""
        bs = context_real.size(0)
        if x1 is None:
            x1 = torch.randn_like(context_real)
        if x2 is None:
            x2 = torch.randn_like(context_img)
        dt = 1.0 / self.ode_steps
        t_grid = torch.arange(
            self.ode_steps, device=context_real.device, dtype=torch.float32
        ) * dt
        t_embs = self._embed_flow_time(t_grid)
        for i in range(self.ode_steps):
            t_emb = t_embs[i].expand(bs, -1)
            x1 = x1 + dt * self.vector_field1(x1, t_emb, context_real)
            x2 = x2 + dt * self.vector_field2(x2, t_emb, context_img)
        return x1, x2

    # -----------------------------------------------------------------------

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        context_real, context_img, \
        head_real, head_img, \
        rel_real, rel_img = self._encode_context(heads, rels, year, month, day)

        bs     = heads.size(0)
        device = tails.device

        tail_real = self.encoder1.get_ent_embedding(tails)
        tail_img  = self.encoder2.get_ent_embedding(tails)
        neg_real  = self.encoder1.get_ent_embedding(neg)
        neg_img  = self.encoder2.get_ent_embedding(neg)

        noise1 = torch.randn_like(tail_real)
        noise2 = torch.randn_like(tail_img)

        if self.ode_steps == 1:
            # Train at t=0 so FM, x_hat, and 1-step Euler test are the same op.
            t_emb0  = self._embed_flow_time(torch.zeros(bs, device=device))
            v_pred1 = self.vector_field1(noise1, t_emb0, context_real)
            v_pred2 = self.vector_field2(noise2, t_emb0, context_img)
            x_hat1  = noise1 + v_pred1
            x_hat2  = noise2 + v_pred2
        else:
            t     = torch.rand(bs, device=device)
            t_emb = self._embed_flow_time(t)
            x_t1  = (1 - t[:, None]) * noise1 + t[:, None] * tail_real
            x_t2  = (1 - t[:, None]) * noise2 + t[:, None] * tail_img
            v_pred1 = self.vector_field1(x_t1, t_emb, context_real)
            v_pred2 = self.vector_field2(x_t2, t_emb, context_img)
            x_hat1  = x_t1 + (1 - t[:, None]) * v_pred1
            x_hat2  = x_t2 + (1 - t[:, None]) * v_pred2

        fm_loss = (
            F.mse_loss(v_pred1, tail_real - noise1)
            + F.mse_loss(v_pred2, tail_img - noise2)
        )

        ent_embs1 = torch.cat([tail_real.unsqueeze(1), neg_real], dim=1)
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
        )

        labels = torch.cat([
            torch.ones(bs, 1, device=device),
            torch.zeros(bs, neg.size(1), device=device),
        ], dim=1)

        lp_loss     = self.lp_loss_fn(type_intes, torch.zeros(bs, dtype=torch.long, device=device))
        contra_loss = _contrastive_loss(type_intes, labels)
        reg_loss    = self.emb_regularizer((head_real, head_img, rel_real, rel_img))
        w_smooth    = ((self.w[1:] - self.w[:-1]) ** 2).mean()

        return (lp_loss
                + 5.0 * contra_loss
                + 1.0 * fm_loss
                + 5.0 * reg_loss
                + 0.01 * w_smooth)

    # -----------------------------------------------------------------------

    def test_forward(self, heads, rels, tails, year, month, day):
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        context_real, context_img, _, _, _, _ = self._encode_context(
            heads, rels, year, month, day
        )

        x_hat1, x_hat2 = self._euler_integrate_pair(context_real, context_img)

        all_real = self.encoder1.get_all_ent_embedding()
        all_img  = self.encoder2.get_all_ent_embedding()

        scores = F.softplus(
            self.dropout(context_real.multiply(x_hat1)).sum(dim=-1, keepdim=True)
          - context_real.mm(all_real.t())
          + self.dropout(context_img.multiply(x_hat2)).sum(dim=-1, keepdim=True)
          - context_img.mm(all_img.t())
        )

        return scores
