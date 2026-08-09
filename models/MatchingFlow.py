"""
MatchingFlowTKG — Conditional Flow Matching for Temporal Knowledge Graphs
==========================================================================

v6 — Fourier Transformer context encoder

Core change: replaces GraphEmbedding's CNN context encoder with a
FourierTransformerEncoder that mixes information in the frequency domain.

Why Fourier mixing for TKGs:
- Entity/relation embeddings in temporal KGs encode periodic patterns
  (daily, monthly, yearly cycles). FFT naturally decomposes these cycles.
- FNet-style token mixing (Lee-Thorp et al. 2021) applies real FFT across
  the feature dimension, producing global feature interactions in O(d log d)
  vs O(d²) for standard attention.
- The temporal signal (year, month, day) is used to modulate the
  frequency-domain representation via learned FiLM-style conditioning,
  giving the encoder fine-grained temporal awareness.

Architecture:
  FourierTransformerEncoder(query_emb, time_emb):
    1. FFT mix: apply torch.fft.rfft along feature dim, keep real part
    2. FiLM condition: scale+shift by time_emb (learnable γ, β)
    3. Feed-forward: LayerNorm → Linear → GELU → Dropout → Linear
    4. Residual: output + original query_emb

Everything else is identical to v3 (the best-performing version):
- Dual encoders for entity/relation embeddings (GraphEmbedding tables only)
- Temporal encoding: NoName formula
- Difference-based scoring: softplus(dropout(ctx*x_hat).sum - ctx@all.T)
- Flow: detached x_hat, 10 ODE steps, CE + contrastive(×5) + FM(×0.1) + N3(×5)
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
    """
    Single-layer FNet-style encoder for a single embedding vector.

    Applies real FFT across the feature dimension, conditions on a temporal
    signal via FiLM (Feature-wise Linear Modulation), then refines with a
    feed-forward network.

    Input:
        query_emb : (bs, d_model)  — head + relation embedding
        time_emb  : (bs, d_model)  — sinusoidal temporal encoding
    Output:
        (bs, d_model)
    """

    def __init__(self, d_model: int, dropout: float, ff_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        # rfft of a real vector of length d produces d//2+1 complex values.
        # We take the real part → same dimension d (with the last dim behaviour
        # of torch.fft.rfft padded back via irfft or a projection).
        # Simpler: project rfft real+imag parts back to d_model.
        rfft_out_dim = (d_model // 2 + 1) * 2   # real + imag concatenated
        self.freq_proj = nn.Linear(rfft_out_dim, d_model, bias=False)

        # FiLM conditioning: time_emb → (gamma, beta) for post-FFT features
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta  = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1   = nn.Linear(d_model, d_model * ff_mult)
        self.ff2   = nn.Linear(d_model * ff_mult, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, query_emb, time_emb):
        # --- 1. Fourier mixing across feature dimension ---
        # rfft along last dim: (bs, d_model) → (bs, d_model//2+1) complex
        freq = torch.fft.rfft(query_emb, dim=-1)           # complex (bs, d//2+1)
        # Concatenate real and imaginary parts → (bs, (d//2+1)*2)
        freq_ri = torch.cat([freq.real, freq.imag], dim=-1)
        mixed   = self.freq_proj(freq_ri)                   # (bs, d_model)

        # --- 2. FiLM temporal conditioning ---
        gamma = self.film_gamma(time_emb)                   # (bs, d_model)
        beta  = self.film_beta(time_emb)                    # (bs, d_model)
        mixed = gamma * mixed + beta                        # scale + shift

        # --- 3. Residual + LayerNorm ---
        x = self.norm1(query_emb + mixed)

        # --- 4. Feed-forward ---
        ff_out = self.drop(F.gelu(self.ff1(x)))
        ff_out = self.ff2(ff_out)
        x = self.norm2(x + self.drop(ff_out))

        return x   # (bs, d_model)


# ---------------------------------------------------------------------------
# VectorField — same as v3
# ---------------------------------------------------------------------------

class VectorField(nn.Module):
    """
    2-layer MLP with skip connection predicting flow velocity.
    Input:  cat([x_t, time_emb, context_emb])  (bs, 3*d_model)
    Output: velocity                            (bs, d_model)
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1   = nn.Linear(3 * d_model, 2 * d_model)
        self.norm1 = nn.LayerNorm(2 * d_model)
        self.fc2   = nn.Linear(2 * d_model, d_model)
        self.skip  = nn.Linear(3 * d_model, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

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
    Conditional Flow Matching TKG model with Fourier Transformer context encoder.

    Context encoding pipeline:
        query_emb = head_emb ± rel_emb   (± per branch, same as NoName)
        time_emb  = sinusoidal(temporal_signal)
        context   = FourierTransformerEncoder(query_emb, time_emb)

    Everything downstream (scoring, flow, losses) is identical to v3.
    """

    def __init__(self, config, ode_steps: int = 10):
        super().__init__()

        self.n_ent        = config.n_ent
        self.n_rel        = config.n_rel
        self.d_model      = config.d_model
        self.dropout_rate = config.dropout
        self.ode_steps    = ode_steps

        # Entity / relation embedding tables (shared with scoring)
        self.encoder1 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )
        self.encoder2 = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate,
        )

        # Fourier Transformer context encoders (replace CNN from GraphEmbedding)
        self.fourier_enc1 = FourierTransformerEncoder(self.d_model, self.dropout_rate)
        self.fourier_enc2 = FourierTransformerEncoder(self.d_model, self.dropout_rate)

        # Flow vector fields
        self.vector_field1 = VectorField(self.d_model, self.dropout_rate)
        self.vector_field2 = VectorField(self.d_model, self.dropout_rate)

        # Regulariser & losses
        self.emb_regularizer = N3(0.004)
        self.lp_loss_fn      = nn.CrossEntropyLoss()

        # Learnable temporal frequency — same init as NoName / Tero
        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float(),
            requires_grad=True,
        )

        # Temporal MLP (feeds into FourierTransformerEncoder FiLM layers)
        self.temporal_mlp = nn.Linear(self.d_model, self.d_model)

        # Flow scalar time embedding MLP
        self.time_mlp = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(self.dropout_rate)

    # -----------------------------------------------------------------------
    # Temporal encoding
    # -----------------------------------------------------------------------

    def _temporal_encoding(self, year, month, day):
        """Returns d_real and d_img (bs, d_model), same formula as NoName."""
        month = month.float()
        day   = day.float()
        year  = year.float()
        safe_month  = month.clamp(min=1.0)
        time_signal = month + day % safe_month + year % safe_month
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))
        d_img  = torch.cos(self.w.view(1, -1) * time_signal.unsqueeze(1))
        return d_real, d_img

    def _temporal_emb(self, d_real):
        """Project temporal encoding through MLP for use as FiLM condition."""
        return self.temporal_mlp(d_real)   # (bs, d_model)

    # -----------------------------------------------------------------------
    # Context encoder — Fourier Transformer replaces CNN
    # -----------------------------------------------------------------------

    def _encode_context(self, heads, rels, year, month, day):
        """
        Returns:
            context_real, context_img  — (bs, d_model) Fourier-encoded context
            head_real, head_img        — (bs, d_model) raw head embeddings
            rel_real, rel_img          — (bs, d_model) temporally-rotated relations
        """
        d_real, d_img = self._temporal_encoding(year, month, day)
        temp_emb_real = self._temporal_emb(d_real)   # FiLM condition for enc1
        temp_emb_img  = self._temporal_emb(d_img)    # FiLM condition for enc2

        head_real = self.encoder1.get_ent_embedding(heads)
        head_img  = self.encoder2.get_ent_embedding(heads)

        # Temporal rotation of relations — identical to NoName.forward()
        r1 = self.encoder1.get_rel_embedding(rels)
        r2 = self.encoder2.get_rel_embedding(rels)
        rel_real = d_real * r1 - d_img * r2
        rel_img  = d_real * r2 + d_img * r1

        # Fourier Transformer context encoding
        # Real branch: query = head + rel, conditioned on d_real
        # Img  branch: query = head - rel, conditioned on d_img
        context_real = self.fourier_enc1(head_real + rel_real, temp_emb_real)
        context_img  = self.fourier_enc2(head_img  - rel_img,  temp_emb_img)

        return context_real, context_img, head_real, head_img, rel_real, rel_img

    # -----------------------------------------------------------------------
    # Euler ODE integration — identical to v3
    # -----------------------------------------------------------------------

    def _euler_integrate(self, context_emb, vector_field):
        bs  = context_emb.size(0)
        x_t = torch.randn_like(context_emb)
        dt  = 1.0 / self.ode_steps

        for i in range(self.ode_steps):
            t_val    = i * dt
            t_scalar = torch.full((bs,), t_val, device=context_emb.device)
            t_emb    = _sinusoidal_time_emb(t_scalar, self.d_model, self.time_mlp)
            v        = vector_field(x_t, t_emb, context_emb)
            x_t      = x_t + dt * v

        return x_t

    # -----------------------------------------------------------------------
    # Training — identical to v3
    # -----------------------------------------------------------------------

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        context_real, context_img, \
        head_real, head_img, \
        rel_real, rel_img = self._encode_context(heads, rels, year, month, day)

        bs = heads.size(0)

        tail_real = self.encoder1.get_ent_embedding(tails)
        tail_img  = self.encoder2.get_ent_embedding(tails)
        neg_real  = self.encoder1.get_ent_embedding(neg)
        neg_img   = self.encoder2.get_ent_embedding(neg)

        # Flow branch 1
        t1       = torch.rand(bs, device=tails.device)
        noise1   = torch.randn_like(tail_real)
        x_t1     = (1 - t1[:, None]) * noise1 + t1[:, None] * tail_real
        t_emb1   = _sinusoidal_time_emb(t1, self.d_model, self.time_mlp)
        v_pred1  = self.vector_field1(x_t1, t_emb1, context_real)
        fm_loss1 = F.mse_loss(v_pred1, tail_real - noise1)
        x_hat1   = (v_pred1 * (1 - t1[:, None]) + x_t1).detach()

        # Flow branch 2
        t2       = torch.rand(bs, device=tails.device)
        noise2   = torch.randn_like(tail_img)
        x_t2     = (1 - t2[:, None]) * noise2 + t2[:, None] * tail_img
        t_emb2   = _sinusoidal_time_emb(t2, self.d_model, self.time_mlp)
        v_pred2  = self.vector_field2(x_t2, t_emb2, context_img)
        fm_loss2 = F.mse_loss(v_pred2, tail_img - noise2)
        x_hat2   = (v_pred2 * (1 - t2[:, None]) + x_t2).detach()

        # Difference-based scores
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
            torch.ones_like(tails).unsqueeze(1),
            torch.zeros_like(neg),
        ], dim=1).float()

        lp_loss     = self.lp_loss_fn(type_intes, torch.zeros(bs, dtype=torch.long, device=tails.device))
        contra_loss = _contrastive_loss(type_intes, labels)
        fm_loss     = fm_loss1 + fm_loss2
        reg_loss    = self.emb_regularizer((head_real, head_img, rel_real, rel_img))

        return lp_loss + 5.0 * contra_loss + 0.1 * fm_loss + 5.0 * reg_loss

    # -----------------------------------------------------------------------
    # Evaluation — identical to v3
    # -----------------------------------------------------------------------

    def test_forward(self, heads, rels, tails, year, month, day):
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        context_real, context_img, _, _, _, _ = self._encode_context(
            heads, rels, year, month, day
        )

        x_hat1 = self._euler_integrate(context_real, self.vector_field1)
        x_hat2 = self._euler_integrate(context_img,  self.vector_field2)

        all_real = self.encoder1.get_all_ent_embedding()
        all_img  = self.encoder2.get_all_ent_embedding()

        scores = F.softplus(
            self.dropout(context_real.multiply(x_hat1)).sum(dim=-1, keepdim=True)
          - context_real.mm(all_real.t())
          + self.dropout(context_img.multiply(x_hat2)).sum(dim=-1, keepdim=True)
          - context_img.mm(all_img.t())
        )

        return scores
