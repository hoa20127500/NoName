import math
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

    Input: concatenation of [x_t, time_emb, context_emb] → shape (batch_size, 3 * d_model)
    Output: velocity vector of shape (batch_size, d_model)
    """

    def __init__(self, d_model: int, dropout: float):
        super(VectorField, self).__init__()

        # Layer 1: 3*d_model → 4*d_model
        self.layer1 = nn.Linear(3 * d_model, 4 * d_model)
        self.norm1 = nn.LayerNorm(4 * d_model)

        # Layer 2: 4*d_model → 2*d_model
        self.layer2 = nn.Linear(4 * d_model, 2 * d_model)
        self.norm2 = nn.LayerNorm(2 * d_model)

        # Layer 3: 2*d_model → d_model
        self.layer3 = nn.Linear(2 * d_model, d_model)

        # Skip connection: 3*d_model → d_model
        self.skip = nn.Linear(3 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_t: torch.Tensor, time_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:          (bs, d_model) — noised sample at flow time t
            time_emb:     (bs, d_model) — scalar flow-time embedding
            context_emb:  (bs, d_model) — conditioning context vector

        Returns:
            velocity:     (bs, d_model) — predicted velocity vector
        """
        # Concatenate all inputs: (bs, 3 * d_model)
        h = torch.cat([x_t, time_emb, context_emb], dim=-1)

        # Keep residual input for skip connection
        residual = h

        # Layer 1: Linear → LayerNorm → ReLU → Dropout
        h = self.layer1(h)
        h = self.norm1(h)
        h = self.relu(h)
        h = self.dropout(h)

        # Layer 2: Linear → LayerNorm → ReLU → Dropout
        h = self.layer2(h)
        h = self.norm2(h)
        h = self.relu(h)
        h = self.dropout(h)

        # Layer 3: Linear (no activation — raw velocity output)
        h = self.layer3(h)

        # Skip connection added to final output
        h = h + self.skip(residual)

        return h


import numpy as np


class MatchingFlowTKG(nn.Module):
    """
    Flow-matching-based temporal knowledge graph link prediction model.

    Uses conditional flow matching to learn a mapping from Gaussian noise to
    tail entity embeddings, conditioned on (head, relation, year, month, day).
    """

    def __init__(self, config, ode_steps: int = 10):
        super(MatchingFlowTKG, self).__init__()

        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout
        self.ode_steps = ode_steps

        # Context encoder: entity/relation embeddings + CNN
        self.encoder = GraphEmbedding(
            self.n_ent, self.n_rel, self.d_model,
            self.dropout_rate, self.dropout_rate, self.dropout_rate
        )

        # Flow-matching velocity field
        self.vector_field = VectorField(self.d_model, self.dropout_rate)

        # Cubic regularizer on embedding factors
        self.emb_regularizer = N3(0.004)

        # Learnable sinusoidal temporal frequency vector
        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float(),
            requires_grad=True
        )

        # Projects sinusoidal time scalar embedding to d_model
        self.time_mlp = nn.Linear(self.d_model, self.d_model)

        # Cross-entropy loss for link prediction
        self.lp_loss_fn = nn.CrossEntropyLoss()

        # Dropout applied before scoring
        self.dropout = nn.Dropout(self.dropout_rate)

    def _encode_context(self, heads, rels, year, month, day):
        """
        Encode the query context (head, relation, time) into a conditioning vector.

        Args:
            heads:  (bs,) integer entity IDs
            rels:   (bs,) integer relation IDs
            year:   (bs,) temporal year component
            month:  (bs,) temporal month component
            day:    (bs,) temporal day component

        Returns:
            context_emb: (bs, d_model) conditioning vector
            head_emb:    (bs, d_model) head entity embedding
            rel_emb:     (bs, d_model) relation embedding
        """
        # Temporal encoding: scalar time signal per sample
        time_signal = month + day % month + year % month  # (bs,)

        # Sinusoidal temporal embeddings
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))  # (bs, d_model)
        d_img = torch.cos(self.w.view(1, -1) * time_signal.unsqueeze(1))   # (bs, d_model) — kept for design parity

        # Head entity and relation embeddings
        head_emb = self.encoder.get_ent_embedding(heads)   # (bs, d_model)
        rel_emb = self.encoder.get_rel_embedding(rels)     # (bs, d_model)

        # Additive query embedding
        query_emb = head_emb + rel_emb                     # (bs, d_model)

        # CNN-based context encoding (uses d_real as temporal input)
        context_emb = self.encoder(query_emb, d_real)      # (bs, d_model)

        return context_emb, head_emb, rel_emb

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        # Step 1: Encode context
        context_emb, head_emb, rel_emb = self._encode_context(heads, rels, year, month, day)

        bs = heads.size(0)

        # Step 2: Retrieve tail embedding (target)
        x1 = self.encoder.get_ent_embedding(tails)  # (bs, d_model)

        # Step 3: Sample scalar flow time t uniformly from [0, 1]
        t_scalar = torch.rand(bs, device=x1.device, dtype=torch.float32)  # (bs,)

        # Step 4: Sample Gaussian noise
        noise = torch.randn_like(x1)  # (bs, d_model)

        # Step 5: Compute interpolated sample x_t = (1-t)*noise + t*x1
        x_t = (1 - t_scalar.unsqueeze(1)) * noise + t_scalar.unsqueeze(1) * x1  # (bs, d_model)

        # Step 6: Target velocity = x1 - noise
        target_v = x1 - noise  # (bs, d_model)

        # Step 7: Compute scalar flow-time embedding via sinusoidal encoding
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.d_model // 2, device=x1.device).float()
            / (self.d_model // 2)
        )  # (d_model//2,)
        temp = t_scalar[:, None] * freqs[None]  # (bs, d_model//2)
        t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (bs, d_model) or (bs, d_model-1) if odd
        if self.d_model % 2:
            t_emb = torch.cat([t_emb, torch.zeros_like(t_emb[:, :1])], dim=-1)
        t_emb = self.time_mlp(t_emb)  # (bs, d_model)

        # Step 8: Predict velocity from vector field
        v_pred = self.vector_field(x_t, t_emb, context_emb)  # (bs, d_model)

        # Step 9: Flow matching loss (MSE between predicted and target velocity)
        fm_loss = F.mse_loss(v_pred, target_v)

        # Step 10: Retrieve negative embeddings
        neg_emb = self.encoder.get_ent_embedding(neg)  # (bs, 500, d_model)

        # Step 11: Compute link prediction scores
        pos_score = (context_emb * x1).sum(dim=-1, keepdim=True)          # (bs, 1)
        neg_score = torch.bmm(neg_emb, context_emb.unsqueeze(-1)).squeeze(-1)  # (bs, 500)
        lp_scores = torch.cat([pos_score, neg_score], dim=1)               # (bs, 501)

        # Step 12: Link prediction loss (cross-entropy, positive is class 0)
        lp_loss = self.lp_loss_fn(lp_scores, torch.zeros(bs, dtype=torch.long, device=x1.device))

        # Step 13: N3 regularization on embedding factors
        reg_loss = self.emb_regularizer.forward((head_emb, rel_emb, x1))

        # Step 14: Return combined loss
        return fm_loss + lp_loss + reg_loss

    def test_forward(self, heads, rels, tails, year, month, day):
        """
        Evaluate by integrating the vector field from t=0 to t=1 using a fixed-step Euler ODE solver,
        then scoring all candidate tail entities by negative squared L2 distance.

        Args:
            heads:  (bs,) integer head entity IDs
            rels:   (bs,) integer relation IDs
            tails:  (bs,) integer tail entity IDs (unused at test time, kept for interface compatibility)
            year:   (bs,) temporal year component
            month:  (bs,) temporal month component
            day:    (bs,) temporal day component

        Returns:
            scores: (bs, n_ent) negative squared L2 distances to each entity embedding
        """
        # Guard: ode_steps must be at least 1
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        # Step 1: Encode context conditioning vector
        context_emb, _, _ = self._encode_context(heads, rels, year, month, day)

        # Step 2: Batch size and device
        bs = heads.size(0)
        device = heads.device

        # Step 3: Initialize x_t from standard Gaussian noise
        x_t = torch.randn(bs, self.d_model, device=device)

        # Step 4: Euler integration from t=0 to t=1
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            # Scalar flow time for this step
            t_scalar = torch.full((bs,), i * dt, device=device, dtype=torch.float32)  # (bs,)

            # Sinusoidal time embedding (identical scheme to train_forward)
            freqs = torch.exp(
                -math.log(10000)
                * torch.arange(0, self.d_model // 2, device=device).float()
                / (self.d_model // 2)
            )  # (d_model//2,)
            temp = t_scalar[:, None] * freqs[None]  # (bs, d_model//2)
            t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)  # (bs, d_model) or (bs, d_model-1) if odd
            if self.d_model % 2:
                t_emb = torch.cat([t_emb, torch.zeros_like(t_emb[:, :1])], dim=-1)
            t_emb = self.time_mlp(t_emb)  # (bs, d_model)

            # Predict velocity and take Euler step
            v = self.vector_field(x_t, t_emb, context_emb)  # (bs, d_model)
            x_t = x_t + dt * v  # (bs, d_model)

        # Step 5: Retrieve all entity embeddings
        all_ent_embs = self.encoder.get_all_ent_embedding()  # (n_ent, d_model)

        # Step 6: Compute scores as negative squared L2 distance
        scores = -torch.cdist(x_t, all_ent_embs).pow(2)  # (bs, n_ent)

        return scores
