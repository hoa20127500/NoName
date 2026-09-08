"""
MatchingFlow — Conditional Flow Matching for temporal KG completion.

Drop-in model for the DE-SimplE trainer/tester: constructed as
MatchingFlow(dataset, params) and called as
forward(heads, rels, tails, years, months, days).

Train: parses de-simple pos+neg groups (size 1+neg_ratio), runs the
generative objective on tail-prediction groups, returns a scalar loss.

Test: returns one score per triple so tester.py ranking is unchanged.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierTransformerEncoder(nn.Module):
    def __init__(self, d_model, dropout, ff_mult=2):
        super(FourierTransformerEncoder, self).__init__()
        rfft_out_dim = (d_model // 2 + 1) * 2
        self.freq_proj = nn.Linear(rfft_out_dim, d_model, bias=False)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * ff_mult)
        self.ff2 = nn.Linear(d_model * ff_mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query_emb, time_emb):
        freq = torch.fft.rfft(query_emb, dim=-1)
        mixed = self.freq_proj(torch.cat([freq.real, freq.imag], dim=-1))
        mixed = self.film_gamma(time_emb) * mixed + self.film_beta(time_emb)
        x = self.norm1(query_emb + mixed)
        ff_out = self.ff2(self.drop(F.gelu(self.ff1(x))))
        return self.norm2(x + self.drop(ff_out))


class VectorField(nn.Module):
    def __init__(self, d_model, dropout):
        super(VectorField, self).__init__()
        self.fc1 = nn.Linear(3 * d_model, 2 * d_model)
        self.norm1 = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, 2 * d_model)
        self.norm2 = nn.LayerNorm(2 * d_model)
        self.fc3 = nn.Linear(2 * d_model, d_model)
        self.skip = nn.Linear(3 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_t, time_emb, context_emb):
        h = torch.cat([x_t, time_emb, context_emb], dim=-1)
        out = self.drop(F.gelu(self.norm1(self.fc1(h))))
        out = self.drop(F.gelu(self.norm2(self.fc2(out))))
        return self.fc3(out) + self.skip(h)


def _contrastive_loss(scores, labels, pos_margin=20.0, neg_margin=-20.0):
    loss_pos = labels * torch.pow(F.relu(scores - pos_margin), 2)
    loss_neg = (1 - labels) * torch.pow(F.relu(neg_margin - scores), 2)
    return torch.mean(loss_pos + loss_neg)


class MatchingFlow(nn.Module):
    def __init__(self, dataset, params):
        super(MatchingFlow, self).__init__()
        self.dataset = dataset
        self.params = params

        self.n_ent = dataset.numEnt()
        self.n_rel = dataset.numRel()
        self.d_model = params.s_emb_dim + params.t_emb_dim
        self.dropout_rate = params.dropout
        self.ode_steps = getattr(params, "ode_steps", 1)
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")

        self.ent_embs_real = nn.Embedding(self.n_ent, self.d_model)
        self.ent_embs_img = nn.Embedding(self.n_ent, self.d_model)
        self.rel_embs_real = nn.Embedding(self.n_rel, self.d_model)
        self.rel_embs_img = nn.Embedding(self.n_rel, self.d_model)

        nn.init.xavier_uniform_(self.ent_embs_real.weight)
        nn.init.xavier_uniform_(self.ent_embs_img.weight)
        nn.init.xavier_uniform_(self.rel_embs_real.weight)
        nn.init.xavier_uniform_(self.rel_embs_img.weight)

        self.fourier_enc1 = FourierTransformerEncoder(self.d_model, self.dropout_rate)
        self.fourier_enc2 = FourierTransformerEncoder(self.d_model, self.dropout_rate)
        self.vector_field1 = VectorField(self.d_model, self.dropout_rate)
        self.vector_field2 = VectorField(self.d_model, self.dropout_rate)

        self.w = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model)).float()
        )
        half = self.d_model // 2
        self.register_buffer(
            "time_freqs",
            torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=torch.float32) / max(half, 1)
            ),
            persistent=False,
        )
        self.temporal_mlp = nn.Linear(self.d_model, self.d_model)
        self.time_mlp = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lp_loss_fn = nn.CrossEntropyLoss()
        self.n3_weight = 0.004

    def forward(self, heads, rels, tails, years, months, days):
        if self.training:
            return self._train_loss(heads, rels, tails, years, months, days)
        return self._score_triples(heads, rels, tails, years, months, days)

    def _embed_flow_time(self, t_scalar):
        temp = t_scalar[:, None].float() * self.time_freqs[None]
        t_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.d_model % 2:
            t_emb = torch.cat([t_emb, t_emb.new_zeros(t_emb.size(0), 1)], dim=-1)
        return self.time_mlp(t_emb)

    def _temporal_encoding(self, year, month, day):
        month = month.float()
        day = day.float()
        year = year.float()
        safe_month = month.clamp(min=1.0)
        time_signal = month + day % safe_month + year % safe_month
        d_real = torch.sin(self.w.view(1, -1) * time_signal.unsqueeze(1))
        d_img = torch.cos(self.w.view(1, -1) * time_signal.unsqueeze(1))
        return d_real, d_img

    def _encode_context(self, heads, rels, years, months, days):
        d_real, d_img = self._temporal_encoding(years, months, days)
        temp_emb_real = self.temporal_mlp(d_real)
        temp_emb_img = self.temporal_mlp(d_img)

        head_real = self.ent_embs_real(heads)
        head_img = self.ent_embs_img(heads)
        r1 = self.rel_embs_real(rels)
        r2 = self.rel_embs_img(rels)
        rel_real = d_real * r1 - d_img * r2
        rel_img = d_real * r2 + d_img * r1

        context_real = self.fourier_enc1(head_real + rel_real, temp_emb_real)
        context_img = self.fourier_enc2(head_img - rel_img, temp_emb_img)
        return context_real, context_img, head_real, head_img, rel_real, rel_img

    def _euler_integrate_pair(self, context_real, context_img, x1=None, x2=None):
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

    def _pair_scores(self, context_real, context_img, x_hat1, x_hat2, tails):
        tail_real = self.ent_embs_real(tails)
        tail_img = self.ent_embs_img(tails)
        return (
            self.dropout(context_real * x_hat1 - context_real * tail_real).sum(dim=-1)
            + self.dropout(context_img * x_hat2 - context_img * tail_img).sum(dim=-1)
        )

    def _score_triples(self, heads, rels, tails, years, months, days):
        same_query = (
            torch.equal(heads, heads[0].expand_as(heads))
            and torch.equal(rels, rels[0].expand_as(rels))
            and torch.equal(years, years[0].expand_as(years))
            and torch.equal(months, months[0].expand_as(months))
            and torch.equal(days, days[0].expand_as(days))
        )
        if same_query:
            ctx_r, ctx_i, _, _, _, _ = self._encode_context(
                heads[:1], rels[:1], years[:1], months[:1], days[:1]
            )
            x1, x2 = self._euler_integrate_pair(ctx_r, ctx_i)
            ctx_r = ctx_r.expand(heads.size(0), -1)
            ctx_i = ctx_i.expand(heads.size(0), -1)
            x1 = x1.expand(heads.size(0), -1)
            x2 = x2.expand(heads.size(0), -1)
            return F.softplus(self._pair_scores(ctx_r, ctx_i, x1, x2, tails))

        ctx_r, ctx_i, _, _, _, _ = self._encode_context(
            heads, rels, years, months, days
        )
        x1, x2 = self._euler_integrate_pair(ctx_r, ctx_i)
        return F.softplus(self._pair_scores(ctx_r, ctx_i, x1, x2, tails))

    def _train_loss(self, heads, rels, tails, years, months, days):
        group = 1 + self.params.neg_ratio
        if heads.size(0) % group != 0:
            raise ValueError(
                "MatchingFlow expected a batch divisible by 1+neg_ratio, got %d"
                % heads.size(0)
            )

        n_groups = heads.size(0) // group
        h = heads.view(n_groups, group)
        r = rels.view(n_groups, group)
        t = tails.view(n_groups, group)
        y = years.view(n_groups, group)
        m = months.view(n_groups, group)
        d = days.view(n_groups, group)

        tail_pred = (h == h[:, :1]).all(dim=1)
        if not tail_pred.any():
            raise ValueError("MatchingFlow training batch has no tail-prediction groups")

        h = h[tail_pred, 0]
        r = r[tail_pred, 0]
        t_pos = t[tail_pred, 0]
        t_neg = t[tail_pred, 1:]
        y = y[tail_pred, 0]
        m = m[tail_pred, 0]
        d = d[tail_pred, 0]

        ctx_r, ctx_i, head_r, head_i, rel_r, rel_i = self._encode_context(
            h, r, y, m, d
        )
        tail_r = self.ent_embs_real(t_pos)
        tail_i = self.ent_embs_img(t_pos)
        noise_r = torch.randn_like(tail_r)
        noise_i = torch.randn_like(tail_i)
        bs = h.size(0)
        device = h.device

        if self.ode_steps == 1:
            t_emb0 = self._embed_flow_time(torch.zeros(bs, device=device))
            v_r = self.vector_field1(noise_r, t_emb0, ctx_r)
            v_i = self.vector_field2(noise_i, t_emb0, ctx_i)
            x_hat_r = noise_r + v_r
            x_hat_i = noise_i + v_i
        else:
            t_rand = torch.rand(bs, device=device)
            t_emb = self._embed_flow_time(t_rand)
            x_t_r = (1 - t_rand[:, None]) * noise_r + t_rand[:, None] * tail_r
            x_t_i = (1 - t_rand[:, None]) * noise_i + t_rand[:, None] * tail_i
            v_r = self.vector_field1(x_t_r, t_emb, ctx_r)
            v_i = self.vector_field2(x_t_i, t_emb, ctx_i)
            x_hat_r = x_t_r + (1 - t_rand[:, None]) * v_r
            x_hat_i = x_t_i + (1 - t_rand[:, None]) * v_i

        fm_loss = F.mse_loss(v_r, tail_r - noise_r) + F.mse_loss(v_i, tail_i - noise_i)

        cand = torch.cat([t_pos.unsqueeze(1), t_neg], dim=1)
        cand_r = self.ent_embs_real(cand)
        cand_i = self.ent_embs_img(cand)
        type_intes = (
            self.dropout(
                ctx_r.mul(x_hat_r).unsqueeze(1) - ctx_r.unsqueeze(1).mul(cand_r)
            ).sum(dim=-1)
            + self.dropout(
                ctx_i.mul(x_hat_i).unsqueeze(1) - ctx_i.unsqueeze(1).mul(cand_i)
            ).sum(dim=-1)
        )

        labels = torch.cat(
            [
                torch.ones(bs, 1, device=device),
                torch.zeros(bs, t_neg.size(1), device=device),
            ],
            dim=1,
        )
        lp_loss = self.lp_loss_fn(
            type_intes, torch.zeros(bs, dtype=torch.long, device=device)
        )
        contra_loss = _contrastive_loss(type_intes, labels)
        n3 = self.n3_weight * (
            head_r.abs().pow(3).sum()
            + head_i.abs().pow(3).sum()
            + rel_r.abs().pow(3).sum()
            + rel_i.abs().pow(3).sum()
        ) / bs
        w_smooth = ((self.w[1:] - self.w[:-1]) ** 2).mean()

        return (
            lp_loss
            + 5.0 * contra_loss
            + 1.0 * fm_loss
            + 5.0 * n3
            + 0.01 * w_smooth
        )
