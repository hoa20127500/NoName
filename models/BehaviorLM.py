"""Pretrained LLM as a behavior encoder for TKG extrapolation.

History facts of the query actor are written as an event trace. A frozen
instruction LM (default Qwen2.5-0.5B-Instruct) encodes that trace; a small
trainable head ranks objects against LM name vectors plus a learned offset.

This is not frozen name-table init (which underperforms scratch). The LM reads
behavior over time, which is how recent TKG-LLM work (ICL / GenTKG) uses
pretrained models for forecasting.
"""
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import _last_token_pool, _readable_name, load_names
from model import LabelSmoothingCrossEntropy

DEFAULT_LM = 'Qwen/Qwen2.5-0.5B-Instruct'


class BehaviorLM(nn.Module):
    def __init__(self, n_ent, n_rel, data_dir, lm_model=DEFAULT_LM, max_length=256,
                 dropout=0.2, eps=0.2, freeze_lm=True):
        super(BehaviorLM, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.max_length = max_length
        self.freeze_lm = freeze_lm
        self.ent_names, self.rel_names = _load_surface_names(data_dir, n_ent, n_rel)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        logging.info('BehaviorLM backbone %s on %s (%s)', lm_model, device, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_model, padding_side='left', trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_model, trust_remote_code=True, torch_dtype=dtype)
        self.lm.to(device)
        self.backbone = getattr(self.lm, 'model', None) or getattr(self.lm, 'transformer')
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False
            self.lm.eval()

        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.ent_delta = nn.Embedding(n_ent, hidden)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.ent_delta.weight)
        self.dropout = nn.Dropout(dropout)
        self.lp_loss_fn = LabelSmoothingCrossEntropy(eps)

        cache_path = os.path.join(
            data_dir, 'cache', '{}_behavior_names.pt'.format(lm_model.replace('/', '_')))
        name_h = _try_load_name_table(cache_path, lm_model, n_ent, hidden)
        if name_h is None:
            name_h = _encode_name_table(
                self.backbone, self.tokenizer, self.ent_names, max_length, device)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({'model': lm_model, 'ent': name_h.cpu()}, cache_path)
            logging.info('Wrote BehaviorLM name cache %s', cache_path)
        else:
            logging.info('Loaded BehaviorLM name cache %s', cache_path)
            name_h = name_h.to(device)
        self.register_buffer('ent_name_h', name_h)

    def train(self, mode=True):
        super(BehaviorLM, self).train(mode)
        if self.freeze_lm:
            self.lm.eval()
        return self

    def _prompts(self, sub, rel, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask):
        texts = []
        batch, max_e = ev_src.size()
        sub = sub.tolist()
        rel = rel.tolist()
        time = time.tolist()
        src = ev_src.tolist()
        rids = ev_rel.tolist()
        dst = ev_dst.tolist()
        ts = ev_time.tolist()
        mask = ev_mask.tolist()
        for i in range(batch):
            actor = self.ent_names[sub[i]]
            lines = ['Actor: {}'.format(actor), 'Past events:']
            saw = False
            for j in range(max_e):
                if not mask[i][j]:
                    continue
                saw = True
                lines.append('[{}] {} -- {} --> {}'.format(
                    ts[i][j],
                    self.ent_names[src[i][j]],
                    self.rel_names[rids[i][j]],
                    self.ent_names[dst[i][j]]))
            if not saw:
                lines.append('(none)')
            lines.append('Predict object of: {} -- {} --> ? at [{}]'.format(
                actor, self.rel_names[rel[i]], time[i]))
            texts.append('\n'.join(lines))
        return texts

    def encode_behavior(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        device = self.ent_name_h.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if self.freeze_lm:
            with torch.no_grad():
                hidden = self.backbone(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    use_cache=False,
                ).last_hidden_state
        else:
            hidden = self.backbone(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                use_cache=False,
            ).last_hidden_state
        pooled = _last_token_pool(hidden.float(), encoded['attention_mask'])
        return self.proj(self.dropout(pooled))

    def scores(self, query_h):
        objects = F.normalize(self.ent_name_h + self.ent_delta.weight, p=2, dim=-1)
        query_h = F.normalize(query_h, p=2, dim=-1)
        return query_h.matmul(objects.transpose(0, 1))

    def forward(self, s_ent, relation, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask):
        texts = self._prompts(s_ent, relation, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask)
        return self.scores(self.encode_behavior(texts))

    def train_forward(self, s_ent, relation, o_ent, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask):
        logits = self.forward(s_ent, relation, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask)
        loss_lp = self.lp_loss_fn(logits, o_ent)
        loss_tp = logits.new_zeros(())
        return loss_lp, loss_tp

    def test_forward(self, s_ent, relation, o_ent, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask,
                     local_weight=1.):
        logits = self.forward(s_ent, relation, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask)
        zeros = logits.new_zeros(logits.size(0))
        return logits, zeros, zeros


def _load_surface_names(data_dir, n_ent, n_rel):
    n_fwd = n_rel // 2
    ent = [_readable_name(n) for n in load_names(os.path.join(data_dir, 'entity2id.txt'), n_ent - 1)]
    ent.append('PAD')
    rel_raw = load_names(os.path.join(data_dir, 'relation2id.txt'), n_fwd)
    rel = [_readable_name(n) for n in rel_raw] + [
        'inverse {}'.format(_readable_name(n)) for n in rel_raw]
    if len(ent) != n_ent:
        ent.extend(['id {}'.format(i) for i in range(len(ent), n_ent)])
        ent = ent[:n_ent]
    if len(rel) != n_rel:
        rel.extend(['rel {}'.format(i) for i in range(len(rel), n_rel)])
        rel = rel[:n_rel]
    return ent, rel


def _encode_name_table(backbone, tokenizer, names, max_length, device):
    hidden_size = int(backbone.config.hidden_size)
    pieces = []
    batch_size = 32
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            chunk = names[start:start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=min(max_length, 32),
                return_tensors='pt',
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = backbone(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                use_cache=False,
            ).last_hidden_state
            pooled = _last_token_pool(hidden.float(), encoded['attention_mask'])
            pieces.append(F.normalize(pooled, p=2, dim=-1).cpu())
            if start == 0:
                logging.info('Encoding entity names with BehaviorLM (%d)', len(names))
    if was_training:
        backbone.train()
    table = torch.cat(pieces, dim=0)
    if table.size(1) != hidden_size:
        raise RuntimeError('name table dim {} != hidden {}'.format(table.size(1), hidden_size))
    table[-1].zero_()
    return table.to(device)


def _try_load_name_table(path, model_name, n_ent, hidden):
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    if payload.get('model') != model_name:
        return None
    ent = payload.get('ent')
    if ent is None or ent.size(0) != n_ent or ent.size(1) != hidden:
        return None
    return ent
