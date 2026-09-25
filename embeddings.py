"""Encode entity/relation names with a small open embedding model.

Default is ``Qwen/Qwen3-Embedding-0.6B``: a dedicated embedding checkpoint
(~0.6B, ~1GB in fp16) that fits Google Colab Free (T4 or CPU). Do not swap in a
chat LLM such as Qwen2.5-Instruct — those are not embedding models, and the 4B/8B
Qwen3-Embedding variants are likely to OOM next to this TKG model on a free T4.

The frozen vectors are projected to ``d_model`` inside ``ProjectedEmbedding``.
"""
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_WIKIDATA_ID = re.compile(r'^Q\d+$')
DEFAULT_EMB_MODEL = 'Qwen/Qwen3-Embedding-0.6B'


class ProjectedEmbedding(nn.Module):
    """Lookup table with an optional linear map down to ``d_model``."""

    def __init__(self, num_embeddings, d_model, pretrained=None, freeze=False):
        super(ProjectedEmbedding, self).__init__()
        if pretrained is None:
            self.source = nn.Embedding(num_embeddings, d_model)
            nn.init.xavier_uniform_(self.source.weight)
            self.proj = nn.Identity()
            return

        weight = torch.as_tensor(pretrained, dtype=torch.float32)
        if weight.dim() != 2 or weight.size(0) != num_embeddings:
            raise ValueError(
                'pretrained shape {} does not match num_embeddings={}'.format(
                    tuple(weight.shape), num_embeddings))
        self.source = nn.Embedding.from_pretrained(weight, freeze=freeze)
        if weight.size(1) == d_model:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(weight.size(1), d_model, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, index):
        return self.proj(self.source(index))

    @property
    def weight(self):
        return self.proj(self.source.weight)


def load_pretrained_tables(data_dir, num_e, num_r, model_name=DEFAULT_EMB_MODEL,
                           batch_size=16, max_length=128, cache_path=None):
    """Return (ent [num_e+1, d], rel [2*num_r, d]) including PAD and inverse rels."""
    cache_path = cache_path or _default_cache_path(data_dir, model_name)
    cached = _try_load_cache(cache_path, num_e, num_r, model_name)
    if cached is None:
        ent_names = load_names(os.path.join(data_dir, 'entity2id.txt'), num_e)
        rel_names = load_names(os.path.join(data_dir, 'relation2id.txt'), num_r)
        texts = (
            [format_entity_text(n) for n in ent_names]
            + [format_relation_text(n, inverse=False) for n in rel_names]
            + [format_relation_text(n, inverse=True) for n in rel_names]
        )
        logging.info('Encoding %d names with %s', len(texts), model_name)
        vectors = encode_texts(texts, model_name, batch_size, max_length)
        ent = vectors[:num_e]
        rel_fwd = vectors[num_e:num_e + num_r]
        rel_inv = vectors[num_e + num_r:]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'model': model_name,
            'ent': ent.cpu(),
            'rel_fwd': rel_fwd.cpu(),
            'rel_inv': rel_inv.cpu(),
        }, cache_path)
        logging.info('Wrote embedding cache %s (%d-d)', cache_path, vectors.size(1))
    else:
        ent, rel_fwd, rel_inv = cached
        logging.info('Loaded embedding cache %s', cache_path)

    pad = torch.zeros(1, ent.size(1), dtype=ent.dtype)
    return torch.cat([ent, pad], dim=0), torch.cat([rel_fwd, rel_inv], dim=0)


def pca_reduce_tables(ent, rel, d_out):
    """Project stacked entity/relation vectors to ``d_out`` with PCA (SVD).

    The PAD row (last entity) stays zeros. ``d_model`` can stay 100 while Qwen
    is 1024; this is a fixed reduction, not a learned Linear.
    """
    d_in = ent.size(1)
    if d_out == d_in:
        return ent, rel
    if d_out > d_in:
        raise ValueError('d_model={} is larger than pretrained dim {}'.format(d_out, d_in))
    body = torch.cat([ent[:-1], rel], dim=0)
    mean = body.mean(dim=0, keepdim=True)
    centered = body - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    reduced = F.normalize(centered.matmul(vh[:d_out].T), p=2, dim=-1)
    n_body_ent = ent.size(0) - 1
    ent_reduced = torch.cat([reduced[:n_body_ent], torch.zeros(1, d_out, dtype=ent.dtype)], dim=0)
    rel_reduced = reduced[n_body_ent:]
    return ent_reduced, rel_reduced


def load_names(path, count):
    names = ['id {}'.format(i) for i in range(count)]
    if not os.path.isfile(path):
        logging.warning('Missing %s; using fallback names', path)
        return names
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            if 0 <= idx < count:
                names[idx] = parts[0]
    return names


def format_entity_text(raw):
    name = _readable_name(raw)
    if _WIKIDATA_ID.match(raw.strip()):
        return 'Wikidata item {}'.format(raw.strip())
    return 'Knowledge graph entity: {}'.format(name)


def format_relation_text(raw, inverse=False):
    name = _readable_name(raw)
    if inverse:
        return 'Knowledge graph inverse relation: {}'.format(name)
    return 'Knowledge graph relation: {}'.format(name)


def encode_texts(texts, model_name, batch_size, max_length):
    from transformers import AutoModel, AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    logging.info('Embedding encoder on %s (%s)', device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    pieces = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pooled = _last_token_pool(hidden, encoded['attention_mask'])
            pieces.append(F.normalize(pooled.float(), p=2, dim=-1).cpu())
            if (start // batch_size) % 20 == 0:
                logging.info('Encoded %d / %d names', min(start + batch_size, len(texts)), len(texts))

    del model, tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(pieces, dim=0)


def _readable_name(raw):
    name = raw.strip().strip('<>').replace('_', ' ')
    return name or 'unknown'


def _default_cache_path(data_dir, model_name):
    safe = model_name.replace('/', '_').replace(':', '_')
    return os.path.join(data_dir, 'cache', '{}.pt'.format(safe))


def _try_load_cache(path, num_e, num_r, model_name):
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    if payload.get('model') != model_name:
        return None
    ent, rel_fwd, rel_inv = payload['ent'], payload['rel_fwd'], payload['rel_inv']
    if ent.size(0) != num_e or rel_fwd.size(0) != num_r or rel_inv.size(0) != num_r:
        logging.warning('Embedding cache size mismatch; recomputing')
        return None
    return ent, rel_fwd, rel_inv


def _last_token_pool(last_hidden_state, attention_mask):
    """Qwen3-Embedding (and other causal models) read the last non-pad token."""
    if attention_mask[:, -1].min() == 1:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    batch = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch, lengths]
