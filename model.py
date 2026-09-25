import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GraphEncoder import RGTEncoder, RGCNEncoder
from models.SequenceEncoder import TransformerEncoder
from models.ConvTransE import ConvTransE
from embeddings import ProjectedEmbedding
from utils import scatter_mean

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config, eps=0.2, time_span=24, timestep=0.1, hmax=5,
                 ent_pretrained=None, rel_pretrained=None, freeze_pretrained=True):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout
        self.transformer_layer_num = config.seqTransformerLayerNum
        self.transformer_head_num = config.seqTransformerHeadNum
        self.PAD_TIME = -1
        self.PAD_ENTITY = self.n_ent - 1
        self.time_span = time_span
        self.timestep = timestep
        self.hmax = hmax

        if ent_pretrained is None:
            self.ent_embeds = nn.Embedding(self.n_ent, self.d_model)
            nn.init.xavier_uniform_(self.ent_embeds.weight)
        else:
            self.ent_embeds = ProjectedEmbedding(
                self.n_ent, self.d_model, pretrained=ent_pretrained, freeze=freeze_pretrained)

        if rel_pretrained is None:
            self.rel_embeds = nn.Embedding(self.n_rel, self.d_model)
            nn.init.xavier_uniform_(self.rel_embeds.weight)
        else:
            self.rel_embeds = ProjectedEmbedding(
                self.n_rel, self.d_model, pretrained=rel_pretrained, freeze=freeze_pretrained)
        self.graph_encoder = RGTEncoder(self.d_model, self.dropout_rate)
        # self.graph_encoder = RGCNEncoder(self.d_model, self.n_rel, self.d_model//2, self.dropout_rate)
        self.seq_encoder = TransformerEncoder(self.d_model, self.d_model, self.transformer_layer_num,
                                              self.transformer_head_num, self.dropout_rate)

        self.linear_inten_layer = nn.Linear(self.d_model * 2, self.d_model, bias=False)
        self.time_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        self.linear_inten_layer1 = nn.Linear(self.d_model * 2, self.d_model, bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.Softplus = nn.Softplus(beta=0.5)
        self.lp_loss_fn = LabelSmoothingCrossEntropy(eps)

        self.conv2 = torch.nn.Conv1d(2, 50, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(50)
        self.fc2 = torch.nn.Linear(self.d_model * 50, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.ent_decoder = ConvTransE(self.n_ent,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)

        nn.init.xavier_uniform_(self.linear_inten_layer.weight)
        nn.init.xavier_uniform_(self.time_inten_layer.weight)
        nn.init.xavier_uniform_(self.linear_inten_layer1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.tp_loss_fn = nn.MSELoss()

    def forward(self, query_entities, query_relations, history_times,
                node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local):
        bs, hist_len, n_nodes = node_ids.size()
        n_edges = edge_src.size(-1)
        h = self.ent_embeds(node_ids)
        e_h = self.rel_embeds(edge_type)
        qeh = self.ent_embeds(query_entities).view(bs, 1, 1, -1).expand(bs, hist_len, n_edges, -1)
        qrh = self.rel_embeds(query_relations).view(bs, 1, 1, -1).expand(bs, hist_len, n_edges, -1)
        bl = bs * hist_len
        h = h.reshape(bl, n_nodes, self.d_model)
        src = edge_src.reshape(bl, n_edges)
        dst = edge_dst.reshape(bl, n_edges)
        mask = edge_mask.reshape(bl, n_edges)
        e_h = e_h.reshape(bl, n_edges, self.d_model)
        qeh = qeh.reshape(bl, n_edges, self.d_model)
        qrh = qrh.reshape(bl, n_edges, self.d_model)
        offsets = torch.arange(bl, device=h.device, dtype=torch.long).unsqueeze(1) * n_nodes
        valid = mask.reshape(-1)
        src_f = (src + offsets).reshape(-1)[valid]
        dst_f = (dst + offsets).reshape(-1)[valid]
        h_f = h.reshape(bl * n_nodes, self.d_model)
        e_f = e_h.reshape(bl * n_edges, self.d_model)[valid]
        qeh_f = qeh.reshape(bl * n_edges, self.d_model)[valid]
        qrh_f = qrh.reshape(bl * n_edges, self.d_model)[valid]
        total_nodes_h = self.graph_encoder.forward_flat(h_f, src_f, dst_f, e_f, qrh_f, qeh_f)
        total_nodes_h = total_nodes_h.view(bl, n_nodes, self.d_model)
        roots = root_local.reshape(bl)
        history_gh = total_nodes_h[torch.arange(bl, device=h.device), roots].view(bs, hist_len, self.d_model)
        query_rel_embeds = self.rel_embeds(query_relations)
        query_ent_embeds = self.ent_embeds(query_entities)
        history_pad_mask = (history_times == -1).unsqueeze(1)
        local_type = node_ids.reshape(bs, -1)
        total_nodes_h = total_nodes_h.reshape(bs, hist_len * n_nodes, self.d_model)
        return query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type

    def link_prediction(self, query_time, query_ent_embeds, query_rel_embeds,
                        history_gh, history_times, history_pad_mask, total_nodes_h, local_type):
        bs, hist_len = history_times.size(0), history_times.size(1)
        #seq_query_input = query_rel_embeds.unsqueeze(1)
        seq_query_time = query_time.view(-1, 1)  # [bs, 1]
        res_history_gh = self.dropout(torch.cat([query_rel_embeds.unsqueeze(1).repeat(1,hist_len,1),history_gh],dim=-1))
        res_history_gh = self.dropout(self.conv2(res_history_gh.reshape(bs,2,-1)))
        res_history_gh = self.bn2(res_history_gh)
        res_history_gh = res_history_gh.view(bs,hist_len, -1)
        history_gh = self.dropout(self.fc2(res_history_gh)) + history_gh
        history_gh = self.layer_norm(history_gh)


        seq_query_input = query_rel_embeds.unsqueeze(1)

        output = self.seq_encoder(history_gh, history_times, seq_query_input, seq_query_time, history_pad_mask)
        output = output[:, -1, :]

        statics_ent_embeds = self.dropout(self.linear_inten_layer(
            torch.cat((query_ent_embeds, query_rel_embeds), dim=-1)))
        inten_raw = F.gelu(self.dropout(self.linear_inten_layer1(
            torch.cat((statics_ent_embeds, output), dim=-1)))) +statics_ent_embeds
        inten_raw = self.layer_norm(inten_raw)
        #global_intes = inten_raw.mm(self.ent_embeds.weight.transpose(0, 1))  # [bs, ent_num]
        global_type = torch.arange(self.n_ent, device=output.device).unsqueeze(0).repeat(bs, 1)
        global_intes = self.ent_decoder(query_ent_embeds,query_rel_embeds, output,self.ent_embeds.weight)

        local_h = total_nodes_h.reshape([bs, -1, self.d_model])

        local_intes = torch.matmul(statics_ent_embeds.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]\
                + torch.matmul(inten_raw.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]  # [bs, max_nodes_num * seq_len]

        intens = self.Softplus(torch.cat([global_intes, local_intes], dim=-1))


        ent_type = torch.cat([global_type, local_type], dim=-1)
        return intens, ent_type,statics_ent_embeds


    def link_prediction_loss(self, intens, type, answers):
        intens = scatter_mean(intens, type, dim=-1)
        loss = self.lp_loss_fn(intens[:, :-1], answers)
        return loss

    def time_prediction_loss(self, estimate_dt, dur_last):
        loss_dt = self.tp_loss_fn(estimate_dt, dur_last)
        return loss_dt

    def ents_score(self, intens, type, local_weight=1.):
        #intens = F.softmax(intens, dim=-1)
        #intens[:, self.n_ent:] = intens[:, self.n_ent:] * 0.65
        output = scatter_mean(intens, type, dim=-1)
        return output[:, :-1]

    def predict_t(self, tail_ent, dur_last, query_ent_embeds, query_rel_embeds, history_gh, history_times, history_pad_mask,statics_ent_embeds):
        n_samples = int(self.hmax / self.timestep) + 1  # add 1 to accomodate zero
        bs, hist_len = history_times.size(0), history_times.size(1)
        dur_last = dur_last // self.time_span
        dur_non_zero_idx = (dur_last > 0).nonzero().squeeze(1)
        dur_last = dur_last[dur_non_zero_idx].type(torch.float)
        if not dur_last.numel():
            return torch.tensor([0.]), torch.tensor([0.])
        dt = torch.linspace(0, self.hmax, n_samples, device=dur_last.device).repeat(dur_last.shape[0], 1)  # [bs, n_sample]

        seq_query_input = query_rel_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1)  # [bs , n_sample, d_model]
        seq_query_time = history_times[dur_non_zero_idx, -1].unsqueeze(1).repeat(1, n_samples) + dt  # [bs, n_sample]
        sampled_seq_output = self.seq_encoder(history_gh[dur_non_zero_idx], history_times[dur_non_zero_idx],
                                              seq_query_input, seq_query_time, history_pad_mask[dur_non_zero_idx])  # [bs, n_sample, d_model]

        inten_layer_input = torch.cat((query_ent_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1),
                                    sampled_seq_output, query_rel_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1)), dim=-1)
        inten_raw = self.time_inten_layer(self.dropout(inten_layer_input))  # [bs, n_sample, d_model]
        o = self.ent_embeds(tail_ent[dur_non_zero_idx]).unsqueeze(1).repeat(1, n_samples, 1)  # [bs, d_model]
        intensity = self.Softplus((inten_raw * o).sum(dim=2))  # [bs, n_sample]

        integral_ = torch.cumsum(self.timestep * intensity, dim=1)
        density = (intensity * torch.exp(-integral_))
        t_pit = dt * density  # [bs, n_sample]
        estimate_dt = (self.timestep * 0.5 * (t_pit[:, 1:] + t_pit[:, :-1])).sum(dim=1)  # shape: n_batch
        return estimate_dt, dur_last


    def train_forward(self, s_ent, relation, o_ent, time, history_times,
                      node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local):
        query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type = \
            self.forward(s_ent, relation, history_times, node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local)

        type_intes, type,statics_ent_embeds = self.link_prediction(time, query_ent_embeds, query_rel_embeds, history_gh, history_times, history_pad_mask,
                                                total_nodes_h, local_type)

        last_time, _ = torch.max(history_times, 1)
        dur_last = time - last_time
        dur_last = torch.where(last_time > 0, dur_last, torch.zeros_like(dur_last))
        estimate_dt, dur_last = self.predict_t(o_ent, dur_last, query_ent_embeds, query_rel_embeds, history_gh, history_times, history_pad_mask,statics_ent_embeds)

        loss_lp = self.link_prediction_loss(type_intes, type, o_ent)
        loss_tp = self.time_prediction_loss(estimate_dt, dur_last)
        # loss_tp = 0
        return loss_lp, loss_tp

    def test_forward(self, s_ent, relation, o_ent, time, history_times,
                     node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local, local_weight=1.):
        query_ent_embeds, query_rel_embeds, history_gh, history_pad_mask, total_nodes_h, local_type = \
            self.forward(s_ent, relation, history_times, node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local)

        type_intes, type,statics_ent_embeds = self.link_prediction(time, query_ent_embeds, query_rel_embeds, history_gh, history_times,
                                                history_pad_mask,
                                                total_nodes_h, local_type)

        scores = self.ents_score(type_intes, type, local_weight)
        #
        last_time, _ = torch.max(history_times, 1)
        dur_last = time - last_time
        dur_last = torch.where(last_time > 0, dur_last, torch.zeros_like(dur_last))
        estimate_dt, dur_last = self.predict_t(o_ent, dur_last, query_ent_embeds, query_rel_embeds, history_gh,
                                               history_times, history_pad_mask,statics_ent_embeds)
        # estimate_dt = 0
        # dur_last = 0

        return scores, estimate_dt, dur_last
