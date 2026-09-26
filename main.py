import argparse
import compat  # noqa: F401 — patch collections for DGL on Python 3.10+
import torch
import os
from tqdm import tqdm
from dataset import *
from embeddings import load_pretrained_tables, pca_reduce_tables
from model import TemporalTransformerHawkesGraphModel
import logging
from collections import namedtuple
from torch.utils.data import DataLoader
from utils import set_logger
from onehop_conf import load_or_build_conf
import math

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Reasoning Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--model_name', type=str, default='GHT',
                        help='GHT: graph+temporal transformer. BehaviorLM: pretrained LLM on event traces.')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--num_works', type=int, default=0,
                        help='DataLoader workers. 0 is required on Colab (each worker copies the full TKG).')

    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='AdamW L2 on non-embedding weights.')
    parser.add_argument('--emb_weight_decay', type=float, default=0.01,
                        help='Stronger L2 on entity/relation tables; they memorize fastest.')

    parser.add_argument('--d_model', default=100, type=int,
                        help='GHT hidden size. Qwen embeddings are 1024-d; they are reduced to this.')
    parser.add_argument('--emb_reduce', default='pca', choices=['pca', 'linear'],
                        help='pca: SVD 1024->d_model then freeze. linear: keep 1024-d tables and train Linear(1024, d_model).')
    parser.add_argument('--data', default='ICEWS14', type=str)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--valid_epoch', default=2, type=int,
                        help='Validate every N epochs so the peak is not missed.')
    parser.add_argument('--patience', default=1, type=int,
                        help='Stop after this many validations without Filter MRR gain. 0 disables.')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='Multiply LR after a validation that does not improve Filter MRR.')
    parser.add_argument('--history_len', default=10, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--seqTransformerLayerNum', default=2, type=int)
    parser.add_argument('--seqTransformerHeadNum', default=2, type=int)

    parser.add_argument('--load_model_path', default='output', type=str)

    parser.add_argument('--history_mode', default='delta_t_windows', type=str)
    parser.add_argument('--nhop', default=1, type=int)
    parser.add_argument('--forecasting_t_win_size', default=1, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=1.0, type=float)

    parser.add_argument('--time_span', default=24, type=int)
    parser.add_argument('--timestep', default=0.1, type=float)
    parser.add_argument('--hmax', default=5, type=int)
    parser.add_argument('--eps', default=0.2, type=float)
    parser.add_argument('--edge_sample', default='one_hop_conf',
                        choices=['one_hop_conf', 'none'],
                        help='one_hop_conf: filter 1-hop edges with data/<dataset>/conf.pkl. '
                             'none: skip conf.pkl and keep all 1-hop edges.')
    parser.add_argument('--desc', default='', type=str)

    parser.add_argument('--warm_up', default=0.0, type=float)

    parser.add_argument('--emb_init', default='scratch', choices=['scratch', 'pretrained'],
                        help='scratch: Xavier tables. pretrained: Qwen3-Embedding over entity/relation names.')
    parser.add_argument('--emb_model', default='Qwen/Qwen3-Embedding-0.6B',
                        help='HuggingFace embedding model. 0.6B fits Colab Free; 4B/8B likely OOM on a free T4.')
    parser.add_argument('--emb_batch_size', default=16, type=int)
    parser.add_argument('--emb_max_length', default=128, type=int)
    parser.add_argument('--tune_pretrained_emb', action='store_true',
                        help='Fine-tune the pretrained tables instead of freezing them (projection still trains)')
    parser.add_argument('--lm_model', default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Causal LM for BehaviorLM. 0.5B fits Colab Free T4; 1.5B is tighter.')
    parser.add_argument('--lm_max_length', default=256, type=int)
    parser.add_argument('--lm_max_events', default=24, type=int,
                        help='Cap on past events packed into one BehaviorLM prompt.')
    parser.add_argument('--lm_tune', action='store_true',
                        help='Finetune the LM backbone (needs more GPU). Default: freeze LM, train projector.')

    return parser.parse_args(args)


def _metric_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _adamw_param_groups(model, weight_decay, emb_weight_decay):
    if not hasattr(model, 'ent_embeds'):
        return [{'params': [p for p in model.parameters() if p.requires_grad],
                 'weight_decay': weight_decay}]
    emb_ids = {id(param) for param in list(model.ent_embeds.parameters()) + list(model.rel_embeds.parameters())}
    other, embeds = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in emb_ids:
            embeds.append(param)
        else:
            other.append(param)
    return [
        {'params': other, 'weight_decay': weight_decay},
        {'params': embeds, 'weight_decay': emb_weight_decay},
    ]


def _run_train_forward(model, batch, args):
    if args.model_name == 'BehaviorLM':
        sub, rel, obj, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask = batch
        return model.train_forward(sub, rel, obj, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask)
    sub, rel, obj, time, node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local, history_times = batch
    return model.train_forward(
        sub, rel, obj, time, history_times,
        node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local)


def _run_test_forward(model, batch, args):
    if args.model_name == 'BehaviorLM':
        sub, rel, obj, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask = batch
        return model.test_forward(sub, rel, obj, time, ev_src, ev_rel, ev_dst, ev_time, ev_mask, args.beta)
    sub, rel, obj, time, node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local, history_times = batch
    return model.test_forward(
        sub, rel, obj, time, history_times,
        node_ids, edge_src, edge_dst, edge_type, edge_mask, root_local, args.beta)


def _make_loader(dataset, batch_size, num_workers, shuffle, pad_entity, output_mode='graph'):
    if output_mode == 'behavior':
        collate_fn = lambda x: dataset.collate_behavior(x)
    else:
        collate_fn = lambda x: dataset.collate_fn(x, pad_entity)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = False
    return DataLoader(**kwargs)

def _batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def test(model, testloader, skip_dict, device):
    model.eval()
    ranks = []
    logs = []
    TimeMSE = 0.
    TimeMAE = 0.
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = _batch_to_device(batch, device)
            sub, rel, obj, time = batch[0], batch[1], batch[2], batch[3]
            scores, estimate_dt, dur_last = _run_test_forward(model, batch, args)

            mse_loss = torch.nn.MSELoss(reduction='sum')(estimate_dt, dur_last)
            mae_loss = torch.nn.L1Loss(reduction='sum')(estimate_dt, dur_last)

            TimeMSE += mse_loss
            TimeMAE += mae_loss

            _, rank_idx = scores.sort(dim=1, descending=True)
            rank = torch.nonzero(rank_idx == obj.view(-1, 1))[:, 1].view(-1)
            ranks.append(rank)

            for i in range(scores.shape[0]):
                src_i = sub[i].item()
                rel_i = rel[i].item()
                dst_i = obj[i].item()
                time_i = time[i].item()

                predict_score = scores[i].tolist()
                answer_prob = predict_score[dst_i]
                for e in skip_dict[(src_i, rel_i, time_i)]:
                    if e != dst_i:
                        predict_score[e] = -1e6
                predict_score.sort(reverse=True)
                filter_rank = predict_score.index(answer_prob) + 1

                logs.append({
                    'Time-aware Filter MRR': 1.0 / filter_rank,
                    'Time-aware Filter HITS@1': 1.0 if filter_rank <= 1 else 0.0,
                    'Time-aware Filter HITS@3': 1.0 if filter_rank <= 3 else 0.0,
                    'Time-aware Filter HITS@10': 1.0 if filter_rank <= 10 else 0.0,
                })

    metrics = {}
    ranks = torch.cat(ranks)
    ranks += 1
    mrr = torch.mean(1.0 / ranks.float())
    metrics['Raw MRR'] = mrr
    for hit in [1, 3, 10]:
        avg_count = torch.mean((ranks <= hit).float())
        metrics['Raw Hit@{}'.format(hit)] = avg_count

    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    metrics['Time MSE'] = TimeMSE / len(testloader.dataset)
    metrics['Time MAE'] = TimeMAE / len(testloader.dataset)
    return metrics


def train_epoch(args, model, traindataloader, optimizer, scheduler, device, epoch):
    model.train()
    with tqdm(total=len(traindataloader), unit='ex') as bar:
        bar.set_description('Train')
        total_loss = 0
        total_num = 0
        for batch in traindataloader:
            if epoch < args.warm_up:
                scheduler.step()
            batch = _batch_to_device(batch, device)
            lp_loss, tp_loss = _run_train_forward(model, batch, args)
            loss = lp_loss + args.alpha * tp_loss
            # loss = lp_loss
            loss.backward()

            total_loss += loss
            total_num += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            bar.update(1)
            bar.set_postfix(loss='%.4f' % loss, lp_loss='%.4f' % lp_loss, tp_loss='%.4f' % tp_loss)

        logging.info('Epoch {} Train Loss: {}'.format(epoch, total_loss/total_num))


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main(args):
    output_path = os.path.join(args.output_root, '{0}_{1}'.format(args.data, args.model_name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = os.path.join(output_path, 'log.txt')
    set_logger(log_file)

    logging.info(args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data_path = os.path.join(args.data_root, args.data)
    trainpath = os.path.join(data_path, 'train.txt')
    validpath = os.path.join(data_path, 'valid.txt')
    testpath = os.path.join(data_path, 'test.txt')
    statpath = os.path.join(data_path, 'stat.txt')
    baseDataset = BaseDataset(trainpath, testpath, statpath, validpath)

    dglGraphDataset = DGLGraphDataset(
        baseDataset.train_snapshots + baseDataset.valid_snapshots + baseDataset.test_snapshots,
        baseDataset.num_e, baseDataset.num_r)

    trainQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.trainQuadruples, baseDataset.num_r)
    if args.edge_sample == 'one_hop_conf':
        logging.info('Loading one-hop relation confidence from {}'.format(
            os.path.join(data_path, 'conf.pkl')))
        edges_conf = torch.tensor(load_or_build_conf(
            data_path, trainQuadruples, baseDataset.num_r))
        edge_sample = True
    else:
        logging.info('Skipping conf.pkl (edge_sample=none); using full 1-hop graphs')
        edges_conf = None
        edge_sample = False
    output_mode = 'behavior' if args.model_name == 'BehaviorLM' else 'graph'
    if args.model_name == 'BehaviorLM' and args.batch_size > 16:
        logging.info('BehaviorLM: lowering batch_size %d -> 8 for the LLM encoder', args.batch_size)
        args.batch_size = 8
    trainQuadDataset = QuadruplesDataset(trainQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                         args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                         edges_conf, edge_sample, 'train',
                                         output_mode, args.lm_max_events)
    trainDataLoader = _make_loader(
        trainQuadDataset, args.batch_size, args.num_works, True, baseDataset.num_e, output_mode)


    validQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.validQuadruples, baseDataset.num_r)
    validQuadDataset = QuadruplesDataset(validQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                        args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                        edges_conf, edge_sample, 'test',
                                        output_mode, args.lm_max_events)
    validDataLoader = _make_loader(
        validQuadDataset, args.batch_size, args.num_works, False, baseDataset.num_e, output_mode)

    testQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.testQuadruples, baseDataset.num_r)
    testQuadDataset = QuadruplesDataset(testQuadruples, args.history_len, dglGraphDataset, baseDataset,
                                        args.history_mode, args.nhop, args.forecasting_t_win_size, args.time_span,
                                        edges_conf, edge_sample, 'test',
                                        output_mode, args.lm_max_events)
    testDataLoader = _make_loader(
        testQuadDataset, args.batch_size, args.num_works, False, baseDataset.num_e, output_mode)

    Config = namedtuple('config', ['n_ent', 'd_model', 'n_rel', 'dropout','seqTransformerLayerNum', 'seqTransformerHeadNum'])
    config = Config(n_ent=baseDataset.num_e + 1,
                    n_rel=baseDataset.num_r * 2,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    seqTransformerLayerNum=args.seqTransformerLayerNum,
                    seqTransformerHeadNum=args.seqTransformerHeadNum)
    ent_pretrained = rel_pretrained = None
    freeze_pretrained = not args.tune_pretrained_emb
    if args.model_name != 'BehaviorLM' and args.emb_init == 'pretrained':
        ent_pretrained, rel_pretrained = load_pretrained_tables(
            data_path,
            baseDataset.num_e,
            baseDataset.num_r,
            model_name=args.emb_model,
            batch_size=args.emb_batch_size,
            max_length=args.emb_max_length,
        )
        logging.info('Pretrained embeddings ent={} rel={} freeze={}'.format(
            tuple(ent_pretrained.shape), tuple(rel_pretrained.shape), freeze_pretrained))
        if args.emb_reduce == 'pca' and ent_pretrained.size(1) != args.d_model:
            logging.info('PCA-reducing pretrained dim {} -> d_model {}'.format(
                ent_pretrained.size(1), args.d_model))
            ent_pretrained, rel_pretrained = pca_reduce_tables(
                ent_pretrained, rel_pretrained, args.d_model)

    if args.model_name == 'BehaviorLM':
        from models.BehaviorLM import BehaviorLM
        model = BehaviorLM(
            n_ent=config.n_ent,
            n_rel=config.n_rel,
            data_dir=data_path,
            lm_model=args.lm_model,
            max_length=args.lm_max_length,
            dropout=args.dropout,
            eps=args.eps,
            freeze_lm=not args.lm_tune,
        )
    else:
        model = TemporalTransformerHawkesGraphModel(
            config, args.eps, args.time_span, args.timestep, args.hmax,
            ent_pretrained=ent_pretrained, rel_pretrained=rel_pretrained,
            freeze_pretrained=freeze_pretrained)
    model.to(device)

    optimizer = torch.optim.AdamW(
        _adamw_param_groups(model, args.weight_decay, args.emb_weight_decay),
        lr=args.lr)
    if args.warm_up > 0:
        warmup_scheduler = WarmUpLR(optimizer, len(trainDataLoader) * args.warm_up)
    else:
        warmup_scheduler = None

    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    def _save_ckpt(path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    if args.do_train:
        logging.info('Start Training......')
        best_mrr = -1.0
        bad_rounds = 0
        best_path = os.path.join(output_path, 'model_best.pth')

        for i in range(args.max_epochs):
            train_epoch(args, model, trainDataLoader, optimizer, warmup_scheduler, device, i)
            if (i + 1) % args.valid_epoch != 0:
                continue

            _save_ckpt(os.path.join(output_path, 'model_{}.pth'.format(i + 1)))

            valid_mrr = None
            for win in range(args.forecasting_t_win_size):
                delta_t = win + 1
                validDataLoader.dataset.delta_t = delta_t
                metrics = test(model, validDataLoader, baseDataset.skip_dict, device)
                for mode in metrics.keys():
                    logging.info('Delta_t {} Valid {} : {}'.format(delta_t, mode, metrics[mode]))
                if valid_mrr is None:
                    valid_mrr = _metric_float(metrics['Time-aware Filter MRR'])

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                bad_rounds = 0
                _save_ckpt(best_path)
                logging.info('New best Filter MRR {:.4f} at epoch {}'.format(best_mrr, i + 1))
            else:
                bad_rounds += 1
                old_lr = optimizer.param_groups[0]['lr']
                if 0.0 < args.lr_decay < 1.0:
                    for group in optimizer.param_groups:
                        group['lr'] *= args.lr_decay
                logging.info(
                    'Valid Filter MRR {:.4f} < best {:.4f}. LR {:.6f} -> {:.6f} (bad {}/{})'.format(
                        valid_mrr, best_mrr, old_lr, optimizer.param_groups[0]['lr'],
                        bad_rounds, args.patience))
                if args.patience > 0 and bad_rounds >= args.patience:
                    logging.info('Early stopping at epoch {}'.format(i + 1))
                    break

        if os.path.isfile(best_path):
            params = torch.load(best_path, map_location=device)
            model.load_state_dict(params['model_state_dict'])
            logging.info('Loaded best model (Filter MRR {:.4f}) from {}'.format(best_mrr, best_path))

    if args.do_test:
        logging.info('Start Testing......')
        for win in range(args.forecasting_t_win_size):
            delta_t = win + 1
            testDataLoader.dataset.delta_t = delta_t
            metrics = test(model, testDataLoader, baseDataset.skip_dict, device)
            for mode in metrics.keys():
                logging.info('Delta_t {} Test {} : {}'.format(delta_t, mode, metrics[mode]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
