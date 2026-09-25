import os.path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import compat  # noqa: F401 — must run before dgl on Python 3.10+
import dgl
import torch


def _dgl_graph(src, dst, num_nodes):
    """Build a homogeneous graph on both old DGL (0.x) and current DGL (1.x/2.x).

    On DGL 0.x ``dgl.graph`` is a *module*, not a constructor — calling it raises
    ``TypeError: 'module' object is not callable``.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    graph_fn = getattr(dgl, 'graph', None)
    if callable(graph_fn):
        return graph_fn((src, dst), num_nodes=int(num_nodes))
    g = dgl.DGLGraph()
    g.add_nodes(int(num_nodes))
    if src.size:
        g.add_edges(src, dst)
    return g


def _dgl_subgraph(g, nodes):
    try:
        sub = g.subgraph(list(nodes), store_ids=False)
    except TypeError:
        sub = g.subgraph(list(nodes))
    return _copy_induced_features(g, sub)


def _dgl_edge_subgraph(g, eids):
    if torch.is_tensor(eids) and eids.dtype == torch.bool:
        eids = eids.nonzero().view(-1)
    eids = _as_long_1d(eids)
    edge_subgraph = getattr(dgl, 'edge_subgraph', None)
    if callable(edge_subgraph):
        try:
            sub = edge_subgraph(g, eids, store_ids=False)
        except TypeError:
            sub = edge_subgraph(g, eids)
        return _copy_induced_features(g, sub)
    method = getattr(g, 'edge_subgraph', None)
    if callable(method):
        try:
            sub = method(eids, store_ids=False)
        except TypeError:
            sub = method(eids)
        return _copy_induced_features(g, sub)
    return _manual_edge_subgraph(g, eids)


def _manual_edge_subgraph(g, eids):
    """Keep only ``eids`` on DGL 0.x, which has no ``dgl.edge_subgraph``."""
    src, dst = g.edges()
    src = _as_long_1d(src)
    dst = _as_long_1d(dst)
    if eids is None or eids.numel() == 0:
        sub = _dgl_graph(np.array([]), np.array([]), num_nodes=1)
        sub.ndata['id'] = torch.tensor([[-1]], dtype=torch.long)
        empty = torch.zeros(0, dtype=torch.long)
        for key in ('type', 'timestamp', 'query_rel', 'query_ent'):
            sub.edata[key] = empty
        return sub

    src_k = src[eids]
    dst_k = dst[eids]
    nodes = torch.unique(torch.cat([src_k, dst_k], dim=0))
    remap = {int(old): new for new, old in enumerate(nodes.tolist())}
    new_src = np.array([remap[int(x)] for x in src_k.tolist()], dtype=np.int64)
    new_dst = np.array([remap[int(x)] for x in dst_k.tolist()], dtype=np.int64)
    sub = _dgl_graph(new_src, new_dst, num_nodes=int(nodes.numel()))
    if _has_ndata(g, 'id'):
        ids = g.ndata['id']
        if ids.dim() > 1:
            ids = ids.squeeze(-1)
        sub.ndata['id'] = ids[nodes].view(-1, 1)
    else:
        sub.ndata['id'] = nodes.view(-1, 1)
    for key in ('type', 'timestamp', 'query_rel', 'query_ent'):
        if _has_edata(g, key):
            sub.edata[key] = g.edata[key][eids]
        else:
            sub.edata[key] = torch.zeros(eids.numel(), dtype=torch.long)
    return sub


def _as_long_1d(value):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.long().reshape(-1)


def _parent_nids(sub):
    for name in ('parent_nid', '_parent_nid'):
        if hasattr(sub, name):
            ids = _as_long_1d(getattr(sub, name))
            if ids is not None and ids.numel() == _num_nodes(sub):
                return ids
    try:
        if '_ID' in sub.ndata:
            return _as_long_1d(sub.ndata['_ID'])
    except (KeyError, TypeError):
        pass
    return None


def _parent_eids(sub):
    for name in ('parent_eid', '_parent_eid'):
        if hasattr(sub, name):
            ids = _as_long_1d(getattr(sub, name))
            if ids is not None and ids.numel() == _num_edges(sub):
                return ids
    try:
        if '_ID' in sub.edata:
            return _as_long_1d(sub.edata['_ID'])
    except (KeyError, TypeError):
        pass
    return None


def _has_edata(g, key):
    try:
        g.edata[key]
        return True
    except (KeyError, TypeError):
        return False


def _has_ndata(g, key):
    try:
        g.ndata[key]
        return True
    except (KeyError, TypeError):
        return False


def _copy_induced_features(parent, sub):
    """Old DGL subgraphs do not copy ndata/edata; pull them via parent ids."""
    nids = _parent_nids(sub)
    eids = _parent_eids(sub)
    if not _has_ndata(sub, 'id') and _has_ndata(parent, 'id') and nids is not None:
        sub.ndata['id'] = parent.ndata['id'][nids]
    n_e = _num_edges(sub)
    for key in ('type', 'timestamp', 'query_rel', 'query_ent'):
        if _has_edata(sub, key):
            continue
        if _has_edata(parent, key) and eids is not None and eids.numel() == n_e and n_e > 0:
            sub.edata[key] = parent.edata[key][eids]
        else:
            sub.edata[key] = torch.zeros(n_e, dtype=torch.long)
    return sub


def _empty_rel_graph(root_node):
    g = dgl.DGLGraph()
    g.add_nodes(1, {'id': torch.tensor([[root_node]], dtype=torch.long)})
    empty = torch.zeros(0, dtype=torch.long)
    for key in ('type', 'query_rel', 'query_ent'):
        g.edata[key] = empty
    return g


def _pad_graph_nodes(g, n_nodes, pad_entity):
    """Clone ``g`` into a mutable graph with ``n_nodes`` (DGL subgraphs are readonly)."""
    src, dst = g.edges()
    src = _as_long_1d(src)
    dst = _as_long_1d(dst)
    n_old = _num_nodes(g)
    extra = max(int(n_nodes) - n_old, 0)
    new_g = _dgl_graph(
        src.cpu().numpy() if src.numel() else np.array([], dtype=np.int64),
        dst.cpu().numpy() if dst.numel() else np.array([], dtype=np.int64),
        num_nodes=n_old + extra,
    )
    if _has_ndata(g, 'id'):
        ids = g.ndata['id']
        if ids.dim() == 1:
            ids = ids.view(-1, 1)
        ids = ids.long().cpu()
    else:
        ids = torch.arange(n_old, dtype=torch.long).view(-1, 1)
    if extra > 0:
        ids = torch.cat([ids, torch.full((extra, 1), int(pad_entity), dtype=torch.long)], dim=0)
    new_g.ndata['id'] = ids
    n_e = int(src.numel())
    for key in ('type', 'timestamp', 'query_rel', 'query_ent'):
        if _has_edata(g, key):
            new_g.edata[key] = g.edata[key]
        else:
            new_g.edata[key] = torch.zeros(n_e, dtype=torch.long)
    return new_g


def _num_edges(g):
    if hasattr(g, 'num_edges') and callable(getattr(g, 'num_edges')):
        return g.num_edges()
    return g.number_of_edges()


def _num_nodes(g):
    if hasattr(g, 'num_nodes') and callable(getattr(g, 'num_nodes')):
        return g.num_nodes()
    return g.number_of_nodes()


def _batch_num_nodes(g):
    attr = getattr(g, 'batch_num_nodes', None)
    if callable(attr):
        return attr()
    if attr is not None:
        return torch.as_tensor(attr)
    return torch.tensor([_num_nodes(g)])


def graph_to_device(g, device):
    """Move a DGL graph (including old BatchedDGLGraph) onto ``device``."""
    to_fn = getattr(g, 'to', None)
    if callable(to_fn):
        try:
            return to_fn(device, non_blocking=True)
        except TypeError:
            return to_fn(device)
    for frame_name in ('ndata', 'edata'):
        frame = getattr(g, frame_name, None)
        if frame is None:
            continue
        try:
            keys = list(frame.keys())
        except Exception:
            continue
        for key in keys:
            val = frame[key]
            if torch.is_tensor(val):
                frame[key] = val.to(device, non_blocking=True)
    return g

class BaseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;
            testpath: File path of test data;
            statpath: File path of entities num and relatioins num;
            validpath: File path of valid data
        """
        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations
        self.skip_dict = self.get_skipdict(self.allQuadruples)

        self.train_snapshots = self.split_by_time(self.trainQuadruples)
        self.valid_snapshots = self.split_by_time(self.validQuadruples)
        self.test_snapshots = self.split_by_time(self.testQuadruples)

        self.time_inverted_index_dict = self.get_time_inverted_index_dict(self.allQuadruples)
        self.reltime2ent_dict = self.get_relation_time_dict(self.allQuadruples)
        self.alltimes = self.get_all_timestamps()

    def get_all_timestamps(self):
        """Get all the timestamps in the dataset.
        return:
            timestamps: a set of timestamps.
        """
        timestamps = set()
        for ex in self.allQuadruples:
            timestamps.add(ex[3])
        return sorted(list(timestamps))

    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, time in quadruples:
            filters[(src, rel, time)].add(dst)
            filters[(dst, rel+self.num_r, time)].add(src)
        return filters

    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            quadrupleList = []
            for line in f:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
        return quadrupleList

    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    @staticmethod
    def split_by_time(data):
        snapshot_list = []
        snapshot = []
        latest_t = 0
        for i in range(len(data)):
            t = data[i][3]
            train = data[i]
            if latest_t != t:
                if len(snapshot):
                    snapshot_list.append((np.array(snapshot).copy(), latest_t))
                snapshot = []
                latest_t = t
            snapshot.append(train[:3])
        if len(snapshot) > 0:
            snapshot_list.append((np.array(snapshot).copy(), latest_t))
        return snapshot_list

    @staticmethod
    def get_reverse_quadruples_array(quadruples, num_r):
        quads = np.array(quadruples)
        quads_r = np.zeros_like(quads)
        quads_r[:, 1] = num_r + quads[:, 1]
        quads_r[:, 0] = quads[:, 2]
        quads_r[:, 2] = quads[:, 0]
        quads_r[:, 3] = quads[:, 3]
        return np.concatenate((quads, quads_r))

    def get_time_inverted_index_dict(self, quadruples):
        index_dict = defaultdict(set)
        for quad in quadruples:
            index_dict[quad[0]].add(quad[3])
            index_dict[quad[2]].add(quad[3])
            index_dict[(quad[0], quad[1])].add(quad[3])
            index_dict[(quad[2], quad[1] + self.num_r)].add(quad[3])
        for k, v in index_dict.items():
            index_dict[k] = sorted(list(v))
        return index_dict

    def get_relation_time_dict(self, quadruples):
        dict = defaultdict(list)
        for quad in quadruples:
            dict[(quad[1], quad[3])].append([quad[0], quad[2], quad[3]])
            dict[(quad[1] + self.num_r, quad[3])].append([quad[2], quad[0], quad[3]])
        return dict

    def get_relation_triples(self,quadruples):
        out_dict = defaultdict(set)
        in_dict = defaultdict(set)
        for facts in quadruples:
            if isinstance(facts, (list, tuple)):
                out_dict[facts[0]].add((facts[1], facts[3]))
                in_dict[facts[2]].add((facts[1], facts[3]))
                in_dict[facts[0]].add((facts[1] + self.num_r, facts[3]))
                in_dict[facts[2]].add((facts[1] + self.num_r, facts[3]))
            else:
                # Handle cases where facts is not iterable or doesn't have enough elements
                print("Invalid facts format:", facts)

        entities = set(out_dict.keys()).union(in_dict.keys())

        r_triples = set()
        return list(r_triples)

class SnapshotIndex(object):
    """Per-timestamp edge list + in-neighbor CSR. Avoids DGL subgraph on 0.1.3."""

    def __init__(self, src, dst, rel, n_ent):
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.rel = np.asarray(rel, dtype=np.int64)
        self.n_ent = int(n_ent)
        if self.src.size == 0:
            self._src_by_dst = self.src
            self._in_ptr = np.zeros(self.n_ent + 1, dtype=np.int64)
            return
        order = np.argsort(self.dst, kind='mergesort')
        self._src_by_dst = self.src[order]
        counts = np.bincount(self.dst, minlength=self.n_ent)
        self._in_ptr = np.zeros(self.n_ent + 1, dtype=np.int64)
        np.cumsum(counts, out=self._in_ptr[1:])

    def in_neighbors(self, node):
        node = int(node)
        if node < 0 or node >= self.n_ent:
            return self._src_by_dst[:0]
        return self._src_by_dst[self._in_ptr[node]:self._in_ptr[node + 1]]

    def collect_nodes(self, root, n_hop):
        root = int(root)
        keep = np.zeros(self.n_ent, dtype=bool)
        keep[root] = True
        frontier = np.array([root], dtype=np.int64)
        for _ in range(max(int(n_hop), 0)):
            if frontier.size == 0:
                break
            chunks = [self.in_neighbors(v) for v in frontier]
            merged = np.concatenate(chunks) if chunks else frontier[:0]
            if merged.size == 0:
                break
            merged = np.unique(merged)
            new = merged[~keep[merged]]
            keep[new] = True
            frontier = new
        return np.nonzero(keep)[0]

    def make_graph(self, root, n_hop, conf_row=None):
        root = int(root)
        nodes = self.collect_nodes(root, n_hop)
        if self.src.size == 0 or nodes.size == 0:
            return _empty_rel_graph(root)
        keep_node = np.zeros(self.n_ent, dtype=bool)
        keep_node[nodes] = True
        emask = keep_node[self.src] & keep_node[self.dst]
        if conf_row is not None:
            conf_np = conf_row.detach().cpu().numpy() if torch.is_tensor(conf_row) else np.asarray(conf_row)
            emask = np.logical_and(emask, conf_np[self.rel] > 0.1)
        src, dst, rel = self.src[emask], self.dst[emask], self.rel[emask]
        if src.size == 0:
            return _empty_rel_graph(root)
        used = np.unique(np.concatenate([src, dst, np.array([root], dtype=np.int64)]))
        if root not in used:
            return _empty_rel_graph(root)
        remap = np.full(self.n_ent, -1, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        g = _dgl_graph(remap[src], remap[dst], num_nodes=int(used.size))
        g.ndata['id'] = torch.from_numpy(np.ascontiguousarray(used)).view(-1, 1)
        g.edata['type'] = torch.from_numpy(np.ascontiguousarray(rel))
        return g


def _bidirectional_edges(triples, num_rels):
    if triples is None or np.asarray(triples).size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty
    src, rel, dst = np.asarray(triples).transpose()
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    rel = rel.astype(np.int64, copy=False)
    return (
        np.concatenate((src, dst)),
        np.concatenate((dst, src)),
        np.concatenate((rel, rel + num_rels)),
    )


class DGLGraphDataset(object):
    def __init__(self, ent_snapshots, n_ent, n_rel):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.n_hyper_rel = 4
        self.snapshots_num = len(ent_snapshots)
        self.snapshots = ent_snapshots
        self.snapshots_index = {}
        for triples, time in ent_snapshots:
            src, dst, rel = _bidirectional_edges(triples, n_rel)
            self.snapshots_index[int(time)] = SnapshotIndex(src, dst, rel, n_ent)
        empty = np.array([], dtype=np.int64)
        self.snapshots_index[-1] = SnapshotIndex(empty, empty, empty, n_ent)
        self.dgl_rel_graphs = {}

    def history_graph(self, time, root_node, n_hop, conf_row=None):
        snap = self.snapshots_index.get(int(time), self.snapshots_index[-1])
        return snap.make_graph(root_node, n_hop, conf_row)

    def get_nhop_subgraph(self, time, root_node, n=2):
        return self.history_graph(time, root_node, n)

    def edge_samples(self, root_node, sub_g, conf):
        return sub_g

    def comp_deg_norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    def get_relation_dglGraph(self, triples, num_nodes, num_rels):
        triples = np.array(list(triples))
        if triples.size != 0:
            src, rel, dst = triples.transpose()
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + num_rels))
        else:
            src, rel, dst = np.array([]), np.array([]), np.array([])
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src, dst)

        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        g.ndata.update({'id': node_id})
        g.edata['type'] = torch.LongTensor(rel)
        return g

    def get_nhop_rel_subgraph(self, root_node, time, n=2, max_nodes=10):
        g = self.dgl_rel_graphs[time]
        total_nodes = set()
        total_nodes.add(root_node)
        for i in range(n):
            step_nodes = total_nodes.copy()
            for node in step_nodes:
                neighbor_n, _ = g.in_edges(node)
                neighbor_n = set(neighbor_n.tolist())

                if len(total_nodes) + len(neighbor_n) > max_nodes:
                  total_nodes |= set(list(neighbor_n)[:max_nodes - len(total_nodes)])
                  #total_nodes = set(list(total_nodes)[:max_nodes])
                  break
                else:
                  total_nodes |= neighbor_n
        sub_g = _dgl_subgraph(g, total_nodes)
        return sub_g

class QuadruplesDataset(Dataset):
    def __init__(self, quadruples, history_len, dglGraphs, baseDataset, history_mode='recent', nhop=2,
                 forecasting_t_windows_size=1, time_span=24, edges_conf=None, edge_sample=False, dataset_type='train'):
        self.quadruples = quadruples
        self.history_len = history_len
        self.dglGraphs = dglGraphs
        self.timeInvDict = baseDataset.time_inverted_index_dict
        self.nhop = nhop
        self.history_mode = history_mode
        self.PAD_TIME = -1
        self.num_r = baseDataset.num_r
        self.forecasting_t_windows_size = forecasting_t_windows_size
        self.time_span = time_span
        self.edges_conf = edges_conf
        self.edge_sample = edge_sample
        self.dataset_type = dataset_type
        self.delta_t = 1

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.quadruples) * self.forecasting_t_windows_size
        else:
            return len(self.quadruples)

    def __getitem__(self, idx):
        if self.dataset_type == 'train':
            quad_idx = idx // self.forecasting_t_windows_size
            delta_t = idx % self.forecasting_t_windows_size + 1
            quad = self.quadruples[quad_idx]
            head_entity, relation, tail_entity, timestamp = quad[0], quad[1], quad[2], quad[3]
            history_graphs, history_times, head_entity_ids, graphs_node_num = \
                self.get_history_graphs(head_entity, relation, timestamp, self.history_mode, delta_t)
            return head_entity, relation, tail_entity, timestamp, \
                  history_graphs, history_times, head_entity_ids, graphs_node_num#, sub_rel_graph, relation_ids, static_ent_graph, static_ent_ids
        else:
            quad = self.quadruples[idx]
            head_entity, relation, tail_entity, timestamp = quad[0], quad[1], quad[2], quad[3]
            history_graphs, history_times, head_entity_ids, graphs_node_num = \
                self.get_history_graphs(head_entity, relation, timestamp, self.history_mode, self.delta_t)
            return head_entity, relation, tail_entity, timestamp, \
                   history_graphs, history_times, head_entity_ids, graphs_node_num#, sub_rel_graph, relation_ids, static_ent_graph, static_ent_ids

    def get_history_graphs(self, head_entity, relation, timestamp, sampled_method='recent', delta_t=1):
        if sampled_method == 'history_copy':
            times = self.timeInvDict[(head_entity, relation)]
            history_times = times[:times.index(timestamp)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]
        elif sampled_method == 'both':
            times1 = self.timeInvDict[(head_entity, relation)]
            times2 = self.timeInvDict[head_entity]
            history_times1 = times1[:times1.index(timestamp)]
            history_times1 = history_times1[max(-self.history_len, -len(history_times1)):]
            history_times2 = times2[:times2.index(timestamp)]
            history_times2 = history_times2[max(-max(self.history_len - len(history_times1),1), -len(history_times2)):]
            history_times = sorted(list(set(history_times1 + history_times2)))
        elif sampled_method == 'delta_t_windows':
            times1 = self.timeInvDict[(head_entity, relation)]
            times2 = self.timeInvDict[head_entity]
            history_times1 = times1[:times1.index(timestamp)]
            history_times1 = list(filter(lambda x: timestamp - x > 0, history_times1))
            history_times1 = history_times1[max(-self.history_len // 2, -len(history_times1)):]
            history_times2 = times2[:times2.index(timestamp)]
            history_times2 = list(filter(lambda x: timestamp - x > 0, history_times2))
            history_times2 = history_times2[max(-(self.history_len - self.history_len // 2), -len(history_times2)):]
            history_times = sorted(list(set(history_times1 + history_times2)))
            if len(history_times) == 0:
                history_times = [-1]
        else:
            times = self.timeInvDict[head_entity]
            history_times = times[:times.index(timestamp)]
            history_times = history_times[max(-self.history_len, -len(history_times)):]

        history_graphs = []
        head_entity_ids = []
        graphs_node_num = []
        conf_row = self.edges_conf[relation] if self.edge_sample else None
        for i, t in enumerate(history_times):
            sub_graph = self.dglGraphs.history_graph(t, head_entity, self.nhop, conf_row)
            n_e = _num_edges(sub_graph)
            if not _has_edata(sub_graph, 'type'):
                sub_graph.edata['type'] = torch.zeros(n_e, dtype=torch.long)
            sub_graph.edata['query_rel'] = torch.full((n_e,), int(relation), dtype=torch.long)
            sub_graph.edata['query_ent'] = torch.full((n_e,), int(head_entity), dtype=torch.long)
            history_graphs.append(sub_graph)
            node_ids = sub_graph.ndata['id']
            if node_ids.dim() > 1:
                node_ids = node_ids.squeeze(1)
            head_entity_ids.append(node_ids.tolist().index(head_entity))
            graphs_node_num.append(_num_nodes(sub_graph))

        #static_rel_graph = dgl.merge(history_rel_graphs)
        #relation_ids = static_rel_graph.ndata['id'].squeeze(1).tolist().index(relation)
        #static_ent_graph = dgl.merge(history_graphs)
        #static_ent_ids = static_ent_graph.ndata['id'].squeeze(1).tolist().index(head_entity)
        #relation_ids = 0
        return history_graphs, history_times, head_entity_ids, graphs_node_num#, static_rel_graph, relation_ids, static_ent_graph, static_ent_ids

    @staticmethod
    def collate_fn(batch, pad_entity):
        batch_data = list(zip(*batch))
        head_entites = batch_data[0]
        relations = batch_data[1]
        tail_entities = batch_data[2]
        timestamps = batch_data[3]

        history_graphs = batch_data[4]  # list
        history_times = batch_data[5]
        head_entity_ids = batch_data[6]
        graphs_node_num = batch_data[7]
        #sub_rel_graph = batch_data[8]
        #relation_ids = batch_data[9]
        #static_ent_graph = batch_data[10]
        #static_ent_ids = batch_data[11]

        max_history_len = max([len(t) for t in history_times] + [1])
        max_nodes_num = max(sum(graphs_node_num, [1]))

        pad_history_graphs = []
        pad_history_times = []
        pad_history_eids = []

        for i in range(len(history_graphs)):
            hgs = history_graphs[i]
            hts = history_times[i]
            heids = head_entity_ids[i]

            if len(hgs) < max_history_len:
                PAD_G = []
                for j in range(max_history_len - len(hgs)):
                    g = dgl.DGLGraph()
                    g.add_nodes(1, {'id': torch.tensor(head_entites[i]).long().view(-1, 1)})
                    g.edata['type'] = torch.LongTensor(np.array([]))
                    g.edata['query_rel'] = torch.ones_like(g.edata['type'])
                    g.edata['query_ent'] = torch.ones_like(g.edata['type'])
                    # g.edata['query_time'] = torch.ones_like(g.edata['type'])

                    # in_deg = g.in_degrees(range(g.number_of_nodes())).float()
                    # in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
                    # norm = 1.0 / in_deg
                    # g.ndata['norm'] = norm.view(-1, 1)

                    PAD_G.append(g)

                PAD_HT = [-1 for j in range(max_history_len - len(hgs))]
                PAD_EID = [0 for j in range(max_history_len - len(hgs))]

                hgs.extend(PAD_G)
                hts.extend(PAD_HT)
                heids.extend(PAD_EID)

            padded = []
            for g in hgs:
                padded.append(_pad_graph_nodes(g, max_nodes_num, pad_entity))

            pad_history_graphs.extend(padded)
            pad_history_times.append(hts)
            pad_history_eids.extend(heids)

        pad_history_graphs = dgl.batch(pad_history_graphs)
        batch_node_ids = torch.tensor(pad_history_eids)
        batchgraph_nodes_num = _batch_num_nodes(pad_history_graphs)
        graph_num = batchgraph_nodes_num.size(0)
        offset_node_ids = batchgraph_nodes_num.unsqueeze(0).repeat(graph_num, 1)
        offset_mask = torch.tril(torch.ones(graph_num, graph_num), diagonal=-1).long()
        offset_node_ids = offset_node_ids * offset_mask
        offset_node_ids = torch.sum(offset_node_ids, dim=1)
        batch_node_ids += offset_node_ids

        head_entites = torch.tensor(head_entites)  # [bs]
        relations = torch.tensor(relations)  # [bs]
        tail_entities = torch.tensor(tail_entities)  # [bs]
        timestamps = torch.tensor(timestamps)  # [bs]
        pad_history_times = torch.tensor(pad_history_times)  # [bs, history_len]

        #sub_rel_graph = dgl.batch(sub_rel_graph)
        #batch_rel_ids = torch.tensor(relation_ids)
        #batchgraph_rels_num = sub_rel_graph.batch_num_nodes()
        #graph_num = batchgraph_rels_num.size(0)
        #offset_rel_ids = batchgraph_rels_num.unsqueeze(0).repeat(graph_num, 1)
        #offset_mask = torch.tril(torch.ones(graph_num, graph_num), diagonal=-1).long()
        #offset_rel_ids = offset_rel_ids * offset_mask
        #offset_rel_ids = torch.sum(offset_rel_ids, dim=1)
        #batch_rel_ids += offset_rel_ids

        #static_ent_graph = dgl.batch(static_ent_graph)
        #batch_static_ent_ids = torch.tensor(static_ent_ids)
        #batchstat_ents_num = static_ent_graph.batch_num_nodes()
        #graph_num = batchstat_ents_num.size(0)
        #offset_ent_ids = batchstat_ents_num.unsqueeze(0).repeat(graph_num, 1)
        #offset_mask = torch.tril(torch.ones(graph_num, graph_num), diagonal=-1).long()
        #offset_ent_ids = offset_ent_ids * offset_mask
        #offset_ent_ids = torch.sum(offset_ent_ids, dim=1)
        #batch_static_ent_ids += offset_ent_ids

        return head_entites, relations, tail_entities, timestamps, pad_history_graphs, pad_history_times, batch_node_ids#, sub_rel_graph, batch_rel_ids, static_ent_graph,batch_static_ent_ids
