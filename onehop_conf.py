import os
import pickle
from collections import defaultdict

import numpy as np

from dataset import BaseDataset


def build_one_hop_conf(train_quadruples, num_r):
    """Relation-pair confidence used by ``edge_sample=one_hop_conf``.

    For each training fact (s, r, o, t), look at earlier facts on (s, o) or
    (o, s) and count which relations appeared. Linear in the number of facts,
    unlike the original nested scan.
    """
    quads = np.asarray(train_quadruples)
    n_rel = num_r * 2
    counts_head = np.zeros(n_rel, dtype=np.float64)
    counts_body = np.zeros((n_rel, n_rel), dtype=np.float64)

    by_time = defaultdict(list)
    for row in quads:
        by_time[int(row[3])].append((int(row[0]), int(row[1]), int(row[2])))

    pair_rels = defaultdict(set)
    for time in sorted(by_time):
        batch = by_time[time]
        for src, rel, dst in batch:
            body = pair_rels[(src, dst)] | pair_rels[(dst, src)]
            if not body:
                continue
            counts_head[rel] += 1
            for body_rel in body:
                counts_body[rel, body_rel] += 1
        for src, rel, dst in batch:
            pair_rels[(src, dst)].add(rel)

    conf = np.zeros((n_rel, n_rel), dtype=np.float64)
    seen = counts_head > 0
    conf[seen] = counts_body[seen] / counts_head[seen, None]
    return conf


def load_or_build_conf(data_path, train_quadruples, num_r):
    conf_path = os.path.join(data_path, 'conf.pkl')
    if os.path.isfile(conf_path):
        with open(conf_path, 'rb') as handle:
            return pickle.load(handle)
    conf = build_one_hop_conf(train_quadruples, num_r)
    with open(conf_path, 'wb') as handle:
        pickle.dump(conf, handle)
    return conf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--data', default='ICEWS14')
    args = parser.parse_args()

    data_path = os.path.join(args.data_root, args.data)
    base = BaseDataset(
        os.path.join(data_path, 'train.txt'),
        os.path.join(data_path, 'test.txt'),
        os.path.join(data_path, 'stat.txt'),
        os.path.join(data_path, 'valid.txt'),
    )
    train = base.get_reverse_quadruples_array(base.trainQuadruples, base.num_r)
    conf = load_or_build_conf(data_path, train, base.num_r)
    print(conf.shape, conf[0, :8])
