import argparse
import numpy as np
from itertools import chain
from utils import settings
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient

class TripletsGenerator():
    def __init__(self):
        self.clusters_dict_high = load_data(settings.BASIC_CLUSTER_HIGH)
        self.clusters_dict_low = load_data(settings.BASIC_CLUSTER_LOW)
        self.pid_index = load_data(settings.PID_INDEX)
        self.wv_client = LMDBClient(settings.LMDB_WORDVEC)
        self.assignments_low = load_data(settings.BASIC_CLUSTER_ARRAY_LOW)
        self.pid_total = list(chain.from_iterable(self.pid_index.values()))
        self.triplets = []

    def get_neg_pairs(self, num_to_generate, excluded_pids):
        cnt = 0
        negs = [None] * num_to_generate
        while cnt < num_to_generate:
            neg = np.random.choice(self.pid_total)
            if neg not in excluded_pids:
                negs[cnt] = neg
                cnt += 1
        return negs

    def prepare_triplet_pid(self, max_num = 100000):
        names = list(self.clusters_dict_high.keys())
        np.random.shuffle(names)
        self.cnt = 0
        for i, name in enumerate(names):
            print('prepare triplet pid ', i, name)
            clusters_high = self.clusters_dict_high[name]
            assignment_low = self.assignments_low[self.assignments_low[:, 0]==name]
            for cluster in clusters_high:
                if len(cluster) <= 1:
                    continue
                group = assignment_low[assignment_low[:, 2]==cluster[0]][0, 1]
                excluded_pids = assignment_low[assignment_low[:, 1]==group]

                num_to_generate = max([6, 0.1 * len(cluster)])
                num_to_generate = min([int(num_to_generate), 30, len(cluster)-1])
                for anchor in cluster:
                    pos = np.random.choice(cluster, num_to_generate, replace=False)
                    neg = self.get_neg_pairs(num_to_generate, excluded_pids)
                    tri = [(anchor, pos[i], neg[i]) for i in range(num_to_generate) if i != anchor]
                    self.cnt += len(tri)
                    self.triplets.extend(tri)
                    if self.cnt > max_num:
                        dump_data(self.triplets, settings.TRIPLET_INDEX)
                        return
            print(self.cnt)
        dump_data(self.triplets, settings.TRIPLET_INDEX)


if __name__ == '__main__':
    TG = TripletsGenerator()
    TG.prepare_triplet_pid(max_num=500000)







