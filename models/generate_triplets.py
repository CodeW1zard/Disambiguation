import numpy as np
from utils.settings import *
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient

class TripletsGenerator():
    def __init__(self):
        self.clusters_dict = load_data(BASIC_CLUSTER)
        self.pid_index = load_data(PID_INDEX)
        self.wv_client = LMDBClient(LMDB_WORDVEC)
        self.pid_total = []
        for _, pids in self.pid_index.items():
            self.pid_total.extend(pids)
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
        names = list(self.clusters_dict.keys())
        np.random.shuffle(names)
        self.cnt = 0
        for i, name in enumerate(names):
            print('prepare triplet pid ', i, name)
            clusters = self.clusters_dict[name]
            index2pid = self.pid_index[name]
            for cluster in clusters:
                excluded_pids = [index2pid[index] for index in cluster]
                if len(cluster) == 1:
                    continue
                num_to_generate = max([6, 0.1 * len(cluster)])
                num_to_generate = min([int(num_to_generate), 30, len(cluster)-1])
                for anchor in cluster:
                    cluster = [pid for pid in cluster if pid != anchor]
                    pos = np.random.choice(cluster, num_to_generate, replace=False)
                    neg = self.get_neg_pairs(num_to_generate, excluded_pids)
                    tri = [(index2pid[anchor], index2pid[pos[i]], neg[i]) for i in range(num_to_generate) if i!=anchor]
                    self.cnt += len(tri)
                    self.triplets.extend(tri)
                    if self.cnt > max_num:
                        dump_data(self.triplets, TRIPLET_INDEX)
                        return
            print(self.cnt)
        dump_data(self.triplets, TRIPLET_INDEX)


if __name__ == '__main__':
    TG = TripletsGenerator()
    TG.prepare_triplet_pid(max_num=500000)







