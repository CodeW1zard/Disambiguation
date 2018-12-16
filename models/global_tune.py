import numpy as np
from utils.settings import *
from utils.data_utils import *

def prepare_triplets(max_num = 100000):
    pid_index = load_data(PID_INDEX)
    pos_pairs = load_data(BASIC_NET)
    pos_pairs = np.concatenate([pos for _, pos in pos_pairs.items()])
    max_num = max([max_num, pos_pairs.shape[0]])

    triplets = []
    for k in range(max_num):
        pos = pos_pairs[k]
        pass
