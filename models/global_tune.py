import numpy as np
from utils.settings import *
from utils.data_utils import *

def prepare_triplets(max_num = 100000):
    with open(POS_PAIRS, 'r') as f:
        pos_pairs = f.readlines()
    num_pos_pairs = len(pos_pairs)
    max_num = min([max_num, num_pos_pairs])

    triplets = []
    for k in range(max_num):
        pos_id = np.random.randint(0, num_pos_pairs+1)
        pos_pair = pos_pairs[pos_id]
