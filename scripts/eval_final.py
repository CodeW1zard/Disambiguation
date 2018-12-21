from itertools import chain
from utils import settings
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
from utils.eval_utils import pairwise_precision_recall_f1
from utils.cluster import clustering

cl = LMDBClient(settings.LMDB_LOCAL_EMB)
assignments = load_json(settings.ASSIGNMENT_JSON)
n_clusters = 100

for i, (name, clusters) in enumerate(assignments.items()):
    if i:
        continue
    pids = chain.from_iterable(clusters)
    embs = cl.get_batch(pids)
    preds = clustering(embs, num_clusters=n_clusters)
    score = pairwise_precision_recall_f1(preds, clusters)
print(score)
