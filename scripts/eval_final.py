from itertools import chain
from utils import settings
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
from utils.eval_utils import pairwise_precision_recall_f1
from utils.cluster import clustering

cl = LMDBClient(settings.LMDB_LOCAL_EMB)
#cl = LMDBClient(settings.LMDB_GLOBALVEC)
assignments = load_json(settings.ASSIGNMENT_JSON)
n_clusters = 100

for i, (name, clusters) in enumerate(assignments.items()):
    print(name)
    pids = chain.from_iterable(clusters)
    true = chain.from_iterable([[i] * len(cluster) for i, cluster in enumerate(clusters)])
    true = list(true)
    embs = cl.get_batch(pids)
    preds = clustering(embs, num_clusters=len(clusters))
    print(len(preds), len(true))
    score = pairwise_precision_recall_f1(preds, true)
    print(score)
	
