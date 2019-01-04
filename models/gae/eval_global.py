import time
import codecs
import numpy as np

from models.gae.input_data import load_local_data
from models.gae.preprocessing import normalize_vectors
from utils.cluster import clustering
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings
from utils import lmdb_utils
from os.path import join

def load_test_names():
    with open(settings.NAME_LIST, 'r') as f:
        names = [name.split('\n')[0] for name in f]
    return names

def gae_for_na(name):
    adj, features, labels = load_local_data(local_na_dir, name=name)
    features = normalize_vectors(features)
    n_clusters = len(set(labels))
    num_nodes = adj.shape[0]
    clusters_pred = clustering(features, num_clusters=n_clusters)
    prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))
    return [prec, rec, f1], num_nodes, n_clusters

def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'global_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name, n_pubs, n_clusters, precision, recall, f1\n')
    metrics = np.zeros(3)
    cnt = 0
    for name in names:
        cur_metric, num_nodes, n_clusters = gae_for_na(name)
        wf.write('{0}, {1}, {2}, {3:.5f}, {4:.5f}, {5:.5f}\n'.format(
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
        wf.flush()
        for i, m in enumerate(cur_metric):
            metrics[i] += m
        cnt += 1
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = cal_f1(macro_prec, macro_rec)
        print('average until now', [macro_prec, macro_rec, macro_f1])
        time_acc = time.time()-start_time
        print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)
    macro_prec = metrics[0] / cnt
    macro_rec = metrics[1] / cnt
    macro_f1 = cal_f1(macro_prec, macro_rec)
    wf.write('average {0:.5f}  {1:.5f}  {2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))
    wf.close()

if __name__ == '__main__':
    IDF_THRESH = settings.IDF_THRESH_HIGH
    LMDB_LOCAL_EMB = settings.LMDB_LOCAL_EMB
    local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESH))
    cl = lmdb_utils.LMDBClient(LMDB_LOCAL_EMB)
    start_time = time.time()
    main()
