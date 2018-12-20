from os.path import join
import numpy as np
import scipy.sparse as sp
from utils import settings
from utils.settings import IDF_THRESH

local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESH))

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))


def load_local_data(path=local_na_dir, name='ke_xu'):
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name), 'path=', path)

    idx_features_labels = np.genfromtxt(join(path, "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str)).ravel()
    print(idx_map.keys())
    for node in edges_unordered:
        if node not in idx_map.keys():
            print(node)
    edges = list(map(idx_map.get, edges_unordered))
    edges = np.array(edges, dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, features, labels

if __name__ == '__main__':
    adj, features, labels = load_local_data(name='bing_chen')
    print(adj)
    print(features)
    print(labels)
