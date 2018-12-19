import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne(features):
    '''
    :param features: N * D
    :return: N * D'
    '''
    return TSNE().fit_transform(features)

def tsne_visualize(features, labels):
    assert features.shape[0] == len(labels)
    embs = tsne(features)
    plt.scatter(embs[:, 0], embs[:, 1], c=labels)