import numpy as np

from random import shuffle
from gensim.models import Word2Vec
from utils.settings import *
from utils.lmdb_utils import LMDBClient
from utils.data_utils import *
from utils.feature_utils import idf_calc
class EmbeddingModel():
    def __init__(self):
        self.model = None

    def train(self, wfpath, size=EMB_DIM):
        data = []
        lc = LMDBClient(LMDB_AUTHOR)
        with lc.db.begin() as txn:
            for cnt, raw in enumerate(txn.cursor()):
                author_feature = deserialize_embedding(raw[1])
                if (cnt+1)%10000 == 0:
                    print('word2vec:', cnt+1)
                    shuffle(author_feature)
                data.append(author_feature)
        self.model = Word2Vec(data, size=size, window=5, min_count=5, workers=10)
        self.model.save(wfpath)

    def load(self, wfpath):
        self.model = Word2Vec.load(wfpath)

    def paper2vec(self):
        paper_lc = LMDBClient(LMDB_AUTHOR)
        vec_lc = LMDBClient(LMDB_WORDVEC)
        idf_dict = load_data(WORD_IDF)
        with paper_lc.db.begin() as txn:
            for k, paper in enumerate(txn.cursor()):
                if (k+1)%10000==0:
                    print('paper2vec', k+1)
                pid = paper[0].decode()
                paper_features = deserialize_embedding(paper[1])
                paper_features = [feature for feature in paper_features if feature in self.model.wv.vocab]
                idfs = np.asarray([idf_dict[feature] for feature in paper_features])
                vecs = np.asarray([self.model.wv[feature] for feature in paper_features]).transpose()
                vec = vecs @ idfs/np.sum(idfs)
                vec_lc.set(pid, vec)




if __name__ == '__main__':
    # EmbeddingModel().train(EMB_WORD2VEC, size=EMB_DIM)
    # idf_calc()
    model = EmbeddingModel()
    model.load(EMB_WORD2VEC)
    model.paper2vec()


