from random import shuffle
from gensim.models import Word2Vec
from utils.settings import *
from utils.lmdb_utils import LMDBClient
from utils.data_utils import deserialize_embedding
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

if __name__ == '__main__':
    # EmbeddingModel().train(EMB_WORD2VEC, size=EMB_DIM)
    idf_calc()


