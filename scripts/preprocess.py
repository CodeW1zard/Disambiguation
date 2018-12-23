from utils.settings import *
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
from utils.feature_utils import extract_author_features, idf_calc
from models.global_embedding import EmbeddingModel

from itertools import chain
from datetime import datetime 

def pubs2txt(rfpath, wfpath):
    start_time = datetime.now()
    pubs = load_json(rfpath)
    with open(wfpath, 'w', encoding='utf-8') as wf:
        for cnt, paper in enumerate(chain.from_iterable(pubs.values())):
                if not (cnt+1)%1000:
                    print('json2txt %d  '%(cnt+1), datetime.now() - start_time)
                # n_authors = len(paper.get('authors', []))
                # if n_authors > 100:
                #     continue
                pid = paper['id']
                line = extract_author_features(paper)
                wf.write(pid + '\t' + line + '\n')


def txt2lmdb(rfpath):
    start_time = datetime.now()
    client = LMDBClient(LMDB_AUTHOR)
    with open(rfpath, 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if not (i+1)%5000:
                print('txt2lmdb %d  '%(i+1), datetime.now()-start_time)
            pid, paper = line.rstrip().split('\t')
            client.set(pid, paper.split())

def dump_author_embs():
    start_time = datetime.now()
    model = EmbeddingModel()
    model.train(EMB_WORD2VEC, size=EMB_DIM)
    model.paper2vec()
    print('dump author embs done ', datetime.now()-start_time)

if __name__ == '__main__':
    start_time = datetime.now()
    pubs2txt(PUBS_JSON, TXT_PATH)
    txt2lmdb(TXT_PATH)
    idf_calc()
    dump_author_embs()
    print('prepocess done ', datetime.now()-start_time)






