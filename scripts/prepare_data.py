import pandas as pd

from utils.settings import *
from utils.data_utils import *
from scripts import json2dataframe
from scripts import json2txt
from scripts import txt2lmdb
from scripts.basic_net import find_idf_pos_pairs
from scripts.basic_net import prepare_clusters
from scripts.basic_net import prepare_pos_pairs

from models.generate_triplets import TripletsGenerator
from models.global_embedding import EmbeddingModel, idf_calc

if __name__ == '__main__':
    print('json2txt')
    pubs = load_json(PUBS_JSON)
    json2txt.pubs2txt(pubs, TXT_PATH)
    
    print('json2dataframe')
    json2dataframe.json2dataframe(PUBS_JSON, PUBS_PARQUET)
    json2dataframe.pid2index(PUBS_PARQUET, PID_INDEX)
    
    print('txt2lmdb')
    txt2lmdb.txt2lmdb(TXT_PATH)

    print('calculate idf')
    idf_calc()
    print('word2vec embedding')
    model = EmbeddingModel()
    model.train(EMB_WORD2VEC, size=EMB_DIM)
    model.paper2vec()
    
    print('generate pos pairs and basic net')
    idf = load_data(WORD_IDF)
    pubs = pd.read_parquet(PUBS_PARQUET)
    pubs = dict(list(pubs.groupby('name')))
    pairs = {}
    for name, pub in pubs.items():
        pairs[name] = find_idf_pos_pairs(pub, idf)
        print(name, 'done')
    dump_data(pairs, BASIC_NET)
    prepare_pos_pairs(POS_PAIRS)
    prepare_clusters(BASIC_CLUSTER)
    
    print('generate triplets')
    TG = TripletsGenerator()
    TG.prepare_triplet_pid(max_num=500000)