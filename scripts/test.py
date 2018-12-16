from utils.settings import *
from utils.lmdb_utils import LMDBClient
from utils.data_utils import *
import pandas as pd

if __name__ == '__main__':
    # pubs = pd.read_parquet(PUBS_PARQUET)
    # cl = LMDBClient(LMDB_AUTHOR)
    # with cl.db.begin() as txn:
    #     f = deserialize_embedding(txn.get(pubs.iloc[0].id.encode()))
    #     print(f)

    idf = load_data(WORD_IDF)
    for i, (k, v) in enumerate(idf.items()):
        if i<100:
            print(k, v)
