import os
import lmdb
from os.path import join
from utils import data_utils
from utils.settings import LMDB_DATA_DIR

# 3GB
map_size = 3221225472

class LMDBClient(object):

    def __init__(self, name, readonly=False):

        os.makedirs(LMDB_DATA_DIR, exist_ok=True)
        self.db = lmdb.open(join(LMDB_DATA_DIR, name), map_size=map_size, readonly=readonly)

    def get(self, key):
        with self.db.begin() as txn:
            value = txn.get(key.encode())
        if value:
            return data_utils.deserialize_embedding(value)
        else:
            return None

    def get_batch(self, keys):
        values = []
        with self.db.begin() as txn:
            for key in keys:
                value = txn.get(key.encode())
                if value:
                    values.append(data_utils.deserialize_embedding(value))
        return values

    def set(self, key, vector):
        with self.db.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), data_utils.serialize_embedding(vector))

    def set_batch(self, generator):
        with self.db.begin(write=True) as txn:
            for key, vector in generator:
                txn.put(key.encode("utf-8"), data_utils.serialize_embedding(vector))
                print(key, self.get(key))