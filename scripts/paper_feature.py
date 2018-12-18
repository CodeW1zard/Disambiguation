from utils.settings import *
from utils.lmdb_utils import LMDBClient
from utils.data_utils import *

def create_paper_feature_database():
    client = LMDBClient(LMDB_FEATURE)

    client.set(pid, paper.split())
