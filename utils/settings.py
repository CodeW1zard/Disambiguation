import os
from os.path import abspath, dirname, join, pardir
###########################
SUFFIX = 'validate'
###########################

PROJ_DIR = abspath(join(dirname('__file__'), pardir))
# PROJ_DIR = 'E:\\MyRepo\\Projects\\Disambiguation'
OUT_DIR = join(PROJ_DIR, 'out')
DATA_DIR = join(PROJ_DIR, 'data')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
LMDB_DATA_DIR = join(DATA_DIR, 'lmdb')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)
os.makedirs(LMDB_DATA_DIR, exist_ok=True)

TXT_PATH = join(EMB_DATA_DIR, 'materials_' + SUFFIX + '.txt')
PUBS_JSON = join(GLOBAL_DATA_DIR, 'pubs_' + SUFFIX + '.json')
PUBS_PARQUET = join(GLOBAL_DATA_DIR, 'pubs_' + SUFFIX + '.parquet')
WORD_IDF = join(EMB_DATA_DIR, 'idf_' + SUFFIX +'.pkl')
EMB_WORD2VEC = join(EMB_DATA_DIR, 'w2v_embs_' + SUFFIX + '.pkl')
PID_INDEX = join(GLOBAL_DATA_DIR, 'pid2index_' + SUFFIX + '.pkl')
BASIC_NET = join(GLOBAL_DATA_DIR, 'basic_net_' + SUFFIX + '.pkl')


LMDB_AUTHOR = 'pub_authors_' + SUFFIX + '.feature'
EMB_DIM = 100






