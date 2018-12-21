import os
from os.path import abspath, dirname, join
###########################
SUFFIX = 'train'
IDF_THRESH = 0.2
###########################

PROJ_DIR = join(abspath(dirname(__file__)), '..')

OUT_DIR = join(PROJ_DIR, 'out')
DATA_DIR = join(PROJ_DIR, 'data')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
LMDB_DATA_DIR = join(DATA_DIR, 'lmdb')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)
os.makedirs(LMDB_DATA_DIR, exist_ok=True)
os.makedirs(GLOBAL_DATA_DIR, exist_ok=True)

TXT_PATH = join(EMB_DATA_DIR, 'materials_' + SUFFIX + '.txt')
WORD_IDF = join(EMB_DATA_DIR, 'idf_' + SUFFIX +'.pkl')
EMB_WORD2VEC = join(EMB_DATA_DIR, 'w2v_embs_' + SUFFIX + '.pkl')

PUBS_JSON = join(GLOBAL_DATA_DIR, 'pubs_' + SUFFIX + '.json')
PUBS_PARQUET = join(GLOBAL_DATA_DIR, 'pubs_' + SUFFIX + '.parquet')
PID_INDEX = join(GLOBAL_DATA_DIR, 'pid2index_' + SUFFIX + '.pkl')

BASIC_NET = join(GLOBAL_DATA_DIR, 'basic_net_' + SUFFIX + '_%f'%(IDF_THRESH) +'.pkl')
POS_PAIRS = join(GLOBAL_DATA_DIR, 'pos_pairs_' + SUFFIX + '_%f'%(IDF_THRESH) + '.txt')
BASIC_CLUSTER = join(GLOBAL_DATA_DIR, 'basic_clusters_' + SUFFIX + '_%f'%(IDF_THRESH) + '.pkl')
TRIPLET_INDEX = join(GLOBAL_DATA_DIR, 'triplet_index_' + SUFFIX + '_%f'%(IDF_THRESH) + '.pkl')
GLOBAL_MODEL = join(OUT_DIR, 'global_model_' + SUFFIX + '_%f'%(IDF_THRESH) + '.h5')

ASSIGNMENT_JSON = join(GLOBAL_DATA_DIR, 'assignment_' + SUFFIX + '.json')
NAME_LIST = join(DATA_DIR, 'name_list_' + SUFFIX + '.txt')
CLUSTER_SIZE = join(OUT_DIR, 'n_clusters_rnn_' + SUFFIX + '.txt')

LMDB_AUTHOR = 'pub_authors_' + SUFFIX + '.feature'
LMDB_WORDVEC = 'pub_vectors_' + SUFFIX + '.feature'
LMDB_GLOBALVEC = 'pub_globalvec_' + SUFFIX + '.feature'
LMDB_LOCAL_EMB = 'pub_localvec_' + SUFFIX + '.feature'
EMB_DIM = 100









