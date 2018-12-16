from utils.settings import *
from utils.lmdb_utils import LMDBClient

def txt2lmdb(fpath):
    client = LMDBClient(LMDB_AUTHOR)
    with open(fpath, 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if not (i+1)%5000:
                print('txt2lmdb %d'%(i+1))
            pid, paper = line.rstrip().split('\t')
            client.set(pid, paper.split())

if __name__ == '__main__':
    txt2lmdb(TXT_PATH)




