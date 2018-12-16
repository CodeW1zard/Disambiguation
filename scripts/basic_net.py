import pandas as pd
import numpy as np
from itertools import combinations
from utils.settings import *
from utils.data_utils import deserialize_embedding, load_data, dump_data
from utils.lmdb_utils import LMDBClient
from utils.string_utils import clean_name
def match_author(x_list, y_list, name):
    return any(x in x_list for x in y_list if x != name)

def match_org(x_list, y_list):
    return any(x in x_list for x in y_list if x != '')

def find_strong_pos_pairs(pub):
    name = pub.loc[:, 'name'].values[0]
    name = clean_name(name)
    num_paper = pub.shape[0]

    author_lists = pub.loc[:, 'authors']
    org_lists = pub.loc[:, 'org']

    match = lambda i, j: match_author(author_lists[i], author_lists[j], name) and match_org(org_lists[i], org_lists[j])
    pairs = np.asarray([(i, j) for i, j in combinations(range(num_paper), 2) if match(i, j)])
    return pairs

def find_idf_pos_pairs(pub, idf, thresh=0.35):
    name = pub.loc[:, 'name'].values[0]
    name = clean_name(name)
    num_paper = pub.shape[0]
    corr = np.zeros((num_paper, num_paper))
    ids = pub.loc[:, 'id'].values
    client = LMDBClient(LMDB_AUTHOR)
    name = '__NAME__' + name
    with client.db.begin() as txn:
        papers = [0] * num_paper
        papers_idf = [0] * num_paper
        for i in range(num_paper):
            papers[i] = deserialize_embedding(txn.get(ids[i].encode()))
            papers_idf[i] = sum([idf[word] for word in papers[i] if word!=name])

        for i, j in combinations(range(num_paper), 2):
            idf12 = sum([idf[word] for word in papers[i] if word in papers[j] and word!=name])
            corr[i, j] = idf12/np.min([papers_idf[i], papers_idf[j]])
        
        pairs = np.asarray(np.where(corr > thresh)).transpose()
    return pairs

if __name__ == '__main__':
    idf = load_data(WORD_IDF)
    pubs = pd.read_parquet(PUBS_PARQUET)
    pubs = dict(list(pubs.groupby('name')))
    pairs = {}
    for name, pub in pubs.items():
        pairs[name] = find_idf_pos_pairs(pub, idf)
        print(name, 'done')
    dump_data(pairs, BASIC_NET)