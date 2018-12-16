import pandas as pd
from utils.settings import *
from utils.data_utils import load_json, dump_data
from pandas.io.json import json_normalize

def json2dataframe(rfpath, wfpath):
    pubs = load_json(rfpath=rfpath)
    names = []
    values = []
    for k, v in pubs.items():
        names.extend([k] * len(v))
        values.extend(v)
    values = json_normalize(values)
    values['name'] = names
    pubs = values
    pubs['org'] = pubs.authors.map(lambda x: list(map(lambda x: x['org'], x)))
    pubs['authors'] = pubs.authors.map(lambda x: list(map(lambda x: x['name'], x)))
    pubs.to_parquet(wfpath, engine='fastparquet')

def pid2index(rfpath, wfpath):
    pubs = pd.read_parquet(rfpath)
    index = {}
    for name, pub in pubs.groupby('name'):
        index[name] = pub.loc[:, 'id'].values
    dump_data(index, wfpath=wfpath)

if __name__ == '__main__':
    json2dataframe(PUBS_JSON, PUBS_PARQUET)
    pid2index(PUBS_PARQUET, PID_INDEX)