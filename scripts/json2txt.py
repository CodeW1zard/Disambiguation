from utils.data_utils import *
from utils.feature_utils import extract_author_features
from utils.settings import *

def pubs2txt(pubs, fpath):
    with open(fpath, 'w', encoding='utf-8') as wf:
        for name, pub in pubs.items():
            for paper in pub:
                line = extract_author_features(paper)
                wf.write(paper['id'] + '\t' + line + '\n')
            print(name, 'done')

if __name__ == '__main__':
    pubs = load_json(PUBS_JSON)
    pubs2txt(pubs, TXT_PATH)




