import math
from collections import defaultdict

from utils import string_utils
from utils.settings import *
from utils.lmdb_utils import LMDBClient
from utils.data_utils import deserialize_embedding
from utils.data_utils import dump_data

def transform_feature(data, f_name):
    if type(data) is str:
        data = data.split()
    assert type(data) is list
    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))
    return features


def extract_common_features(item):
    title_features = transform_feature(string_utils.clean_sentence(item["title"], stemming=True), "title")
    keywords_features = []
    keywords = item.get("keywords")
    if keywords:
        keywords_features = transform_feature(' '.join([string_utils.clean_sentence(k) for k in keywords]), 'keyword')
    venue_features = []
    venue_name = item.get('venue', '')
    if len(venue_name) > 2:
        venue_features = transform_feature(string_utils.clean_sentence(venue_name), "venue")
    return title_features, keywords_features, venue_features

# def extract_author_features(item, order=None):
#     title_features, keywords_features, venue_features = extract_common_features(item)
#     for i, author in enumerate(item["authors"]):
#         if order is not None and i != order:
#             continue
#         name_feature = []
#         org_features = []
#         org_name = string_utils.clean_name(author.get("org", ""))
#         org_name = org_name.replace(')', '').replace('(', '').replace(',', ' ')
#         if len(org_name) > 2:
#             org_features.extend(transform_feature(org_name, "org"))
#         for j, coauthor in enumerate(item["authors"]):
#             if i == j:
#                 continue
#             coauthor_name = coauthor.get("name", "")
#             coauthor_org = string_utils.clean_name(coauthor.get("org", ""))
#             if len(coauthor_name) > 2:
#                 name_feature.extend(
#                     transform_feature([string_utils.clean_name(coauthor_name)], "name")
#                 )
#             if len(coauthor_org) > 2:
#                 org_features.extend(
#                     transform_feature(string_utils.clean_sentence(coauthor_org), "org")
#                 )
#         author_feature = name_feature + org_features + title_features + keywords_features + venue_features
#         author_feature = ' '.join(author_feature)
#     return author_feature

def extract_author_features(item):
    title_features, keywords_features, venue_features = extract_common_features(item)

    name_features = []
    org_features = []
    for i, author in enumerate(item['authors']):
        name = string_utils.clean_name(author.get('name', ''))
        name = transform_feature(name, 'name')

        org = string_utils.clean_sentence(author.get('org', ''))
        org = org + ' ' + org.replace(' ', '_')
        org = transform_feature(org, 'org')

        if name:
            name_features.extend(name)
        if org:
            org_features.extend(org)

    title_features = ' '.join(title_features) + ' '
    keywords_features = ' '.join(keywords_features) + ' '
    venue_features = ' '.join(venue_features) + ' '
    name_features = ' '.join(name_features) + ' '
    org_features = ' '.join(org_features) + ' '

    author_features = name_features + org_features + title_features + keywords_features + venue_features
    return author_features

def idf_calc():
    df = defaultdict(int)

    lc = LMDBClient(LMDB_AUTHOR)
    with lc.db.begin() as txn:
        n_doc = txn.stat()['entries']
        for cnt, raw in enumerate(txn.cursor()):
            if (cnt+1)%10000 == 0:
                print('idf_calc %d'%(cnt+1))
            author_feature = deserialize_embedding(raw[1])
            for word in author_feature:
                df[word] += 1

    idf_dict = defaultdict(float, [(word, math.log(n_doc/cnt)) for word, cnt in df.items()])
    dump_data(idf_dict, WORD_IDF)
