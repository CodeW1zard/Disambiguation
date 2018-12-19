import os
from os.path import join
from numpy.random import shuffle
from utils.lmdb_utils import LMDBClient
from utils import data_utils
from utils import settings

def gen_local_data(idf_threshold=settings.IDF_THRESH):
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    name_to_pubs_test = data_utils.load_data(settings.BASIC_CLUSTER)
    pid_dict = data_utils.load_data(settings.PID_INDEX)
    lc_inter = LMDBClient(settings.LMDB_GLOBALVEC)
    pos_pairs = data_utils.load_data(settings.BASIC_NET)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        cur_person_dict = name_to_pubs_test[name]
        pid_index = pid_dict[name]
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for aid, items in enumerate(cur_person_dict):
            # if len(items) < 5:
            #     continue
            for index in items:
                pids2label[pid_index[index]] = aid
                pids.append(pid_index[index])
        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()

        pairs = pos_pairs[name]
        with open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w') as f:
            pid_index = pid_dict[name]
            for i, j in pairs:
                f.write(pid_index[i] + '\t' + pid_index[j] + '\n')
        print('prepare local data', name, 'done')

if __name__ == '__main__':
    gen_local_data(idf_threshold=settings.IDF_THRESH)
    print('done')
