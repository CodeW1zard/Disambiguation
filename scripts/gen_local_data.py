import os
import argparse
from os.path import join
from numpy.random import shuffle
from utils.lmdb_utils import LMDBClient
from utils import data_utils
from utils import settings

def gen_local_data(idf_threshold, global_vec, basic_net):
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    # 这里一应该读入ASSIGNMENT_JSON 但由于竞赛方给的数据缺失，所以只好采用聚类结构BASIC_CLUSTER
    # name_to_pubs_test = data_utils.load_json(settings.ASSIGNMENT_JSON)
    name_to_pubs_test = data_utils.load_json(settings.ASSIGNMENT_JSON)
    pid_dict = data_utils.load_data(settings.PID_INDEX)
    lc_inter = LMDBClient(global_vec)
    pos_pairs = data_utils.load_data(basic_net)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        cur_person_dict = name_to_pubs_test[name]
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for aid, items in enumerate(cur_person_dict):
            # if len(items) < 5:
            #     continue
            # for index in items:
            #      pids2label[pid_index[index]] = aid
            #      pids.append(pid_index[index])
            for pid in items:
                pids2label[pid] = aid
                pids.append(pid)
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
                pid1 = pid_index[i]
                pid2 = pid_index[j]
                if pid1 in pids and pid2 in pids:
                    f.write(pid1 + '\t' + pid2 + '\n')
        print('prepare local data', name, 'done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, help="idf threshold, high or low", type=str)
    args = parser.parse_args()
    mode = args.mode
    if mode == 'high':
        idf_thresh = settings.IDF_THRESH_HIGH
        basic_net = settings.BASIC_NET_HIGH
        global_vec = settings.LMDB_GLOBALVEC_HIGH
    elif mode=='low':
        idf_thresh = settings.IDF_THRESH_LOW
        basic_net = settings.BASIC_NET_LOW
        global_vec = settings.LMDB_GLOBALVEC_HIGH
    else:
        print('wrong mode error!')
        raise ValueError

    gen_local_data(idf_threshold=idf_thresh, global_vec=global_vec, basic_net=basic_net)
    print('done')
