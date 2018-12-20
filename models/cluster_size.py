from os.path import join
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from utils.lmdb_utils import LMDBClient
from utils import data_utils
from utils import settings

from itertools import chain

lc = LMDBClient(settings.LMDB_GLOBALVEC)

data_cache = {}


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def create_model(k=300):
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(k, 64)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])

    return model


def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False):
    xs, ys = [], []
    for b in range(batch_size):
        num_clusters = np.random.randint(min, max)
        sampled_clusters = np.random.choice(len(clusters), num_clusters, replace=False)
        items = []
        for c in sampled_clusters:
            items.extend(clusters[c])
        x = []
        sampled_points = []
        while len(sampled_points) <= k:
            p = np.random.choice(items)
            emb = lc.get(p)
            if emb is not None:
                x.append(emb)
        # sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        # for p in sampled_points:
        #     if p in data_cache:
        #         x.append(data_cache[p])
        #     else:
        #         print("a")
        #         v = lc.get(p)
        #         if not v:
        #             x.append(v)
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    return np.stack(xs), np.stack(ys)


def gen_train(clusters, k=300, batch_size=1000, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, flatten=flatten)


def gen_test(name_to_pubs_test, k=300, flatten=False):
    # name_to_pubs_test = data_utils.load_data(settings.BASIC_CLUSTER)
    pid_dict = data_utils.load_data(settings.PID_INDEX)
    xs, ys = [], []
    names = []
    for name in name_to_pubs_test:
        pid_index = pid_dict[name]
        names.append(name)
        num_clusters = len(name_to_pubs_test[name])
        x = []
        items = list(chain.from_iterable(name_to_pubs_test[name]))
        sampled_points = []
        while len(sampled_points) <= k:
            p = pid_index[np.random.choice(items)]
            emb = lc.get(p)
            if emb is not None:
                x.append(emb)
        # sampled_points = np.random.choice(items, k, replace=True)
        # sampled_points = [pid_index[ind] for ind in sampled_points]
        #
        # for p in sampled_points:
        #     if p in data_cache:
        #         x.append(data_cache[p])
        #     else:
        #         v = lc.get(p)
        #         if not v:
        #             x.append(v)
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def run_rnn(k=300, seed=1106, split=0.9):
    np.random.seed(seed)
    name_to_pubs = data_utils.load_json(settings.ASSIGNMENT_JSON)
    names = list(name_to_pubs.keys())
    num_train = int(len(names) * split)
    names_train = names[:num_train]
    name_to_pubs_test = dict((name, item) for name, item in name_to_pubs.items() if name not in names_train)

    clusters = []
    for name, pubs in name_to_pubs.items():
        if name not in names_train:
            continue
        clusters.extend(pubs)
    # for i, c in enumerate(clusters):
    #     if i % 100 == 0:
    #         print(i, len(c), len(clusters))
    #     for pid in c:
    #         v = lc.get(pid)
    #         if not v:
    #             data_cache[pid] = v
    # print(model.summary())

    model = create_model(k=k)
    test_names, test_x, test_y = gen_test(name_to_pubs_test, k=k)
    model.fit_generator(gen_train(clusters, k=k, batch_size=1000), steps_per_epoch=100, epochs=1000,
                        validation_data=(test_x, test_y))
    kk = model.predict(test_x)
    wf = open(join(settings.CLUSTER_SIZE), 'w')
    for i, name in enumerate(test_names):
        wf.write('{}\t{}\t{}\n'.format(name, test_y[i], kk[i][0]))
    wf.close()


if __name__ == '__main__':
    run_rnn()