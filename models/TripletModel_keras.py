import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam

from utils import settings
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
from utils import eval_utils


class TripletModel():

    def train_triplets_model(self, train_prop=0.85):
        X1, X2, X3 = self.retrieve_data()
        n_triplets = len(X1)
        n_train = int(n_triplets * train_prop)

        self.model, self.inter_model = self.create_triplet_model()

        X_anchor, X_pos, X_neg = X1[:n_train], X2[:n_train], X3[:n_train]
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}
        self.model.fit(X, np.ones((n_train, 2)), batch_size=64, epochs=15, shuffle=True, validation_split=0.1)


        test_triplets = (X1[n_train:], X2[n_train:], X3[n_train:])
        eval_utils.full_auc(self.model, test_triplets)

    def create_triplet_model(self):
        emb_anchor = Input(shape=(settings.EMB_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(settings.EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(settings.EMB_DIM, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(self.l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(self.euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(self.euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        model.compile(loss=self.triplet_loss, optimizer=Adam(lr=0.01), metrics=[self.accuracy])

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))

        return model, inter_layer

    def l2Norm(self, x):
        return K.l2_normalize(x, axis=-1)

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def triplet_loss(self, _, y_pred):
        margin = K.constant(settings.MARGIN)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))

    def accuracy(self, _, y_pred):
        return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])

    def save(self):
        model_json = self.model.to_json()
        with open(settings.GLOBAL_MODEL_JSON, 'w') as wf:
            wf.write(model_json)
        self.model.save_weights(settings.GLOBAL_MODEL_H5)

    def load(self):
        rf = open(settings.GLOBAL_MODEL_JSON, 'r')
        model_json = rf.read()
        rf.close()
        self.model = model_from_json(model_json)
        self.model.load_weights(settings.GLOBAL_MODEL_H5)

    def retrieve_data(self):
        dataset = CustomDataset()
        anchors = []
        poss = []
        negs = []
        for data in dataset:
            anchor, pos, neg = data
            if np.isnan(anchor).any() or np.isnan(pos).any() or np.isnan(neg).any():
                continue
            anchors.append(anchor)
            poss.append(pos)
            negs.append(neg)
        rnds = np.random.permutation(len(anchors))
        anchors = np.array(anchors)[rnds]
        poss = np.array(poss)[rnds]
        negs = np.array(negs)[rnds]
        print(poss.shape)
        return anchors, poss, negs

    def generate_global_emb(self):
        wv_cl = LMDBClient(settings.LMDB_WORDVEC)
        gb_cl = LMDBClient(settings.LMDB_GLOBALVEC)
        values = []
        nan_pids = []
        pids = []
        with wv_cl.db.begin() as txn:
            for pid, value in txn.cursor():
                value = deserialize_embedding(value)
                if np.isnan(value).any():
                    nan_pids.append(pid.decode())
                    continue
                pids.append(pid.decode())
                values.append(value)
        values = np.stack(values)
        inter_embs = eval_utils.get_hidden_output(self.model, values)
        for i, pid in enumerate(pids):
            gb_cl.set(pid, inter_embs[i])
        for pid in nan_pids:
            gb_cl.set(pid, None)
        print('generate global emb done!')

class CustomDataset():
    def __init__(self):
        self.triplets = load_data(settings.TRIPLET_INDEX)
        self.cl = LMDBClient(settings.LMDB_WORDVEC)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        with self.cl.db.begin() as txn:
            triplet_vec = [deserialize_embedding(txn.get(pid.encode())) for pid in triplet]
        return triplet_vec

    def __len__(self):
        return len(self.triplets)

if __name__=='__main__':
    model = TripletModel()
    model.train_triplets_model()
    model.save()
    # model.load()
    model.generate_global_emb()
