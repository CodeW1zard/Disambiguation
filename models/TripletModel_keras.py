from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam

from utils.settings import *
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
from utils import eval_utils
import numpy as np

class TripletModel():

    def train_triplets_model(self, train_prop=0.8):
        X1, X2, X3 = self.retrieve_data()
        n_triplets = len(X1)
        n_train = int(n_triplets * train_prop)

        model, inter_model = self.create_triplet_model()

        X_anchor, X_pos, X_neg = X1[:n_train], X2[:n_train], X3[:n_train]
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}
        model.fit(X, np.ones((n_train, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)
        model.save(GLOBAL_MODEL)

        test_triplets = (X1[n_train:], X2[n_train:], X3[n_train:])
        eval_utils.full_auc(model, test_triplets)

        loaded_model = load_model(GLOBAL_MODEL)
        print('triplets model loaded')

        auc_score = eval_utils.full_auc(loaded_model, test_triplets)
        print('auc ', auc_score)

    def create_triplet_model(self):
        emb_anchor = Input(shape=(EMB_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM, ), name='neg_input')

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
        margin = K.constant(1)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))

    def accuracy(self, _, y_pred):
        return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])

    def save(self):
        dump_data(self.model, GLOBAL_MODEL)

    def load(self):
        self.model = load_data(GLOBAL_MODEL)

    def retrieve_data(self):
        dataset = CustomDataset()
        anchors = []
        poss = []
        negs = []
        for i in range(len(dataset)):
            anchor, pos, neg = dataset[i]
            anchors.append(anchor)
            poss.append(pos)
            negs.append(neg)
        rnds = np.random.permutation(len(dataset))
        anchors = np.array(anchors)[rnds]
        poss = np.array(poss)[rnds]
        negs = np.array(negs)[rnds]
        print(poss.shape)
        return anchors, poss, negs

class CustomDataset():
    def __init__(self):
        self.triplets = load_data(TRIPLET_INDEX)
        self.cl = LMDBClient(LMDB_WORDVEC)

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