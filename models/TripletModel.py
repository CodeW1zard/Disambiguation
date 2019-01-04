import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
from utils import settings
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIPLET_LAYER_SIZE1 = 128
TRIPLET_LAYER_SIZE2 = 64

class TripletModel():
    def __init__(self, margin):
        self.margin = torch.tensor(margin).to(device)
        self.__zero = torch.tensor(0.).to(device).double()

    def criterion(self, anchor, pos, neg):

        dist1 = torch.mean(torch.sqrt(torch.sum((anchor - pos) ** 2, 1)))
        dist2 = torch.mean(torch.sqrt(torch.sum((anchor - neg) ** 2, 1)))
        loss = torch.max(dist1 - dist2 + self.margin, self.__zero)
        return loss

    def train(self, num_epoch=1, learning_rate=1e-3):
        train_dataset = CustomDataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=4,
                                                   shuffle=True)
        self.model = NeuralNet().to(device).double()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        dists1 = []
        dists2 = []
        losses = []
        for epoch in range(num_epoch):
            for i, triplet in enumerate(train_loader):
                anchor, pos, neg = triplet
                encode_anchor = self.model(anchor.to(device))
                encode_pos = self.model(pos.to(device))
                encode_neg = self.model(neg.to(device))


                loss = self.criterion(encode_anchor, encode_pos, encode_neg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                dist1 = torch.mean(torch.sum((encode_anchor - encode_pos) ** 2, 1)).item()
                dist2 = torch.mean(torch.sum((encode_anchor - encode_neg) ** 2, 1)).item()
                dists1.append(dist1)
                dists2.append(dist2)
                losses.append(loss.item())
                if not (i+1)%1000:
                    print('Epoch [{}/{}], Step {}, Loss: {:.4f}'
                          .format(epoch + 1, num_epoch, i + 1, loss.item()))
                if not loss:
                    plt.subplot(121)
                    plt.plot(dists1)
                    plt.plot(dists2)

                    plt.subplot(122)
                    plt.plot(losses)
                    plt.show()
                    return



    def save(self):
        dump_data(self.model, GLOBAL_MODEL)

    def generate_global_emb(self):
        wv_cl = LMDBClient(settings.LMDB_WORDVEC)
        gb_cl = LMDBClient(LMDB_GLOBALVEC)
        with wv_cl.db.begin() as txn:
            for pid, value in txn.cursor():
                pid = pid.decode()
                value = deserialize_embedding(value)
                value = torch.tensor(value).to(device)
                emb = self.model(value).detach().cpu().numpy()
                gb_cl.set(pid, emb)
        print('generate global emb done!')

    def evaluate(self):
        train_dataset = CustomDataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=1,
                                                   shuffle=True)
        dists_before = []
        dists_after = []
        for i, triplet in enumerate(train_loader):
            anchor, pos, neg = triplet
            dist1 = torch.mean(torch.sum((anchor - pos)**2, 1))
            dist2 = torch.mean(torch.sum((anchor - neg)**2, 1))
            dists_before.append([dist1.detach().numpy(), dist2.detach().numpy()])

            encode_anchor = self.model(anchor.to(device))
            encode_pos = self.model(pos.to(device))
            encode_neg = self.model(neg.to(device))

            distance1 = torch.mean(torch.sum((encode_anchor - encode_pos)**2, 1))
            distance2 = torch.mean(torch.sum((encode_anchor - encode_neg)**2, 1))
            dists_after.append([distance1.detach().cpu().numpy(), distance2.detach().cpu().numpy()])

        dists_before = np.array(dists_before)
        plt.subplot(221)
        plt.hist(dists_before[:, 0], color='r', alpha=0.5)
        plt.hist(dists_before[:, 1], color='b', alpha=0.5)
        plt.legend(['pos-anchor', 'neg-anchor'])
        plt.subplot(222)
        plt.hist(dists_before[:, 0] - dists_before[:, 1])

        dists_after = np.array(dists_after)
        plt.subplot(223)
        plt.hist(dists_after[:, 0], color='r', alpha=0.5)
        plt.hist(dists_after[:, 1], color='b', alpha=0.5)
        plt.legend(['pos-anchor', 'neg-anchor'])
        plt.subplot(224)
        plt.hist(dists_after[:, 0] - dists_after[:, 1])
        plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(settings.EMB_DIM, TRIPLET_LAYER_SIZE1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(TRIPLET_LAYER_SIZE1, TRIPLET_LAYER_SIZE2),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        norm = torch.norm(out, dim=1, keepdim=True)
        out = out.div(norm)
        return out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.triplets = load_data(TRIPLET_INDEX)
        self.cl = LMDBClient(settings.LMDB_WORDVEC)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        with self.cl.db.begin() as txn:
            triplet_vec = [deserialize_embedding(txn.get(pid.encode())) for pid in triplet]
        return triplet_vec

    def __len__(self):
        return len(self.triplets)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, help="idf threshold, high or low", type=str)
    args = parser.parse_args()
    mode = args.mode

    if mode == 'high':
        TRIPLET_INDEX = settings.TRIPLET_INDEX_HIGH
        GLOBAL_MODEL_H5 = settings.GLOBAL_MODEL_H5_HIGH
        GLOBAL_MODEL_JSON = settings.GLOBAL_MODEL_JSON_HIGH
        LMDB_GLOBALVEC = settings.LMDB_GLOBALVEC_HIGH
    elif mode == 'low':
        TRIPLET_INDEX = settings.TRIPLET_INDEX_LOW
        GLOBAL_MODEL_H5 = settings.GLOBAL_MODEL_H5_LOW
        GLOBAL_MODEL_JSON = settings.GLOBAL_MODEL_JSON_LOW
        LMDB_GLOBALVEC = settings.LMDB_GLOBALVEC_LOW
    else:
        print('wrong mode!')
        raise ValueError
    model = TripletModel(margin=10)
    model.train()
    # model.save()
    # model.evaluate()