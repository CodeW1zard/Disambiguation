import torch
import torch.nn as nn
import torch.utils.data
from utils.settings import *
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIPLET_LAYER_SIZE1 = 128
TRIPLET_LAYER_SIZE2 = 64
BATCH_SIZE = 4
class TripletModel():
    def __init__(self):
        self.__zero = torch.zeros(BATCH_SIZE).to(device).long()
        self.__one = torch.ones(BATCH_SIZE).to(device).long()

    def scale(self, tensor):
        mean = torch.mean(tensor, dim=1, keepdim=True)
        std = torch.std(tensor, dim=1, keepdim=True)
        return (tensor-mean).div(std)

    def train(self, num_epoch=1, learning_rate=1e-3):
        train_dataset = CustomDataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)
        self.model = NeuralNet().to(device).double()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(num_epoch):
            for i, triplet in enumerate(train_loader):
                anchor, pos, neg = triplet

                anchor = self.scale(anchor)
                pos = self.scale(pos)
                neg = self.scale(neg)

                anchor_pos, anchor_neg = \
                    self.model(anchor.to(device), pos.to(device), neg.to(device))

                loss = criterion(anchor_pos, self.__one) + (criterion(anchor_neg, self.__zero))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if not (i+1)%1000:
                    print('Epoch [{}/{}], Step {}, Loss: {:.4f}'
                          .format(epoch + 1, num_epoch, i + 1, loss.item()))
        plt.plot(losses)
        plt.show()
        return



    def save(self):
        dump_data(self.model, GLOBAL_MODEL)

    def generate_global_emb(self):
        wv_cl = LMDBClient(LMDB_WORDVEC)
        gb_cl = LMDBClient(LMDB_GLOBALVEC)
        with wv_cl.db.begin() as txn:
            for pid, value in txn.cursor():
                pid = pid.decode()
                value = deserialize_embedding(value)
                value = torch.tensor(value).to(device)
                emb = self.model(value).detach().cpu().numpy()
                gb_cl.set(pid, emb)
        print('generate global emb done!')

    def load(self):
        self.model = load_data(GLOBAL_MODEL)

    def evaluate(self):
        train_dataset = CustomDataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)
        dists_before = []
        dists_after = []
        for i, triplet in enumerate(train_loader):
            anchor, pos, neg = triplet
            anchor = self.scale(anchor)
            pos = self.scale(pos)
            neg = self.scale(neg)

            distance1 = torch.mean(torch.sum((anchor - pos)**2, 1))
            distance2 = torch.mean(torch.sum((anchor - neg)**2, 1))
            dists_before.append([distance1.detach().cpu().numpy(), distance2.detach().cpu().numpy()])

            encode_anchor = self.model.layer(anchor.to(device))
            encode_pos = self.model.layer(pos.to(device))
            encode_neg = self.model.layer(neg.to(device))

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
        self.layer = nn.Sequential(
            nn.Linear(EMB_DIM, TRIPLET_LAYER_SIZE1),
            nn.ReLU(),
            nn.Linear(TRIPLET_LAYER_SIZE1, TRIPLET_LAYER_SIZE2),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(TRIPLET_LAYER_SIZE2, 2),
            nn.Sigmoid())

    def forward(self, anchor, pos, neg):
        out_anchor = self.layer(pos)
        out_pos = self.layer(pos)
        out_neg = self.layer(neg)
        # out_anchor = out_anchor.div(torch.norm(out_anchor, dim=1, keepdim=True))
        # out_pos = out_pos.div(torch.norm(out_pos, dim=1, keepdim=True))
        # out_neg = out_neg.div(torch.norm(out_neg, dim=1, keepdim=True))
        anchor_pos = out_anchor - out_pos
        anchor_neg = out_anchor - out_neg
        out_anchor_pos = self.fc(anchor_pos)
        out_anchor_neg = self.fc(anchor_neg)
        return out_anchor_pos, out_anchor_neg

class CustomDataset(torch.utils.data.Dataset):
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
    model.train(num_epoch=10)
    # model.save()
    # model.load()
    model.evaluate()