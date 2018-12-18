import torch
import torch.nn as nn
import torch.utils.data
from utils.settings import *
from utils.data_utils import *
from utils.lmdb_utils import LMDBClient
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIPLET_LAYER_SIZE1 = 128
TRIPLET_LAYER_SIZE2 = 64

class TripletModel():
    def __init__(self, margin):
        self.margin = torch.tensor(margin).to(device)
        self.__zero = torch.tensor(0.).to(device).double()

    def criterion(self, anchor, pos, neg):
        # print(anchor.shape, pos.shape, neg.shape)
        loss = torch.sum((anchor - pos)**2) - torch.sum((anchor - neg)**2) + self.margin
        # print(loss.item())
        loss = torch.max(loss, self.__zero)
        return loss

    def train(self, num_epoch=10, learning_rate=1e-2):
        train_dataset = CustomDataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=4,
                                                   shuffle=True)
        self.model = NeuralNet().to(device).double()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epoch):
            for i, triplet in enumerate(train_loader):
                anchor, pos, neg = triplet
                encode_anchor = self.model(anchor.to(device))
                encode_pos = self.model(pos.to(device))
                encode_neg = self.model(neg.to(device))

                loss = self.criterion(encode_anchor, encode_pos, encode_neg)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not (i+1)%1000:
                    print('Epoch [{}/{}], Step {}, Loss: {:.4f}'
                          .format(epoch + 1, num_epoch, i + 1, loss.item()))

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

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(EMB_DIM, TRIPLET_LAYER_SIZE1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(TRIPLET_LAYER_SIZE1, TRIPLET_LAYER_SIZE2),
            nn.ReLU())
        self.norm_layer = nn.LayerNorm(TRIPLET_LAYER_SIZE2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.norm_layer(out)
        return out

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.triplets = load_data(TRIPLET_INDEX)
        self.cl = LMDBClient(LMDB_WORDVEC)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        triplet = self.triplets[index]
        with self.cl.db.begin() as txn:
            triplet_vec = [deserialize_embedding(txn.get(pid.encode())) for pid in triplet]
        return triplet_vec

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.triplets)

if __name__=='__main__':
    model = TripletModel(margin=1000)
    model.train()