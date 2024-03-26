from sklearn import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.cuda
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from Conv_Model import Trainer, dataprep

class Linear_Net(nn.Module):
    def __init__(self, lr: float = 0.00085, device: str = None):
        super().__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'gpu') if device is None else device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 256)
        self.lin3 = nn.Linear(256, 64)
        self.lin4 = nn.Linear(64, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.ID = "Linear_Net"

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x
        

    def train_step(self, X, y) -> tuple:
        'Train the model on a single sample and return the prediction and loss'

        X = X.flatten()
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        pred = self(X).unsqueeze(0).to(self.device)

        outcome = torch.tensor([y], dtype=torch.long).to(self.device)

        self.optimizer.zero_grad()

        loss = self.criterion(pred, outcome)
        loss.backward()

        self.optimizer.step()

        prob = F.softmax(pred, dim=1)
        conf = prob[0][torch.argmax(pred)]
        return torch.argmax(pred).item(), loss.item(), conf.item()
    
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = dataprep(Linear_Net())

    rtt = 3

    trainer = Trainer(Linear_Net, X_train, y_train, X_test, y_test)
    trainer.train(epochs=rtt)
    acc = trainer.test()
    print(f"Accuracy: {acc}")