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
import time

# from Trainer import Trainer

class Conv_Net(nn.Module):
    def __init__(self, lr: float = 0.00085, device: str = None):
        super().__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'gpu') if device is None else device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 2)

        self.fc1 = nn.Linear(16 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.ID = "Conv_Net"

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Output size = (6-2)/2 + 1 = 3
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # Output size = (3-2)/2 + 1 = 1
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        

    def train_step(self, X, y) -> tuple:
        'Train the model on a batch of samples and return the predictions and loss'

        X = X.to(self.device)
        pred = self(X).to(self.device)

        outcome = y.to(self.device)

        self.optimizer.zero_grad()

        loss = self.criterion(pred, outcome)
        loss.backward()

        self.optimizer.step()

        prob = F.softmax(pred, dim=1)
        conf = prob[torch.argmax(pred, dim=1)]
        return torch.argmax(pred, dim=1), loss.item(), conf




