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
        'Train the model on a single sample and return the prediction and loss'
        X = torch.tensor(X[0], dtype=torch.float).unsqueeze(0)
        X = X.view(1, 1, 8, 8).to(self.device)  # Reshape to (batch_size, num_channels, height, width)
        pred = self(X).to(self.device)
        temp_out = torch.zeros(1, 10)
        temp_out[0][int(y)] = 1
        outcome = temp_out.to(self.device)

        self.optimizer.zero_grad()

        loss = self.criterion(pred, outcome)
        loss.backward()

        self.optimizer.step()
        conf = pred[0][torch.argmax(pred).item()]
        

        return torch.argmax(pred).item(), loss.item(), conf.item()

class Trainer:
    def __init__(self, 
                 net: type, X_train, y_train, X_test, y_test, lr: float = None, device: str = None, 
                 parallel: bool = False):
        
        self.device = ('cuda' if torch.cuda.is_available() else 'gpu') if device is None else device
        self.net = net().to(self.device)

        if torch.cuda.device_count() > 1 and parallel:
            print('Using', torch.cuda.device_count(), 'GPUs.')
            self.net = nn.DataParallel(self.net)

        if lr is not None: self.net.lr = lr
        self.net.device = self.device

        if self.net.ID == "Conv_Net":
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, epochs=2, show_load=True, plot_loss=False):
        'Train the model for a number of epochs and return the loss list if plot_loss=True. Returns the accuracy of last Epoch.'

        if plot_loss: loss_list = []
        
        for epoch in range(epochs):
            score = np.zeros(len(self.X_train))

            print(f'\nEpoch: {epoch+1}/{epochs},', f'Model: {self.net.ID}')
            if show_load:
                iterator = tqdm(enumerate(zip(self.X_train, self.y_train)), ascii=True, total=len(self.X_train))
            else:
                iterator = enumerate(zip(self.X_train, self.y_train))

            for i, xy in iterator:
                x, y = xy
                pred, loss, conf = self.net.train_step(x, y)
                if pred == y:
                    score[i] = 1

                if plot_loss: loss_list.append(loss)

                
            print(f'Epoch {epoch+1} Accuracy: {np.mean(score)}')
        
        if plot_loss: return loss_list, np.mean(score)
        return np.mean(score)

    def test(self, X = None, Y = None, show_load=False):
        '''Test the model on the test data and return the accuracy.\n
        If X and Y are None, test the model on all provided data.
        If X and Y are not None, test the model on the provided data.
        If only X is provided, function returns predictions.'''

        try:
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        

            elif len(X.shape) == 2:
                X = np.array([X])
        except AttributeError:
            pass

        if isinstance(Y, np.int32):
            Y = np.array([Y])

        give_preds = False
        if X is None:
            X = self.X_test
            if Y is None:
                Y = self.y_test
            else: 
                Y = np.zeros(len(self.X_test))
                give_preds = True
        else:
            if Y is None:
                Y = np.zeros(len(X))
                give_preds = True
            X = np.array(X)
            if self.net.ID == "Conv_Net":
                X = np.expand_dims(X, axis=1)
            
        score = np.zeros(len(X))
        if show_load:
            iterator = tqdm(enumerate(zip(X, Y)), ascii=True, total=len(X))
        else:
            iterator = enumerate(zip(X, Y))
        for i, xy in iterator:
            x, y = xy
            if self.net.ID == "Conv_Net":
                x = torch.tensor(x[0], dtype=torch.float).unsqueeze(0)
                x = x.view(1, 1, 8, 8).to(self.net.device)
            elif self.net.ID == "Linear_Net":
                x = x.flatten()
                x = torch.tensor(x, dtype=torch.float).to(self.net.device)
            pred = self.net(x)
            if self.net.ID == 'Linear_Net':
                pred = pred.unsqueeze(0)

            
            prob = F.softmax(pred, dim=1)
            conf = prob[0][torch.argmax(pred)].item()

            pred = torch.argmax(pred).item()
            
            if not give_preds:
                if pred == y:
                    score[i] = 1
            else:
                Y[i] = pred

        if give_preds:
            return Y, conf
        return np.mean(score)
    
def dataprep(network, data=None):
    'Prepare the data for training and testing the model. If data is None, load the digits dataset and split it.'
    if data is None:
        digits = datasets.load_digits()
        ims, labs = digits.images, digits.target
        X_train, X_test, y_train, y_test = train_test_split(ims, labs, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = data
    return X_train, X_test, y_train, y_test




if __name__ == '__main__':
    X_train, X_test, y_train, y_test = dataprep(Conv_Net())

    rtt = 8

    trainer = Trainer(Conv_Net, X_train, y_train, X_test, y_test)
    trainer.train(epochs=rtt)
    acc = trainer.test()
    print(acc)