from sklearn import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn import datasets

from resources import k_fold_cross_validation
from Flat_Model import Linear_Net
from Conv_Model import Conv_Net

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

        # Convert your data to PyTorch tensors
        X_train_tensor = torch.from_numpy(self.X_train).float()
        y_train_tensor = torch.from_numpy(self.y_train).long()

        # Create a dataset from your tensors
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        # Create a DataLoader from your dataset
        train_loader = DataLoader(train_dataset, batch_size=32)  # Choose your batch size

        # Modify your training loop to use the DataLoader
        for epoch in range(epochs):
            if show_load:
                print(f'\nEpoch: {epoch+1}/{epochs},', f'Model: {self.net.ID}')
            for i, (x_batch, y_batch) in enumerate(train_loader):
                pred, loss, conf = self.net.train_step(x_batch, y_batch)
                if plot_loss: loss_list.append(loss)

        accuracy = np.mean([((p == y) * 1).item() for p, y in zip(pred, y_batch)])
        if plot_loss: return loss_list, accuracy
        return accuracy

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


if __name__ == '__main__':
    digits = datasets.load_digits()
    ims, labs = digits.images, digits.target
    accs = []
    for X_train, X_test, y_train, y_test in k_fold_cross_validation(ims, labs, 5):
        trainer = Trainer(Conv_Net, X_train, y_train, X_test, y_test)
        trainer.train(epochs=16)
        acc = trainer.test()
        print(acc)
        accs.append(acc)
    print('Average accuracy:', np.mean(accs))