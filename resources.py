from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def k_fold_cross_validation(X, y, k:int):
    'Yield k (training, testing) folds from the input dataset X, y. Iterator'
    X, y = shuffle(X, y, random_state=42)
    fold_size = len(X) // k

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i + 1 != k else len(X)
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        yield X_train, X_test, y_train, y_test

def plot_loss(loss:list, network:str) -> None:
    plotloss = []
    curr_loss = []
    for i, val in enumerate(loss):
        if i % round((len(loss)/100)) == 1 or i == len(loss)-1:
            plotloss.append(np.mean(curr_loss))
            curr_loss = []
        else:
            curr_loss.append(val)

    plt.plot(plotloss, label=f'{network}')
    plt.xlabel(f'$\\approx x \cdot {round((len(loss)/100))}$ Training steps')
    plt.ylabel('Loss')
    plt.title(f'Loss over training steps, {network} Training')
    plt.show()