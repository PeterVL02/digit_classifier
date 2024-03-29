from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

## Import models and functions from other files
from Conv_Model import Conv_Net, Trainer, dataprep
from Flat_Model import Linear_Net
from ensemble import *


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



if __name__ == '__main__':
    digits = datasets.load_digits()
    ims, labs = digits.images, digits.target

    ens_rolling_acc = []
    lin_rolling_acc = []
    conv_rolling_acc = []
    svm_rolling_acc = []
    gnb_rolling_acc = []

    fold = 0
    for X_train, X_test, y_train, y_test in k_fold_cross_validation(ims, labs, k=5):
        fold += 1
        print(f'Commencing fold {fold}:\n')

        ## Reshape data for SciKit models
        n_samples, width, height = X_train.shape
        X_train_reshaped = X_train.reshape((n_samples, width * height))

        n_samples = X_test.shape[0]
        X_test_reshaped = X_test.reshape((n_samples, width * height))

        ## Boring stuff some dude from YouTube did;
        svmmodel = SVC(kernel='linear')
        svmmodel.fit(X_train_reshaped, y_train)
        print('Support Vector Classifier model accuracy:',SVMacc:=svmmodel.score(X_test_reshaped, y_test))

        ## Boring stuff that i'll do:

        gnb = GaussianNB()
        gnb.fit(X_train_reshaped, y_train)
        print('Naïve Bayes model accuracy:',GNBacc:=gnb.score(X_test_reshaped, y_test))

        ## Not nearly as fun as making your own model


        ## Define number of epochs
        rttL, rttC = 3, 8

        ## Instantiate trainers for each model
        trainerL = Trainer(Linear_Net, X_train, y_train, X_test, y_test)
        trainerC = Trainer(Conv_Net, X_train, y_train, X_test, y_test)

        ## Train models and get training loss
        Lloss, LinTrainAcc = trainerL.train(epochs=rttL, plot_loss=True)
        Closs, ConTrainAcc = trainerC.train(epochs=rttC, plot_loss=True)

        ## Test models and get accuracy
        accL = trainerL.test()
        accC = trainerC.test()

        accENS = ensemble(Lin=trainerL, Conv=trainerC, X_test=X_test, y_test=y_test, weighted=True, 
                          LinTrainAcc=LinTrainAcc, ConTrainAcc=ConTrainAcc)


        ## Print results
        print('\n-------------------------------------------------------------')
        print('Linear Net accuracy:',accL)
        print('Conv Net accuracy:',accC)
        print('Ensemble Net accuracy:',accENS)
        print('Support Vector Classifier accuracy:',SVMacc)
        print('Naïve Bayes accuracy:',GNBacc)
        print('-------------------------------------------------------------')
        print('Conv Net improvement:',f'{(accC-SVMacc)*100:.2f}%')
        print('Linear Net improvement:',f'{(accL-SVMacc)*100:.2f}%')
        print('Ensemble Net improvement:',f'{(accENS-SVMacc)*100:.2f}%')

        if accC > max(SVMacc, GNBacc) or accL > max(SVMacc, GNBacc) or accENS > max(SVMacc, GNBacc):
            print('We beat the SciKit models! At least those I have created...')
            

        ## Plot loss
        # plot_loss(Lloss, 'Linear Net')
        # plot_loss(Closs, 'Conv Net')

        print('-------------------------------------------------------------')
        ens_rolling_acc.append(accENS)
        lin_rolling_acc.append(accL)
        conv_rolling_acc.append(accC)
        svm_rolling_acc.append(SVMacc)
        gnb_rolling_acc.append(GNBacc)

    print('Ensemble Rolling Accuracy:',np.mean(ens_rolling_acc))
    print('Linear Rolling Accuracy:',np.mean(lin_rolling_acc))
    print('Conv Rolling Accuracy:',np.mean(conv_rolling_acc))
    print('SVM Rolling Accuracy:',np.mean(svm_rolling_acc))
    print('GNB Rolling Accuracy:',np.mean(gnb_rolling_acc))