from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np


## Import models and functions from other files
from Conv_Model import Conv_Net
from Flat_Model import Linear_Net
from ensemble import *
from resources import *
from Batch_Trainer import Trainer
from performance import main_optimized_train



if __name__ == '__main__':
    digits = datasets.load_digits()
    ims, labs = digits.images, digits.target

    ens_rolling_acc = []
    lin_rolling_acc = []
    conv_rolling_acc = []
    svm_rolling_acc = []
    gnb_rolling_acc = []

    fold = 0

    k = 10

    for X_train, X_test, y_train, y_test in k_fold_cross_validation(ims, labs, k=k):
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
        rttL, rttC = 17, 25

        ## TODO: Find out whether PERFORMANCE.PY TO FIND OPTIMAL EPOCHS AND TRAIN (MAIN_OPTIMIZED_TRAIN) is better
        ## or if we should just train the models with a fixed number of epochs
        ## ANSWER: It seems to work best with a fixed number of epochs, but we can still use the performance.py functions

        # reportL, accL, LinTrainAcc, trainerL = main_optimized_train(Linear_Net, X_train, X_test, y_train, y_test)
        # reportC, accC, ConTrainAcc, trainerC = main_optimized_train(Conv_Net, X_train, X_test, y_train, y_test)

        
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

    print('\nEnsemble Rolling Accuracy:',np.mean(ens_rolling_acc))
    print('Linear Rolling Accuracy:',np.mean(lin_rolling_acc))
    print('Conv Rolling Accuracy:',np.mean(conv_rolling_acc))
    print('SVM Rolling Accuracy:',np.mean(svm_rolling_acc))
    print('GNB Rolling Accuracy:',np.mean(gnb_rolling_acc))
    print('-------------------------------------------------------------')
    print(f'Ensemble Net improvement over SciKit SVM: {(np.mean(ens_rolling_acc)-np.mean(svm_rolling_acc))*100:.2f}%')
    if np.mean(ens_rolling_acc) > np.mean(svm_rolling_acc):
        print(f'We beat the SciKit SVM model in {k} fold cross-validation!')