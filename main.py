from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

## Import models and functions from other files
from Conv_Model import Conv_Net, Trainer, dataprep
from Flat_Model import Linear_Net
from ensemble import *

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

    data = train_test_split(ims, labs, test_size=0.2, random_state=42)

    ## Boring stuff some dude from YouTube did;

    bx, by = digits.data[:-10], digits.target[:-10]
    bx_train, bx_test, by_train, by_test = train_test_split(bx, by, test_size=0.2, random_state=42)
    svmmodel = SVC(kernel='linear')
    svmmodel.fit(bx_train, by_train)
    print('Support Vector Classifier model accuracy:',SVMacc:=svmmodel.score(bx_test, by_test))

    ## Boring stuff that i'll do:

    gnb = GaussianNB()
    gnb.fit(bx_train, by_train)
    print('Naïve Bayes model accuracy:',GNBacc:=gnb.score(bx_test, by_test))

    ## Not nearly as fun as making your own model

    ## Remove data=data to perform the split here. Then the top 3 lines can be removed.
    ## I have done it this way to train on the same data for each model.
    X_trainL, X_testL, y_trainL, y_testL = dataprep(Linear_Net(), data=data) 
    X_trainC, X_testC, y_trainC, y_testC = dataprep(Conv_Net(), data=data)

    ## Define number of epochs
    rttL, rttC = 3, 8

    ## Instantiate trainers for each model
    trainerL = Trainer(Linear_Net, X_trainL, y_trainL, X_testL, y_testL)
    trainerC = Trainer(Conv_Net, X_trainC, y_trainC, X_testC, y_testC)

    ## Train models and get training loss
    Lloss = trainerL.train(epochs=rttL, plot_loss=True)
    Closs = trainerC.train(epochs=rttC, plot_loss=True)

    ## Test models and get accuracy
    accL = trainerL.test()
    accC = trainerC.test()

    accENS = ensemble(Lin=trainerL, Conv=trainerC, X_test=X_testL, y_test=y_testL)


    ## Print results
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
    plot_loss(Lloss, 'Linear Net')
    plot_loss(Closs, 'Conv Net')
    print('-------------------------------------------------------------')