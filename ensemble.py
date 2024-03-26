from Conv_Model import Conv_Net, Trainer, dataprep
from Flat_Model import Linear_Net
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


def prep_ims(ims, labs):
    if len(ims.shape) == 3:
        ims = ims.reshape(ims.shape[0], ims.shape[1], ims.shape[2], 1)

    elif len(ims.shape) == 2:
        ims = np.array([ims])

    if isinstance(labs, np.int32):
        labs = np.array([labs])

    return ims, labs

def train_models():
    digits = datasets.load_digits()
    ims = digits.images
    labs = digits.target
    data = train_test_split(ims, labs, test_size=0.2, random_state=42)
    X_trainL, X_testL, y_trainL, y_testL = dataprep(Linear_Net(), data=data) 
    X_trainC, X_testC, y_trainC, y_testC = dataprep(Conv_Net(), data=data)
    ## TODO: Fix the shape of the data, dataprep and prep ims should do the same.
    ## TODO: Eliminate the need for prep_ims/dataprep by making data handling universal
    X_test, y_test = prep_ims(data[1], data[3])
    trainerL = Trainer(Linear_Net, X_trainL, y_trainL, X_testL, y_testL)
    trainerC = Trainer(Conv_Net, X_trainC, y_trainC, X_testC, y_testC)

    ## Train models and get training loss
    Lloss = trainerL.train(epochs=1)
    Closs = trainerC.train(epochs=1)
    return trainerL, trainerC, X_test, y_test


def ensemble(X,y):
    TL, TC, X_test, y_test = train_models()

    for i, xy in enumerate(zip(X_test,y_test)):
        x, y = xy
        print(TL.test(X=[x]), y)


if __name__ == '__main__':
    ensemble(None, None)
    ## TODO DATA FUNGERER SLET IKKE. FIX DET. HVAD MED CONF??