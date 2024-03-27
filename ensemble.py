from Conv_Model import Conv_Net, Trainer, dataprep
from Flat_Model import Linear_Net
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

def train_models():
    digits = datasets.load_digits()
    ims = digits.images
    labs = digits.target

    X_train, X_test, y_train, y_test = train_test_split(ims, labs, test_size=0.2, random_state=42)

    trainerL = Trainer(Linear_Net, X_train, y_train, X_test, y_test)
    trainerC = Trainer(Conv_Net, X_train, y_train, X_test, y_test)

    ## Train models and get training loss
    trainerC.train(epochs=1)
    trainerL.train(epochs=1)
    return trainerL, trainerC, X_test, y_test


def ensemble(X,y):
    TL, TC, X_test, y_test = train_models()

    for i, xy in enumerate(zip(X_test,y_test)):
        x, y = xy
        print(TL.test(X=x), y)
        print(TC.test(X=x), y)


if __name__ == '__main__':
    ensemble(None, None)
    ## TODO DATA FUNGERER SLET IKKE. FIX DET. HVAD MED CONF??