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
    trainerC.train(epochs=8)
    trainerL.train(epochs=3)
    return trainerL, trainerC, X_test, y_test


def ensemble(Lin=None, Conv=None, X_test=None,y_test=None, show_stats=False, 
             weighted = False):
    if Lin is None or Conv is None:
        TL, TC, X_test, y_test = train_models()
    
    else:
        TL, TC = Lin, Conv

    LinACC = TL.test(X=X_test, Y=y_test, show_load=False)
    ConACC = TC.test(X=X_test, Y=y_test, show_load=False)

    preds = []
    score = np.zeros(len(X_test))
    for i, xy in enumerate(zip(X_test,y_test)):
        x, y = xy
        predL, confL = TL.test(X=x, show_load=False)
        predC, confC = TC.test(X=x, show_load=False)

        predL, predC = predL[0], predC[0]

        ## TODO Perhaps look at **TRAINING** accuracy to determine if one model should weigh more
        ## Important that we do not use test accuracy (LinACC and ConAcc), since we do not know that yet
        if confL < confC: 
            curr_pred = predC
            preds.append(predC)
        else:
            curr_pred = predL
            preds.append(predL)
        
        if curr_pred == y:
            score[i] = 1
    if show_stats:
        print('LINEAR ACCURACY:', LinACC)
        print('CONV ACCURACY:', ConACC)
        print('ENSEMBLE ACCURACY:', np.mean(score))
        print('Ensemble improved accuracy by:', (np.mean(score) - max(LinACC, ConACC))*100, '%')
    return np.mean(score)
        


if __name__ == '__main__':
    ensemble(show_stats=True)