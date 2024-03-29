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
    CLast_Epoch_Acc = trainerC.train(epochs=8)
    Llast_Epoch_Acc = trainerL.train(epochs=3)
    return trainerL, trainerC, X_test, y_test, CLast_Epoch_Acc, Llast_Epoch_Acc


def ensemble(Lin=None, Conv=None, X_test=None,y_test=None, LinTrainAcc=None, ConTrainAcc=None, 
             show_stats=False, weighted=False):
    if Lin is None or Conv is None:
        TL, TC, X_test, y_test, ConTrainAcc, LinTrainAcc = train_models()
    
    else:
        TL, TC = Lin, Conv

    LinACC = TL.test(X=X_test, Y=y_test, show_load=False)
    ConACC = TC.test(X=X_test, Y=y_test, show_load=False)

    preds = []
    score = np.zeros(len(X_test))
    decision_model = np.zeros(len(X_test)) ## 1 for Conv, 0 for Linear
    for i, xy in enumerate(zip(X_test,y_test)):
        x, y = xy
        predL, confL = TL.test(X=x, show_load=False)
        predC, confC = TC.test(X=x, show_load=False)

        predL, predC = predL[0], predC[0]

        if confL * (LinTrainAcc**2 if weighted else 1) < confC * (ConTrainAcc**2 if weighted else 1): 
            curr_pred = predC
            preds.append(predC)
            decision_model[i] = 1
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
        print(f'Decision Distribution: {np.mean(decision_model)} Conv, {1-np.mean(decision_model)} Linear')
    return np.mean(score)
        


if __name__ == '__main__':
    ensemble(show_stats=True, weighted=True)