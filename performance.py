from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from Conv_Model import Conv_Net
from Trainer import Trainer
from Flat_Model import Linear_Net
from ensemble import *
from resources import *

def visualize_performance(trainer, X_test, y_test):
    # Get predictions
    y_pred, _ = trainer.test(X_test)


    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))
    return classification_report(y_test, y_pred), y_pred

def running_test(X_train, X_test, y_train, y_test, epochs, model, plot=True):
    trainer = Trainer(model, X_train, y_train, X_test, y_test)
    acc = []
    train_acc = []
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs},', f'Model: {trainer.net.ID}')

        train_acc.append(trainer.train(epochs=1, show_load=False))
        
        acc.append(trainer.test())

    if plot:
        sns.lineplot(acc, label='Test Accuracy')
        sns.lineplot(train_acc, label='Train Accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy over epochs')
        plt.show()
    return acc, train_acc

def find_optimal_epoch(ims, labs, model, max_epochs=8, k=5, plot=True):
    accs = []
    for X_train, X_test, y_train, y_test in k_fold_cross_validation(ims, labs, k=k):
        print(f'\nFinding Optimal Epoch:\nCommencing fold {len(accs)+1}:\n')
        acc, _ = running_test(X_train, X_test, y_train, y_test, epochs=max_epochs, 
                                       model=model, plot=plot)
        accs.append(acc)
        
    opt = [acc.index(max(acc)) for acc in accs]

    ## TODO: Running test returns the trainer, so find_optimal_epoch can call 
    ## itself recursively if last epoch is best seen yet

    return np.median(opt) + 1

def main_performance(ims, labs, model, max_epochs=8, k=5):
    reps = []
    accuracies = []
    for X_train, X_test, y_train, y_test in k_fold_cross_validation(ims, labs, k=k):
        opt = round(find_optimal_epoch(X_train, y_train, model, max_epochs=max_epochs, k=k, plot=False))
        print('Recommended Epochs:', opt)
        acc = 0
        trainer = Trainer(model, X_train, y_train, X_test, y_test)
        trainer.train(epochs=opt)
        report, preds = visualize_performance(trainer, X_test, y_test)
        for y, pred in zip(y_test, preds):
            if y == pred:
                acc += 1
        reps.append(report)
        accuracies.append(acc/len(y_test))
    return reps, accuracies



if __name__ == '__main__':
    digits = datasets.load_digits()
    ims, labs = digits.images, digits.target
    reps, accuracies = main_performance(ims, labs, Conv_Net, max_epochs=8, k=5)
    for i, rep in enumerate(reps):
        print(f'\nFold {i+1}:\n', rep)
        print(f'Accuracy: {accuracies[i]}')
    print(f'\nAverage accuracy: {np.mean(accuracies)}')