from argparse import ArgumentParser
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

def train_test_prep():
    '''
    This function will load the MNIST data, scale it to a 0 to 1 range, and split it into test/train sets. 
    '''
    tData = np.load('./adultTrain.npy') 
    dataDimension = tData.shape[1]
    X_train = tData[:,range(0, dataDimension-1)]
    y_train = np.ravel(tData[:,[dataDimension-1]])

    testData = np.load('./adultTest.npy') 
    X_test = testData[:,range(0, dataDimension-1)]
    y_test = np.ravel(testData[:,[dataDimension-1]])

    return X_train, X_test, y_train, y_test

def create_stratified_splits(x, y, num=5):
    datasets = []
    from sklearn.cross_validation import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(y, num, test_size=0.1, random_state=0)
    for train_index, test_index in sss:
        datasets.append((x[train_index], y[train_index]))
    return datasets

def train_dbn_dataset(dataset, x_test, y_test, alpha, nhidden, epochs, batch_size, noises=[]):
    from nolearn.dbn import DBN
    num_classes = len(set(y_test))
    print "Number of classes", num_classes
    x_train, y_train = dataset
    dbn_model = DBN([x_train.shape[1], nhidden, num_classes],
                    learn_rates = alpha,
                    learn_rate_decays = 0.9,
                    epochs = epochs,
                    verbose = 1,
                    nesterov=False,
                    minibatch_size=batch_size,
                    noises = noises)

    dbn_model.fit(x_train, y_train)
    from sklearn.metrics import classification_report, accuracy_score
    y_true, y_pred = y_test, dbn_model.predict(x_test) # Get our predictions
    print(classification_report(y_true, y_pred)) # Classification on each digit
    print(roc_auc_score(y_true, y_pred)) # Classification on each digit
    return y_pred, roc_auc_score(y_true, y_pred)

def main(args):
    alpha = args.alpha
    nhidden = args.hidden
    epochs = args.epochs
    noises = eval(args.noises)
    batch_size = args.batch_size
    print "Using configuration"
    print "learning rate", alpha
    print "Number of Hidden units", nhidden
    print "Epochs", epochs
    print "batch size", batch_size
    print "Noise std-devs", noises
    x_train, x_test, y_train, y_test = train_test_prep()
    datasets = create_stratified_splits(x_train, y_train)
    preds = []
    for dataset in datasets:
        y_pred, score = train_dbn_dataset(dataset, x_test, y_test, alpha, nhidden, epochs, batch_size=batch_size, noises=noises)
        preds.append(y_pred)
    preds_arr = np.vstack(preds)
    preds_std = np.std(preds_arr, axis=0)
    print "Mean prediction variability is",np.mean(preds_std)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate",  dest="alpha", default=0.3, type=float, help="Learning rate (default:0.3)")
    parser.add_argument("--num_hidden_units", dest="hidden", default=300, type=int, help="Number of hidden units (default:300)")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="Number of epochs (default:10)")
    parser.add_argument("--batch_size", dest="batch_size", default=64, help="Batch size for SGD (default:64)")
    parser.add_argument("--noises", dest="noises", default='[]', help="Noises as list (default: [])")
    args = parser.parse_args()
    main(args)
