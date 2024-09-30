import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as skl
import sklearn.metrics as skm

def _eval_perf(model, x, y):
    # your code here
    return acc, auc


def _eval_model(model, trainx, trainy, valx, valy):
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "val-acc": val_acc, "val-auc": val_auc}


def generate_train_val(x, y, valsize):
    # your code here
    return {"train-x": train_x, "train-y": train_y,
            "val-x": test_x, "val-y": test_y}


def generate_kfold(x, y, k):
    fold_assignments=np.zeros(x.shape[0])
    # your code here
    return fold_assignments


def eval_holdout(x, y, valsize, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    # your code here
    return results


def eval_kfold(x, y, k, logistic):
    # generate the k-folds
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    #your code here
    return results


def eval_mccv(x, y, valsize, s, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    # your code here
    return results
