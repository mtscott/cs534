import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import sklearn.linear_model as skl
from sklearn import preprocessing
from sklearn import FunctionTransformer
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def do_nothing(train, test):
    #your code here
    ptrain = train
    ptest = test
    return ptrain, ptest

def do_std(train, test):
    # your code here
    # Normalize the Data to Gaussian using scikit learn
    scalar = preprocessing.StandardScaler()
    scalar.fit(train)
    trainx = scalar.transform(train)

    testx = scalar.transform(test)
    return trainx, testx


def do_log(train, test):
    # your code here
    transformer = FunctionTransformer(np.log + 0.1, validate=True)
    trainx = transformer.transform(train)
    testx = transformer.transform(train)
    return trainx, testx


def do_bin(train, test):
    # your code here
    binarizer = preprocessing.Binarizer().fit(train)  # fit does nothing
    trainx = binarizer.transform(train)
    testx = binarizer.transform(test)
    return trainx, testx


def eval_nb(trainx, trainy, testx, testy):
    test_prob = np.zeros(testx.shape[0])
    # your code here
    gnb = GaussianNB()

    predy = gnb.fit(trainx, trainy).predict(testx)

    test-acc = ((testy == predy).sum())/ testx.shape[0]
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}


def eval_lr(trainx, trainy, testx, testy):
    test_prob = np.zeros(textx.shape[0])
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}

