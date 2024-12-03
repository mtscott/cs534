import numpy as np
# from matplotlib import pyplot
# import matplotlib.pyplot as plt
# import pandas as pd
from sklearn import linear_model
import sklearn.linear_model as skl
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score


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
    def logepsp(x):
        return np.log(0.1 + x)
    
    transformer = FunctionTransformer(logepsp, validate=True)
    trainx = transformer.transform(train)
    testx = transformer.transform(test)
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
    gnb.fit(trainx,trainy)

    # Predicting Data
    y_train_pred = gnb.predict(trainx)
    y_test_pred = gnb.predict(testx)

    # Calculating Accuracy
    train_acc = gnb.score(xt, trainy)
    test_acc = f1_score(y_test_pred,testy)

    # Calculating AUC
    fpr_train, tpr_train, thresholds = roc_curve(trainy, y_train_pred)
    train_auc = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = roc_curve(testy, y_test_pred)
    test_auc = auc(fpr_test, tpr_test)

    # Calculating Probability
    test_prob = gnb.predict_proba(testx)
    test_prob = test_prob[:,1]

    dict = {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}
    return dict


def eval_lr(trainx, trainy, testx, testy):
    test_prob = np.zeros(testx.shape[0])
    # your code here
    lr = LogisticRegression(penalty=None)
    lr.fit(trainx, trainy)

    y_train_pred = lr.predict(trainx)
    y_test_pred = lr.predict(testx)
    # cnf_mat_train = confusion_matrix(trainy, y_train_pred)
    # cnf_mat_test = confusion_matrix(testy, y_test_pred)

    # Calculating Accuracy
    train_acc = accuracy_score(trainy, y_train_pred)
    test_acc = accuracy_score(testy, y_test_pred)

    # Calculating AUC
    fpr_train, tpr_train, thresholds = roc_curve(trainy, y_train_pred)
    train_auc = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = roc_curve(testy, y_test_pred)
    test_auc = auc(fpr_test, tpr_test)

    # Calculating Probability
    test_prob = lr.predict_proba(testx)
    test_prob = test_prob[:,0]

    dict = {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}
    
    return dict

