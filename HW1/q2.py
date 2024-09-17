import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.linearmodel import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(trainx, valx, testx):

    return 1

def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(trainx, trainy)

    # Make predictions using the testing set
    train_pred = regr.predict(trainx)
    val_pred = regr.predict(valx)
    test_pred = regr.predict(testx)

    tr_rmse = np.sqrt(mean_squared_error(trainy,train_pred))
    tr_r2 = r2_score(trainy,train_pred)
    va_rmse = np.sqrt(mean_squared_error(valy,val_pred))
    va_r2 = r2_score(valy,val_pred)
    te_rmse = np.sqrt(mean_squared_error(testy,test_pred))
    te_r2 = r2_score(testy,test_pred)

    ansdict = {
         'train-rmse':  tr_rmse,
         'train-r2':    tr_r2,
         'val-rmse':    va_rmse,
         'val-r2':      va_r2,
         'test-rmse':   te_rmse,
         'test-r2':     te_r2

    }

    return ansdict

def eval_linear2(trainx, trainy, valx, valy, testx, testy):
    trnvalx = np.append(trainx,valx)
    trnvaly = np.append(trainy, valy)
    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(trnvalx, trnvaly)

    # Make predictions using the testing set
    train_pred = regr.predict(trainx)
    val_pred = regr.predict(valx)
    test_pred = regr.predict(testx)

    tr_rmse = np.sqrt(mean_squared_error(trainy,train_pred))
    tr_r2 = r2_score(trainy,train_pred)
    va_rmse = np.sqrt(mean_squared_error(valy,val_pred))
    va_r2 = r2_score(valy,val_pred)
    te_rmse = np.sqrt(mean_squared_error(testy,test_pred))
    te_r2 = r2_score(testy,test_pred)

    ansdict = {
         'train-rmse':  tr_rmse,
         'train-r2':    tr_r2,
         'val-rmse':    va_rmse,
         'val-r2':      va_r2,
         'test-rmse':   te_rmse,
         'test-r2':     te_r2

    }
    
    return ansdict

def eval_ridge(trainx, trainy, valx, valy, testx,
testy, alpha):
    
    return 0

def eval_lasso(trainx, trainy, valx, valy, testx,
testy, alpha):
    
    return 0