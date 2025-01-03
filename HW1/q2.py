import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(trainx, valx, testx):
    # Normalize the Data to Gaussian using scikit learn
    scalar = preprocessing.StandardScaler()
    scalar.fit(trainx)
    trainx = scalar.transform(trainx)

    valx = scalar.transform(valx)

    testx = scalar.transform(testx)
    return trainx, valx, testx

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
    trnvalx = np.concatenate((trainx,valx), axis=0)
    trnvaly = np.concatenate((trainy, valy), axis=0)
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

def eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha):
    # Create ridge regression object
    ridg = Ridge(alpha = alpha)

    # Train the model using the training sets
    ridg.fit(trainx, trainy)
    print(ridg.coef_)

    # Make predictions using the testing set
    train_pred = ridg.predict(trainx)
    val_pred = ridg.predict(valx)
    test_pred = ridg.predict(testx)

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
    

def eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha):
    # Create lasso regression object
    lass = Lasso(alpha = alpha)

    # Train the model using the training sets
    lass.fit(trainx, trainy)
    print(lass.coef_)

    # Make predictions using the testing set
    train_pred = lass.predict(trainx)
    val_pred = lass.predict(valx)
    test_pred = lass.predict(testx)

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