import numpy as np
# from matplotlib import pyplot
# import matplotlib.pyplot as plt
# import pandas as pd
from sklearn import linear_model
import sklearn.linear_model as skl
from sklearn import preprocessing
import sklearn.preprocessing as skp
from sklearn.preprocessing import FunctionTransformer
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sklearn.metrics as skm
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import r_regression, SequentialFeatureSelector
from scipy import stats
from scipy.stats import pearsonr


class FeatureSelection:  

    def rank_correlation(self, x, y):
       ranks = np.zeros(y.shape[0])
       #your code here
       ranks = r_regression(x, y)
       ranks = np.argsort(np.abs(ranks))
       return ranks

    def lasso(self, x, y):
       #ranks = np.zeros(y.shape[0])
       # your code here
       las = skl.Lasso(alpha = 0.01)
       las.fit(x, y)
       ranks = np.argsort(np.abs(las.coef_))
       return ranks

    def stepwise(self, x, y):
        # ranks = np.zeros(y.shape[0])
        # your code here
        linreg = skl.LinearRegression()
        sfs = SequentialFeatureSelector(linreg)
        sfs.fit(x,y)
        ranks = np.argsort(sfs.get_support())
        return ranks

class Regression:
    def ridge_lr(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        # Create ridge regression object
        ridg = skl.Ridge()

        # Train the model using the training sets
        ridg.fit(train_x, train_y)
        #print(ridg.coef_)

        # Make predictions using the testing set
        train_pred = ridg.predict(train_x)
        test_pred = ridg.predict(test_x)

        tr_rmse = np.sqrt(root_mean_squared_error(train_y,train_pred))
        tr_r2 = r2_score(train_y,train_pred)
        te_rmse = np.sqrt(root_mean_squared_error(test_y,test_pred))
        te_r2 = r2_score(test_y,test_pred)

        test_prob = {
            'train-rmse':  tr_rmse,
            'train-r2':    tr_r2,
            'test-rmse':   te_rmse,
            'test-r2':     te_r2

        }
        return test_prob

    def tree_regression(self, train_x, train_y, test_x, test_y):
        test_prob = np.zeros(test_x.shape[0])
        regr_1 = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
        regr_1.fit(train_x, train_y)

        train_pred = regr_1.predict(train_x)
        test_pred = regr_1.predict(test_x)

        tr_rmse = np.sqrt(root_mean_squared_error(train_y,train_pred))
        tr_r2 = r2_score(train_y,train_pred)
        te_rmse = np.sqrt(root_mean_squared_error(test_y,test_pred))
        te_r2 = r2_score(test_y,test_pred)

        test_prob = {
            'train-rmse':  tr_rmse,
            'train-r2':    tr_r2,
            'test-rmse':   te_rmse,
            'test-r2':     te_r2

        }

        return test_prob

