import numpy as np
import pandas as pd
import datetime
from scipy import stats
from scipy.stats import pearsonr
from q1 import *
from q2 import *

def preprocess_data(trainx, valx, testx):
    # Normalize the Data to Gaussian using scikit learn
    scalar = preprocessing.StandardScaler()
    scalar.fit(trainx)
    trainx = scalar.transform(trainx)

    valx = scalar.transform(valx)

    testx = scalar.transform(testx)
    return trainx, valx, testx

# Import Energy Data Set
energy_train = pd.read_csv("energydata/energy_train.csv")
energy_val = pd.read_csv("energydata/energy_val.csv")
energy_test = pd.read_csv("energydata/energy_test.csv")

# Preprocess 
energy_train['date'] = pd.to_datetime(energy_train['date'])
energy_val['date'] = pd.to_datetime(energy_val['date'])
energy_test['date'] = pd.to_datetime(energy_test['date'])

energy_train['date'] = pd.to_numeric(energy_train['date'])//6e10
energy_val['date'] = pd.to_numeric(energy_val['date'])//6e10
energy_test['date'] = pd.to_numeric(energy_test['date'])//6e10

print(energy_train.head())
energy_train.info()
print(energy_train['date'])


pre_energy_train, pre_energy_val, pre_energy_test = preprocess_data(energy_train, energy_val, energy_test)

trainx = np.concatenate((pre_energy_train[:,2:],pre_energy_train[:,:0]), axis = 1)
trainy = pre_energy_train[:,1]
print(f"shape of train x:{trainx.shape}")
print(f"shape of train y:{trainy.shape}")

valx = np.concatenate((pre_energy_val[:,2:],pre_energy_val[:,:0]), axis = 1)
valy = pre_energy_val[:,1]

testx = np.concatenate((pre_energy_test[:,2:],pre_energy_test[:,:0]), axis = 1)
testy = pre_energy_test[:,1]

"""
Question # 1: Energy Appliance Regression
"""
# Question 1d
# -----------

print(f"data type train x: {trainx.dtype}")
print(f"data type train y: {trainy.dtype}")
print(f"dim train x: {trainx.ndim}")
print(f"dim train y: {trainy.ndim}")
feature = FeatureSelection()
ranks_corr = feature.rank_correlation(trainx, trainy)
print('Features from rank_correlation:\t', ranks_corr[:10])

ranks_lasso = feature.lasso(trainx, trainy)
print('Features from rank_lasso:\t', ranks_lasso[:10])

ranks_stepwise = feature.stepwise(trainx, trainy)
print('Features from rank_stepwise:\t', ranks_stepwise[:10])

# Question 1f
# -----------
reg = Regression()
print('Question 1f')
print('-------------------------\n')
full_ans = reg.ridge_lr(trainx,trainy,testx,testy)
print(f'Full Features:\n{full_ans}')

corr_ans = reg.ridge_lr(trainx[ranks_corr[:10]],trainy[ranks_corr[:10]],testx[ranks_corr[:10]],testy[ranks_corr[:10]])
print(f'Corr Features:\n{corr_ans}')

lasso_ans = reg.ridge_lr(trainx[ranks_lasso[:10]],trainy[ranks_lasso[:10]],testx[ranks_lasso[:10]],testy[ranks_lasso[:10]])
print(f'Lasso Features:\n{lasso_ans}')

step_ans = reg.ridge_lr(trainx[ranks_stepwise[:10]],trainy[ranks_stepwise[:10]],testx[ranks_stepwise[:10]],testy[ranks_stepwise[:10]])
print(f'Corr Features:\n{step_ans}')

print('Question 1h')
print('-------------------------\n')
reg_ans = reg.tree_regression(trainx,trainy, testx, testy)
print(f'Tree Regressor:\n{reg_ans}')


"""
Question # 2: Spam Classification using Naive Bayes, Random Forests, and GBDT
"""
# Question 2a
# -----------

# Import Spam Data Set
spam_train = pd.read_csv("spam/spam.train.dat", sep = '\s',engine='python').to_numpy()
spam_test = pd.read_csv("spam/spam.test.dat", sep = '\s', engine='python').to_numpy()

spam_trainx, spam_trainy = spam_train[:,:-1], spam_train[:,-1]
spam_testx, spam_testy = spam_test[:,:-1], spam_test[:,-1]