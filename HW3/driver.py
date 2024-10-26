import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
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

ranks_lasso = feature.Lasso(trainx, trainy)
print('Features from rank_lasso:\t', ranks_lasso[:10])

ranks_stepwise = feature.stepwise(trainx, trainy)
print('Features from rank_stepwise:\t', ranks_stepwise[:10])

# Question 1f
# -----------
reg = Regression()
print('Question 1f')
print('-------------------------\n')
full_ans = reg.Ridge(trainx,trainy,testx,testy)
print(f'Full Features:\n{full_ans}\n')

corr_ans = reg.Ridge(trainx[ranks_corr[:10]],trainy[ranks_corr[:10]],testx[ranks_corr[:10]],testy[ranks_corr[:10]])
print(f'Corr Features:\n{corr_ans}\n')

lasso_ans = reg.Ridge(trainx[ranks_lasso[:10]],trainy[ranks_lasso[:10]],testx[ranks_lasso[:10]],testy[ranks_lasso[:10]])
print(f'Lasso Features:\n{lasso_ans}\n')

step_ans = reg.Ridge(trainx[ranks_stepwise[:10]],trainy[ranks_stepwise[:10]],testx[ranks_stepwise[:10]],testy[ranks_stepwise[:10]])
print(f'Stepwise Features:\n{step_ans}\n')

print('Question 1h')
print('-------------------------\n')
for i in [2,5,8]:
    for j in [1,3,5]:
        reg_ans = reg.DecisionTreeRegressor(trainx,trainy, testx, testy, max_depth=i, min_items=j)
        print(f'Tree Regressor: (depth ={i},items ={j}):\n{reg_ans}\n')

print('Question 1i')
print('-------------------------\n')
allx = np.vstack((trainx,testx))
ally = np.concatenate((trainy,testy))

reg_ans = reg.DecisionTreeRegressor(allx,ally, testx, testy, max_depth=2, min_items=5)
print(f'Tree Regressor: (depth ={2},items ={5}):\n{reg_ans}\n')

print('Question 1j')
print('-------------------------\n')
regr_h = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
regr_h.fit(trainx, trainy)


fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,8), dpi = 300)
tree.plot_tree(regr_h,
               filled = True)
plt.show()

regr_i = tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
regr_i.fit(allx, ally)

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,8), dpi = 300)
tree.plot_tree(regr_i,
               filled = True)
plt.show()



"""
Question # 2: Spam Classification using Naive Bayes, Random Forests, and GBDT
"""
# Question 2a
# -----------

# Import Spam Data Set
spam_train = pd.read_csv("spam/spam.train.dat", header=None, sep = '\s',engine='python').to_numpy()
spam_test = pd.read_csv("spam/spam.test.dat", header=None, sep = '\s', engine='python').to_numpy()

spam_trainx, spam_trainy = spam_train[:,:-1], spam_train[:,-1]
spam_testx, spam_testy = spam_test[:,:-1], spam_test[:,-1]


print('Question 2b')
print('-------------------------\n')

def generate_kfold(x, y, k):
    # fold_assignments=np.zeros(x.shape[0])
    # your code here
    fold_assignments = np.random.randint(k, size=x.shape[0])
    return fold_assignments

def eval_kfold(x, y, k, logistic):
    # generate the k-folds
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    #your code here
    # Initialize results
    train_acc_cum = np.array([])
    train_auc_cum = np.array([])
    val_acc_cum = np.array([])
    val_auc_cum = np.array([])

    # Evoke the k - Folds
    folds = generate_kfold(x, y, k)
    #Iterate through the K's
    for i in range(k):

        # Split train and test data based on K-fold assignment
        train_idx = [folds[j] !=i for j in range(len(folds))]
        test_idx = [folds[j] ==i for j in range(len(folds))]
        trainx = x[train_idx,:]
        trainy = y[train_idx]
        testx = x[test_idx,:]
        testy = y[test_idx]

        logistic.fit(trainx, trainy)

        y_train_pred = logistic.predict(trainx)
        y_test_pred = logistic.predict(testx)

        # Calculating Accuracy
        train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
        val_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

        # Calculating AUC
        fpr_train, tpr_train, thresholds = skm.roc_curve(trainy, y_train_pred)
        train_auc = skm.auc(fpr_train, tpr_train)
        fpr_test, tpr_test, thresholds = skm.roc_curve(testy, y_test_pred)
        val_auc = skm.auc(fpr_test, tpr_test)

        train_acc_cum = np.append(train_acc_cum, train_acc)
        train_auc_cum = np.append(train_auc_cum, train_auc)
        val_acc_cum = np.append(val_acc_cum, val_acc)
        val_auc_cum = np.append(val_auc_cum, val_auc)

    results = {"train-acc": np.mean(train_acc_cum),
               "train-auc": np.mean(train_auc_cum),
               "val-acc": np.mean(val_acc_cum),
               "val-auc": np.mean(val_auc_cum)}
        
    return results

bestacc = 0
bestauc = 0
for i in [8,12,16]:
    for j in [5,10,15]:
        for k in [9,12,15]:
            rfc = RandomForestClassifier(n_estimators=i,max_depth=j,min_samples_leaf=k)
            rfc_dict = eval_kfold(spam_trainx, spam_trainy, 10, rfc)
            print(f'RFC: (num={i},depth ={j}, items = {k}):\n{rfc_dict}\n')
            if rfc_dict['val-acc'] >= bestacc:
                print(f'num={i},\tdepth ={j},\t items = {k}')
                bestacc = rfc_dict['val-acc']


print('Question 2c')
print('-------------------------\n')
rfc_ans = eval_randomforest(spam_trainx, spam_trainy, spam_testx, spam_testy, 16,15,9)
print(f'Best Params:\n{rfc_ans}\n')

print('Question 2d')
print('-------------------------\n')

# Access feature importances
rfc = RandomForestClassifier(n_estimators=16,max_depth=15,min_samples_leaf=9)
rfc.fit(spam_trainx,spam_trainy)
importances = rfc.feature_importances_

# Create a dataframe for better visualization

forest_importances = pd.Series(importances)
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


print('Question 2f')
print('-------------------------\n')
bestacc = 0
bestauc = 0
for i in [10,11,12,13,14,15,16]:
    for j in [0.1,0.01,0.001, 0.0001,0.00001]:
        gbdt = GradientBoostingClassifier(n_estimators=i, learning_rate=j)
        gbdt_dict = eval_kfold(spam_trainx, spam_trainy, 10, rfc)
        print(f'GBDT: (num={i},learn rate ={j}):\n{gbdt_dict}\n')
        if gbdt_dict['val-acc'] >= bestacc:
            print(f'num={i}\t,learning rate ={j}')
            besti = i
            bestj = j
            bestacc = gbdt_dict['val-acc']

print('Question 2g')
print('-------------------------\n')
gbdt_ans = eval_gbdt(spam_trainx, spam_trainy, spam_testx, spam_testy, 12,0.00001)
print(f'Best Params:\n{gbdt_ans}\n')