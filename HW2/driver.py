import pandas as pd
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from q2 import *
from q3 import *
from scipy import stats

# Read Data In
spam_train = pd.read_csv("spam/spam.train.dat", sep = '\s',engine='python').to_numpy()
spam_test = pd.read_csv("spam/spam.test.dat", sep = '\s', engine='python').to_numpy()

# -------------------------------------
# ------------ Question 2 -------------
# -------------------------------------

# Separate X and y so as to not alter the classificiation
trainx, trainy = spam_train[:,:-1], spam_train[:,-1]
testx, testy = spam_test[:,:-1], spam_test[:,-1]


# Preprocess Data 4 Different Ways
dn_trainx, dn_testx = do_nothing(trainx,testx)
st_trainx, st_testx = do_std(trainx,testx)
lo_trainx, lo_testx = do_log(trainx,testx)
bi_trainx, bi_testx = do_bin(trainx,testx)


# Apply Naive Bayes model to each Preprocessing Scheme
fpr_dn_n, tpr_dn_n, dn_dict_n = eval_nb(dn_trainx, trainy, dn_testx, testy)
fpr_st_n, tpr_st_n,st_dict_n = eval_nb(st_trainx, trainy, st_testx, testy)
fpr_lo_n, tpr_lo_n,lo_dict_n = eval_nb(lo_trainx, trainy, lo_testx, testy)
fpr_bi_n, tpr_bi_n,bi_dict_n = eval_nb(bi_trainx, trainy, bi_testx, testy)

print('Starting 2(c)\n')
print('----------------------------------\n')
print('Do Nothing\n', dn_dict_n, '\n')
print('Do Standardization\n', st_dict_n, '\n')
print('Do Log\n',lo_dict_n, '\n')
print('Do Binary\n', bi_dict_n, '\n')

# Apply Logistic Regression model to each Preprocessing Scheme
fpr_dn_l, tpr_dn_l,dn_dict_l = eval_lr(dn_trainx, trainy, dn_testx, testy)
fpr_st_l, tpr_st_l,st_dict_l = eval_lr(st_trainx, trainy, st_testx, testy)
fpr_lo_l, tpr_lo_l,lo_dict_l = eval_lr(lo_trainx, trainy, lo_testx, testy)
fpr_bi_l, tpr_bi_l,bi_dict_l = eval_lr(bi_trainx, trainy, bi_testx, testy)

print('Starting 2(e)\n')
print('----------------------------------\n')
print('Do Nothing\n', dn_dict_l, '\n')
print('Do Standardization\n', st_dict_l, '\n')
print('Do Log\n',lo_dict_l, '\n')
print('Do Binary\n', bi_dict_l, '\n')

# Generating the ROC curves All NB, All LR, then Best of Each
plt.plot(fpr_dn_n,tpr_dn_n,label="Nothing")
plt.plot(fpr_st_n,tpr_st_n,label="Standardization")
plt.plot(fpr_lo_n,tpr_lo_n,label="Logarithmic")
plt.plot(fpr_bi_n,tpr_bi_n,label="Binarize")
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparing ROC for Naive Bayes')
plt.show()

plt.plot(fpr_dn_l,tpr_dn_l,label="Nothing")
plt.plot(fpr_st_l,tpr_st_l,label="Standardization")
plt.plot(fpr_lo_l,tpr_lo_l,label="Logarithmic")
plt.plot(fpr_bi_l,tpr_bi_l,label="Binarize")
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparing ROC for Logistic Regression')
plt.show()

plt.plot(fpr_lo_n,tpr_lo_n,label="NB")
plt.plot(fpr_lo_l,tpr_lo_l,label="LR")
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparing ROC for Naive Bayes and Logistics Regression')
plt.show()

# -------------------------------------
# ------------ Question 3 -------------
# -------------------------------------

# We use lo_trainx, lo_testx to preprocess our data.
trainx, trainy = spam_train[:,:-1], spam_train[:,-1]
testx, testy = spam_test[:,:-1], spam_test[:,-1]
trainx, testx = do_log(trainx,testx)

x = np.concatenate((trainx, testx))
y = np.append(trainy, testy)

max1 = 0
max2 = 0
max3 = 0
max4 = 0

for c in np.arange(0.1,1,0.1):
    ridge_lr = LogisticRegression(max_iter=5000, C = 1./c, penalty='l2')
    lasso_lr = LogisticRegression(max_iter=5000, solver='liblinear', C = 1./c, penalty='l1')

    results_1r = eval_kfold(x, y, 5, ridge_lr)
    if results_1r['val-acc'] >= max1:
        abest1 = c
        max1 = results_1r['val-acc']
    results_2r = eval_kfold(x, y, 10, ridge_lr)
    if results_2r['val-acc'] >= max2:
        abest2 = c
        max2 = results_2r['val-acc']

    results_1l = eval_kfold(x, y, 5, lasso_lr)
    if results_1r['val-acc'] >= max3:
        abest3 = c
        max3 = results_1l['val-acc']
    results_2l = eval_kfold(x, y, 10, lasso_lr)
    if results_2l['val-acc'] >= max4:
        abest4 = c
        max4 = results_2l['val-acc']

    

print('Starting 3(g)\n')
print('----------------------------------\n')
print('Ridge Results, k=5\n', results_1r, '\n')
print('Ridge Results, k=10\n', results_2r, '\n')
print('Lasso Results, k=5\n', results_1l, '\n')
print('Lasso Results, k=10\n', results_2l, '\n')
print(f"Best a for Ridge k=5:\t{abest1}\n")
print(f"Best a for Ridge k=10:\t{abest2}\n")
print(f"Best a for Lasso k=5:\t{abest3}\n")
print(f"Best a for Lasso k=10:\t{abest4}\n")
print(f"Max Test Acc for Ridge k=5:\t{max1}\n")
print(f"Max Test Acc for Ridge k=10:\t{max2}\n")
print(f"Max Test Acc for Lasso k=5:\t{max3}\n")
print(f"Max Test Acc or Lasso k=10:\t{max4}\n")


max1 = 0
max2 = 0
max3 = 0
max4 = 0

for vs in np.array([46,92,138]):
    for c in np.arange(0.001,1.00000001,0.1):
        ridge_lr = LogisticRegression(max_iter=5000, C = 1./c, penalty='l2')
        lasso_lr = LogisticRegression(max_iter=5000, solver = 'liblinear', C = 1./c, penalty='l1')

        results_1r = eval_mccv(x, y, vs, 5, ridge_lr)
        if results_1r['val-acc'] >= max1:
            abest1 = c
            vs1 = vs
            max1 = results_1r['val-acc']
        results_2r = eval_mccv(x, y,vs, 10, ridge_lr)
        if results_2r['val-acc'] >= max2:
            vs2=vs
            abest2 = c
            max2 = results_2r['val-acc']

        results_1l = eval_mccv(x, y, vs,5, lasso_lr)
        if results_1r['val-acc'] >= max3:
            abest3 = c
            vs3 = vs
            max3 = results_1l['val-acc']
        results_2l = eval_mccv(x, y, vs, 10, lasso_lr)
        if results_2l['val-acc'] >= max4:
            abest4 = c
            vs4 = vs
            max4 = results_2l['val-acc']

    

print('Starting 3(h)\n')
print('----------------------------------\n')
print('Ridge Results, k=5\n', results_1r, '\n')
print('Ridge Results, k=10\n', results_2r, '\n')
print('Lasso Results, k=5\n', results_1l, '\n')
print('Lasso Results, k=10\n', results_2l, '\n')
print(f"Best a for Ridge k=5:\t{abest1}\n")
print(f"Best a for Ridge k=10:\t{abest2}\n")
print(f"Best a for Lasso k=5:\t{abest3}\n")
print(f"Best a for Lasso k=10:\t{abest4}\n")
print(f"Max Test Acc for Ridge k=5:\t{max1}\n")
print(f"Max Test Acc for Ridge k=10:\t{max2}\n")
print(f"Max Test Acc for Lasso k=5:\t{max3}\n")
print(f"Max Test Acc or Lasso k=10:\t{max4}\n")
print(f"Value Size for Ridge k=5:\t{vs1}\n")
print(f"Value Size for Ridge k=10:\t{vs2}\n")
print(f"Value Size for Lasso k=5:\t{vs3}\n")
print(f"Value Size for Lasso k=10:\t{vs4}\n")

ridge_kf = LogisticRegression(max_iter=5000, C = 1./0.530, penalty='l2')
lasso_kf = LogisticRegression(max_iter=5000, solver='liblinear', C = 1./0.992, penalty='l1')

ridge_mc = LogisticRegression(max_iter=5000, C = 1./0.950, penalty='l2')
lasso_mc = LogisticRegression(max_iter=5000, solver = 'liblinear', C = 1./0.569, penalty='l1')

dict_rk = eval_kfold(x, y, 10, ridge_kf)
dict_lk = eval_kfold(x, y, 10, lasso_kf)

dict_rmc = eval_mccv(x, y, 92, 5, ridge_mc)
dict_lmc = eval_mccv(x, y, 92, 5, lasso_mc)

print('Starting 3(i)\n')
print('----------------------------------\n')
print('Ridge KF\n', dict_rk, '\n')
print('Lasso KF\n', dict_lk, '\n')
print('Ridge MC\n',dict_rmc, '\n')
print('Lasso MC\n',dict_lmc, '\n')