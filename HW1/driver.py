import pandas as pd
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from q2 import *
from elastic import *
import datetime
from scipy import stats

# Read Data In
energy_train = pd.read_csv("energydata/energy_train.csv")
energy_val = pd.read_csv("energydata/energy_val.csv")
energy_test = pd.read_csv("energydata/energy_test.csv")

"""
1. delete date column
2. feature engineer 'date' column to make it numeric
"""
energy_train['date'] = pd.to_datetime(energy_train['date'])
energy_val['date'] = pd.to_datetime(energy_val['date'])
energy_test['date'] = pd.to_datetime(energy_test['date'])

energy_train['date'] = pd.to_numeric(energy_train['date'])//6e10
energy_val['date'] = pd.to_numeric(energy_val['date'])//6e10
energy_test['date'] = pd.to_numeric(energy_test['date'])//6e10
 
# Display the data
print(energy_train.head())
energy_train.info()

# Preprocess data
pre_energy_train, pre_energy_val, pre_energy_test = preprocess_data(energy_train, energy_val, energy_test)


print(pre_energy_train)
print(f"shape of processed train:{pre_energy_train.shape}")

# Split Data from feature
trainx = np.concatenate((pre_energy_train[:,2:],pre_energy_train[:,:0]), axis = 1)
trainy = pre_energy_train[:,1]
#trainy = trainy.reshape(-1,1)
print(f"shape of train x:{trainx.shape}")
print(f"shape of train y:{trainy.shape}")

valx = np.concatenate((pre_energy_val[:,2:],pre_energy_val[:,:0]), axis = 1)
valy = pre_energy_val[:,1]

testx = np.concatenate((pre_energy_test[:,2:],pre_energy_test[:,:0]), axis = 1)
testy = pre_energy_test[:,1]


# Eval Linear 1
print("Starting Linear 1\n")
print('------------------------')
errdict1 = eval_linear1(trainx, trainy, valx, valy, testx, testy)
print(errdict1)

# Eval Linear 2
print("Starting Linear 2\n")
print('------------------------')
errdict2 = eval_linear2(trainx, trainy, valx, valy, testx, testy)

print(errdict2)

lowRidge = 100.
lowLasso = 100.

for alpha in np.linspace(0.005, 0.5, 6):
    # Eval Ridge
    print(f"Starting Ridge for: $alpha$ = {alpha}")
    print('------------------------')
    errdictRidge = eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha)
    if errdictRidge['val-rmse'] < lowRidge:
        print(errdictRidge['val-rmse'])
        lowRidge = errdictRidge['val-rmse']
        aidxR = alpha
    print(errdictRidge)

#print(aidxR, lowRidge)

for alpha in np.linspace(0.05, 0.5, 6):
    # Eval Lasso
    print(f"Starting Lasso:{alpha}")
    print('------------------------')
    errdictLasso = eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha)
    print(errdictLasso)
    if errdictLasso['val-rmse'] < lowLasso:
        print(errdictLasso['val-rmse'])
        lowLasso = errdictLasso['val-rmse']
        aidxL = alpha

print(aidxL, lowLasso)

"""
Applying the Data to our own SGD.
"""

# Learning Rate Exploration

lowRidge = 9.095945e-13
lowLasso = 0.02792

etactr = 0
neta, nepoch = 6, 50
yRidge = np.zeros((neta, nepoch))
yLasso = np.zeros((neta, nepoch))

for eta in np.logspace(-10,0,neta):
    elastic1 = ElasticNet(lowRidge, 0.5, eta, 100, nepoch)
    errR = elastic1.train(trainx,trainy)
    elastic2 = ElasticNet(lowLasso, 0.5, eta, 100, nepoch)
    errL = elastic2.train(trainx,trainy)
    for epoch in np.arange(nepoch):
        yRidge[etactr, epoch] = errR[epoch]
        yLasso[etactr, epoch] = errL[epoch]
    
    etactr += 1


plt.figure(1)
x = np.arange(nepoch)
plt.plot(x, yRidge[0,:], label = "eta = 1e-10")
plt.plot(x, yRidge[1,:], label = 'eta = 1e-8')
plt.plot(x, yRidge[2,:], label = 'eta = 1e-6')
plt.plot(x, yRidge[3,:], label = "eta = 1e-4")
plt.plot(x, yRidge[4,:], label = 'eta = 1e-2')
plt.plot(x, yRidge[5,:], label = 'eta = 1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Rates for Ridge Regression')
plt.legend()
plt.show()

plt.figure(2)
x = np.arange(nepoch)
plt.plot(x, yLasso[0,:], label = "eta = 1e-10")
plt.plot(x, yLasso[1,:], label = 'eta = 1e-8')
plt.plot(x, yLasso[2,:], label = 'eta = 1e-6')
plt.plot(x, yLasso[3,:], label = "eta = 1e-4")
plt.plot(x, yLasso[4,:], label = 'eta = 1e-2')
plt.plot(x, yLasso[5,:], label = 'eta = 1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Rates for Lasso Regression')
plt.legend()
plt.show()


etahat = 1e-6
ntrain, p = np.shape(trainx)
nval, p = np.shape(valx)
ntest, p = np.shape(testx)

for alpha in np.linspace(0.0,1.0,11):
   elastic3 = ElasticNet(lowLasso, alpha, etahat, 100, nepoch)
   err3 = elastic3.train(trainx,trainy)
   betahat = elastic3.beta

   # Predicting
   predtrain = elastic3.predict(trainx)
   trainrmse = 1./np.sqrt(ntrain) * npla.norm((predtrain - trainy),2)
   res1 = stats.linregress(predtrain, trainy)
   trainr2 = res1.rvalue**2

   predval = elastic3.predict(valx)
   valrmse = 1./np.sqrt(nval) * npla.norm((predval - valy),2)
   res2 = stats.linregress(predval, valy)
   valr2 = res2.rvalue**2

   predtest = elastic3.predict(testx)
   testrmse = 1./np.sqrt(ntest) * npla.norm((predtest - testy),2)
   res3 = stats.linregress(predtest, testy)
   testr2 = res3.rvalue**2

   print(f"alpha value:{alpha}")
   print('--------------------\n')
   print(f"train RMSE:\t {trainrmse:4.5f}")
   print(f"train r2:\t {trainr2:0.5f}")

   print(f"val RMSE:\t {valrmse:4.5f}")
   print(f"val r2:\t\t {valr2:0.5f}")

   print(f"test RMSE:\t {testrmse:4.5f}")
   print(f"test r2:\t {testr2:0.5f}\n")


print(f"My Ridge Coeffs:\t {elastic1.beta}\n")
print(f"Their Ridge Coeffs:\t {elastic1.beta}\n")
print(f"My Lasso Coeffs:\t {elastic2.beta}\n")
print(f"Their Lasso Coeffs:\t {elastic1.beta}\n")

 



