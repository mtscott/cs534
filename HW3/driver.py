import numpy as np
import pandas as pd

# Import Energy Data Set
energy_train = pd.read_csv("energydata/energy_train.csv")
energy_val = pd.read_csv("energydata/energy_val.csv")
energy_test = pd.read_csv("energydata/energy_test.csv")


# Import Spam Data Set
spam_train = pd.read_csv("spam/spam.train.dat", sep = '\s',engine='python').to_numpy()
spam_test = pd.read_csv("spam/spam.test.dat", sep = '\s', engine='python').to_numpy()

spam_trainx, spam_trainy = spam_train[:,:-1], spam_train[:,-1]
spam_testx, spam_testy = spam_test[:,:-1], spam_test[:,-1]

