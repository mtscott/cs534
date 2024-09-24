import pandas as pd
import numpy as np
from q2 import preprocess_data
from elastic import *

energy_train = pd.read_csv("energydata/energy_train.csv")
energy_val = pd.read_csv("energydata/energy_val.csv")
energy_test = pd.read_csv("energydata/energy_test.csv")

print(f"energy_test shape: {energy_test.shape}")
print(energy_train.sample(3))

"""
1. delete date column
2. feature engineer 'date' column to make it numeric
"""
# pre_energy_train, pre_energy_val, pre_energy_test = preprocess_data(energy_train, energy_val, energy_test)
