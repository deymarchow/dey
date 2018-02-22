import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.datasets  import load_boston
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.svm import SVC

data_diab = pd.read_csv("diabetes.csv",header=0)
print (data_diab)

X=data_diab['Age']
Y=data_diab['BloodPressure']

plt.scatter(data_diab["Age"], data_diab["BloodPressure"])
plt.xlabel("Age")
plt.ylabel("BloodPressure")
plt.show()
