# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.datasets  import load_boston
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.svm import SVC
boston = load_boston()

# np.maen.Y_pred
 
bos = pd.DataFrame(boston.data)
print (bos)

bos.columns = boston.feature_names
print (bos.columns)

bos["PRICE"]= boston.target

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']
info_X= len (X)
print (info_X)
info_Y= len (Y)
print (info_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)
print (Y_pred)
print (len(Y))
print (len (Y_pred))

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
 #regresion logisticca
print ("el promedio es :",np.mean(Y_pred))

list=[]
gp = Y
for p in gp:
    if p>np.mean(gp): list.append(1)
    else: list.append(0)
np.array(list)
print (list)
print (len(list))