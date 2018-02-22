import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetesdataset =pd.read_csv("phone_dataset.csv",header=0)
print (diabetesdataset)
desdiabetes=pd.DataFrame(diabetesdataset)
X=desdiabetes['Pregnancies']
Y=desdiabetes['Age']
"""
x_train = tuple(range(768))
x_train = np.asarray(x_train)
y_train=tuple(range(768))
y_train = np.asarray(y_train)


x_train = x_train.reshape(768,1)


y_train.shape


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,)
print (X_train.shape)
print (Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

lg = LinearRegression()

lg.fit(X_train,Y_train)

Y_pred = lg.predict(X_test)
print (Y_pred)
print (len(Y))
print (len (Y_pred))

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()
"""