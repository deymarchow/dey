# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

TELE_OP=pd.read_csv('TELE.csv',header=0)

TELE_OP.replace(" ", np.nan, inplace = True)
TELE_OP.dropna(subset=["TotalCharges"], axis=0, inplace = True)
TELE_OP.reset_index(drop = True, inplace = True)
cleanup_nums = {"PhoneService": {"Yes": 1, "No": 0}, "PaperlessBilling": {"Yes": 1, "No": 0}, "Churn": {"Yes": 1, "No": 0}}
TELE_OP.replace(cleanup_nums, inplace=True)
cleanup_nums = {"MultipleLines": {"Yes": 1, "No": 0, "No phone service": 2}}
TELE_OP.replace(cleanup_nums, inplace=True)
cleanup_nums = {"InternetService": {"Fiber optic": 1, "DSL": 2, "No": 0}}
TELE_OP.replace(cleanup_nums, inplace=True)
cleanup_nums = {"OnlineSecurity": {"Yes": 1, "No": 0, "No internet service": 2}, "DeviceProtection": {"Yes": 1, "No": 0, "No internet service": 2}, "TechSupport": {"Yes": 1, "No": 0, "No internet service": 2}, "StreamingTV": {"Yes": 1, "No": 0, "No internet service": 2}, "StreamingMovies": {"Yes": 1, "No": 0, "No internet service": 2}, "Contract": {"Month-to-month": 1, "Two year": 3, "One year": 2}, "PaymentMethod": {"Electronic check": 1, "Mailed check": 2, "Bank transfer (automatic)": 3, "Credit card (automatic)": 4}}
TELE_OP.replace(cleanup_nums, inplace=True)
cleanup_nums = {"OnlineBackup": {"Yes": 1, "No": 0, "No internet service": 2}}
TELE_OP.replace(cleanup_nums, inplace=True)
cleanup_nums = {"gender": {"Male": 1, "Female": 0}, "Partner": {"Yes": 1, "No": 0}, "Dependents": {"Yes": 1, "No": 0}}
TELE_OP.replace(cleanup_nums, inplace=True)

TELE_OP=TELE_OP.drop('customerID', axis=1)


PROSS= TSNE(n_components=2).fit_transform(TELE_OP)
RES=pd.DataFrame(PROSS)

print RES
X=np.array(RES[0])
Y=np.array(RES[1])
print X
print Y

#plt.scatter(X,Y)
plt.scatter(X,Y, c=[cm.Paired(c)for c in TELE_OP.tenure])

plt.xlabel('NAME 0')
plt.ylabel('NAME 1')
plt.show()
#kmeans
#agrupasion de clusters
#relacion de true

