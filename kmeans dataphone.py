import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans


dataset = pd.read_csv('diabetes.csv', header=0)
x = dataset.iloc[:, [1, 2, 3, 4]].values

diab = []

for i in range(1, 4):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)# funcion split, funcion o dataframe .split aplay,
    diab.append(kmeans.inertia_)

kmeans = KMeans(n_clusters =4, init = 'k-means++', max_iter = 10, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Pregnancies')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Glucose')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green', label = 'BloodPressure')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'YELLOW', label = 'SkinThickness')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'PINK', label = 'Insulin')
plt.scatter(x[y_kmeans == 5, 0], x[y_kmeans == 5, 1], s = 50, c = 'PURPLE', label = 'BMI')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'BLACK', label = 'Centroids')
plt.xlabel('')
plt.ylabel('')
plt.show()
plt.legend()