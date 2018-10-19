# -*- coding: utf-8 -*-
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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

Names=np.array(TELE_OP)

PROSS= TSNE(n_components=2).fit_transform(Names)
RES=pd.DataFrame(PROSS)

#CLUSTERS Y ANALISIS DE SILUETAS
range_n_clusters = [2, 3, 4]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    
    ax1.set_xlim([-0.1, 1])
    
    ax1.set_ylim([0, len(PROSS) + (n_clusters + 1) * 10])

    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(PROSS)

    
    silhouette_avg = silhouette_score(PROSS, cluster_labels)
    print("para los n_clusters =", n_clusters,
          "PROMEDIO DE SILUETA:", silhouette_avg)

    
    sample_silhouette_values = silhouette_samples(PROSS, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

       
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        
        y_lower = y_upper + 10  

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(PROSS[:, 0], PROSS[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

 
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("REDUCCION DIMENSIONAL")
    ax2.set_xlabel("---- 0 ----")
    ax2.set_ylabel("---- 1 ----")

    plt.suptitle(("siluetas "
                  "N de cluster = %d" % n_clusters))
    plt.show()

"""                 
#plt.scatter(X,Y)
plt.scatter(X,Y, c=[cm.Paired(c)for c in TELE_OP.tenure])

plt.xlabel('NAME 0')
plt.ylabel('NAME 1')
plt.show()
"""