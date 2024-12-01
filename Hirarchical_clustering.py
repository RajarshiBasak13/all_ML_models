import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 4 - Clustering\Section 24 - K-Means Clustering\Python\Mall_Customers.csv")
X = df.iloc[:,[3,4]].values
print(X)

import scipy.cluster.hierarchy as hc
hc.dendrogram(hc.linkage(X,method="ward"))
plt.title("dendogram")
plt.xlabel("salary")
plt.ylabel("score")
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_pred = ac.fit_predict(X)
print(y_pred)

plt.scatter(X[y_pred==0,0], X[y_pred==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_pred==1,0], X[y_pred==1,1],s=100,c="green",label="cluster 2")
plt.scatter(X[y_pred==2,0], X[y_pred==2,1],s=100,c="blue",label="cluster 3")
plt.scatter(X[y_pred==3,0], X[y_pred==3,1],s=100,c="pink",label="cluster 4")
plt.scatter(X[y_pred==4,0], X[y_pred==4,1],s=100,c="black",label="cluster 5")
plt.title("Hierarchical clustering")
plt.xlabel("salary")
plt.ylabel("score")
plt.show()