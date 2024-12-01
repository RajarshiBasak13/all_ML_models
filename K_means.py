import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 4 - Clustering\Section 24 - K-Means Clustering\Python\Mall_Customers.csv")
X = df.iloc[:,3:].values
print(X)
WCSS_li = []

from sklearn.cluster import KMeans
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,n_init=10,max_iter=300,init="k-means++")
    kmeans.fit(X)
    WCSS_li.append(kmeans.inertia_)
plt.plot(range(1,11),WCSS_li)
plt.title("WCSS Graph")
plt.xlabel("no of cluster")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5,n_init=10,init="k-means++",max_iter=300,random_state=0)
kmeans_perd = kmeans.fit_predict(X)
print(kmeans_perd)

plt.scatter(X[kmeans_perd==0,0],X[kmeans_perd==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(X[kmeans_perd==1,0],X[kmeans_perd==1,1],s=100,c="green",label="Cluster 2")
plt.scatter(X[kmeans_perd==2,0],X[kmeans_perd==2,1],s=100,c="blue",label="Cluster 3")
plt.scatter(X[kmeans_perd==3,0],X[kmeans_perd==3,1],s=100,c="pink",label="Cluster 4")
plt.scatter(X[kmeans_perd==4,0],X[kmeans_perd==4,1],s=100,c="yellow",label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="black",label="centroids")
plt.show()
