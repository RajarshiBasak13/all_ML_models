import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 3 - Classification\Section 17 - Kernel SVM\Python\Social_Network_Ads.csv")
X=data_df.iloc[:,:-1].values
y=data_df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.decomposition import KernelPCA
feature_extraction = KernelPCA(n_components=2,kernel="rbf")
X_train = feature_extraction.fit_transform(X_train)
X_test = feature_extraction.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))