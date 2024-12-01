import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 3 - Classification\Section 16 - Support Vector Machine (SVM)\Python\Social_Network_Ads.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,X_train,y_train,cv=10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())

from sklearn.model_selection import GridSearchCV
parameters = [
    {"C":[1.0,2.0,3.0,4.0,0.5,0.6,0.1,0.01],"kernel":["rbf"],"gamma":[0.5,0.4,0.3,0.6,0.7,0.8,0.9]},
    {"C":[1.0,2.0,3.0,0.1,0.01],"kernel":["linear"]}
]
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,n_jobs=-1,cv=10,scoring="accuracy")
grid_search.fit(X_train,y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

