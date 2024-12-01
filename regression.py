import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n\n----------------------Regression-------------------------------")
data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv")
X = data_df.iloc[:,:-1].values
y = data_df.iloc[:,1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
X = imputer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_train_pred = regressor.predict(X_train)

#plot Age Vs Salary [Training Set]
plt.scatter(X_train,y_train, color="Red")
plt.plot(X_train,y_train_pred, color="blue")
plt.title("Age Vs Salary [Training Set]")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

#plot Age Vs Salary [Test Set]
plt.scatter(X_test,y_test, color="Red")
plt.plot(X_train,y_train_pred, color="blue")
plt.title("Age Vs Salary [Test Set]")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()