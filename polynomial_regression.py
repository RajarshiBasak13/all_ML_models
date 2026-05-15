import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")
X = data_df.iloc[:,1:-1].values
y = data_df.iloc[:,-1].values
print(X,y)

from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)

"""plt.scatter(X,y,c="red")
plt.plot(X,linReg.predict(X),c = 'blue')
plt.title("Level vs Salary(simple Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
"""
from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree=4)
X_trans = polyFeature.fit_transform(X)
print(X_trans)
linReg2 = LinearRegression()
linReg2.fit(X_trans,y)
X_grid = np.arange(X.min(),X.max(),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

from sklearn.metrics import mean_squared_error
print("reporttttt",mean_squared_error(y, linReg2.predict(X_trans)))

import dagshub
dagshub.init(repo_owner='RajarshiBasak13', repo_name='all_ML_models', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  
mlflow.set_tracking_uri('https://dagshub.com/RajarshiBasak13/all_ML_models.mlflow')
mlflow.set_experiment("Polynomial_regression")
with mlflow.start_run():
    mlflow.log_param("model", "polynomial_regression")
    mlflow.log_param("degree", 4)
    mlflow.log_metric("mse",mean_squared_error(y, linReg2.predict(X_trans)))
    mlflow.sklearn.log_model(sk_model=linReg2,name="polynomial_regression")


model_name = 'polynomial_regression'
version = '1'
loaded_model = mlflow.sklearn.load_model(model_uri=f'models:/{model_name}/{version}')
print(loaded_model.predict(polyFeature.fit_transform([[6.5]])))



