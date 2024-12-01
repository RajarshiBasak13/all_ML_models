import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 8 - Decision Tree "
                      r"Regression\Python\Position_Salaries.csv")
X = data_df.iloc[:, 1:-1].values
y = data_df.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

X_grid = np.arange(X.min(), X.max(), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, c="red")
plt.plot(X_grid, regressor.predict(X_grid), c="blue")
plt.title("Age Vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()
