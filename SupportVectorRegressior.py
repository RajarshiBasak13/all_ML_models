import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 7 - Support Vector Regression ("
                      r"SVR)\Python\Position_Salaries.csv")
X = data_df.iloc[:, 1:-1].values
y = data_df.iloc[:, -1].values

#feature Scaling
from sklearn.preprocessing import StandardScaler
scalar_x = StandardScaler()
X = scalar_x.fit_transform(X)
scalar_y = StandardScaler()
len_y = len(y)
y = np.array(y).reshape((len_y,1))
y = scalar_y.fit_transform(y).reshape((len_y))
print(y)
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")  # we are using gausian svr thats why rbf, we will not use linear as we know this relation is non-linear,so we will use rbf or poly
regressor.fit(X,y)

X_grid = np.arange(scalar_x.inverse_transform(X).min(),scalar_x.inverse_transform(X).max(),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
print(regressor.predict(scalar_x.transform(X_grid)))
plt.scatter(scalar_x.inverse_transform(X), scalar_y.inverse_transform(np.array(y).reshape((len_y,1))), c='red')
plt.plot(X_grid, scalar_y.inverse_transform(np.array(regressor.predict(scalar_x.transform(X_grid))).reshape((len(X_grid),1))), c="blue")
plt.title("Age Vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

print(scalar_y.inverse_transform(np.array(regressor.predict(scalar_x.transform(np.array([6.5]).reshape((1,1))))).reshape((1,1))))

