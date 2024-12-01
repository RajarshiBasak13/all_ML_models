import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 6 - Polynomial Regression\Python\Position_Salaries.csv")
X = data_df.iloc[:,1:-1].values
y = data_df.iloc[:,-1].values
print(X,y)

from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)

plt.scatter(X,y,c="red")
plt.plot(X,linReg.predict(X),c = 'blue')
plt.title("Level vs Salary(simple Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree=4)
X_trans = polyFeature.fit_transform(X)
print(X_trans)
linReg2 = LinearRegression()
linReg2.fit(X_trans,y)
X_grid = np.arange(X.min(),X.max(),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,c="red")
plt.plot(X_grid,linReg2.predict(polyFeature.fit_transform(X_grid)),c="blue")
plt.title("Level vs Salary(Polynomial linear regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

print(linReg.predict([[6.5]]))
print(linReg2.predict(polyFeature.fit_transform([[6.5]])))



