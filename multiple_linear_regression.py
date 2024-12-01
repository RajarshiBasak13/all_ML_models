import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")
print(data_df.head())
X = data_df.iloc[:, :-1].values
y = data_df.iloc[:, -1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_transfer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
X = column_transfer.fit_transform(X)
# now X contains encoded categorical columns with a dummy trap column.
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(X_train.shape)


X_train = np.append(np.ones((40,1)).astype(np.int32), X_train, axis=1)
X_train1 = X_train[:, [0,1,2,3,4,5]]
train_df1 = pd.DataFrame(np.append(X_train1, np.reshape(y_train,(40,1)), axis=1), columns=["a0","a1","a2","a3","a4","a5","a6"]).astype(np.int32)
import statsmodels.formula.api as sm
statsmodel_regressor1 = sm.ols("a6 ~ a0 + a1 + a2 + a3 + a4 + a5",train_df1).fit()
print(statsmodel_regressor1.summary())

X_train1 = X_train[:, [0,1,3,4,5]]
train_df1 = pd.DataFrame(np.append(X_train1, np.reshape(y_train,(40,1)), axis=1), columns=["a0","a1","a3","a4","a5","a6"]).astype(np.int32)
import statsmodels.formula.api as sm
statsmodel_regressor1 = sm.ols("a6 ~ a0 + a1 + a3 + a4 + a5",train_df1).fit()
print(statsmodel_regressor1.summary())

X_train1 = X_train[:, [0,3,4,5]]
train_df1 = pd.DataFrame(np.append(X_train1, np.reshape(y_train,(40,1)), axis=1), columns=["a0","a3","a4","a5","a6"]).astype(np.int32)
import statsmodels.formula.api as sm
statsmodel_regressor1 = sm.ols("a6 ~ a0 + a3 + a4 + a5",train_df1).fit()
print(statsmodel_regressor1.summary())

X_train1 = X_train[:, [0,3,5]]
train_df1 = pd.DataFrame(np.append(X_train1, np.reshape(y_train,(40,1)), axis=1), columns=["a0","a3","a5","a6"]).astype(np.int32)
import statsmodels.formula.api as sm
statsmodel_regressor1 = sm.ols("a6 ~ a0 + a3 + a5",train_df1).fit()
print(statsmodel_regressor1.summary())

X_train1 = X_train[:, [0,3]]
train_df1 = pd.DataFrame(np.append(X_train1, np.reshape(y_train,(40,1)), axis=1), columns=["a0","a3","a6"]).astype(np.int32)
import statsmodels.formula.api as sm
statsmodel_regressor1 = sm.ols("a6 ~ a0 + a3",train_df1).fit()
print(statsmodel_regressor1.summary())