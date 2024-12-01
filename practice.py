import pandas as pd

df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python\50_Startups.csv")
print(df)

data = df.values
X = data[:,:-1]
y = data[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9,random_state=0)

from sklearn.compose import _column_transformer
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
print(LR.score(X_test,y_test))