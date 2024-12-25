import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data "
                      r"Preprocessing --------------------\Python\Data.csv")
# X = data_df[list(data_df.columns)[:3]]
print("Train Data and Test data:-");
X = data_df.iloc[:, :3].values
print(X)
y = data_df.iloc[:,-1].values
print(y)

print("Filling NaN in numeric column with imputer :-")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

print("Filling NaN in categorical columns :-")
#now we need to work with categorical column to preprocess and replace null value.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)
print(y)
col_transfer = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
X = col_transfer.fit_transform(X)
print(X)

print("Train_Test Split :-")
#train test data split
from sklearn.model_selection import train_test_split;
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train,"\n",X_test)

print("After Feature Scaling :- ")
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform((X_test))
print(X_train)
print(X_test)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning "
                      r"A-Z Dataset\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data "
                      r"Preprocessing --------------------\Python\Data.csv")
data = data_df.values
X = data[:,:-1]
y = data[:,-1]

from sklearn.impute import SimpleImputer
X[:,1:] = np.array(SimpleImputer(missing_values=np.nan,strategy="mean").fit_transform(X[:,1:]))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
y = LabelEncoder().fit_transform(y)
X = np.array(ColumnTransformer(transformers=[("Country_encode",OneHotEncoder(),[0])], remainder="passthrough").fit_transform(X)).astype(int)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

np.set_printoptions(suppress=True)

print(X,y)



