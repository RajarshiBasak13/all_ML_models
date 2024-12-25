import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 8 - Deep Learning\Section 39 - Artificial Neural Networks (ANN)\Python\Churn_Modelling.csv")
print(df.iloc[:,3:13].head().to_string())
data = df.values

X = data[:,3:-1]
y = data[:,-1]

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

col_trans = ColumnTransformer(transformers=[("gender_trans",OrdinalEncoder(),[2]),("Geography_trans",OneHotEncoder(),[1])],remainder="passthrough")
X = col_trans.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = np.array(y_train,dtype=np.float32)
y_test = np.array(y_test,dtype=np.float32)

from tf_keras.models import Sequential
from tf_keras.layers import Dense
ANN = Sequential()
ANN.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
ANN.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
ANN.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
ANN.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")
ANN.fit(X_train,y_train,batch_size=20,epochs=20)
pred = ANN.predict(X_test)
print(np.array(pred.reshape(len(pred))).round()==y_test)
print(np.array(pred.reshape(len(pred))).round())
print(y_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(np.array(pred.reshape(len(pred))).round(),y_test))
print(accuracy_score(np.array(pred.reshape(len(pred))).round(),y_test))




