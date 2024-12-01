import pandas as pd
data = pd.read_csv(r"C:\Users\Rajarshi Basak\Study_Metarials\Machine Learning\Machine Learning\Machine Learning A-Z Dataset\Part 8 - Deep Learning\Section 39 - Artificial Neural Networks (ANN)\Python\Churn_Modelling.csv")
print(data.iloc[:,3:13].head())

X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
encoder = LabelEncoder()
X[:,2] = encoder.fit_transform(X[:,2:3])
encoder = ColumnTransformer(transformers=[("Encoder",OneHotEncoder(),[1])],remainder="passthrough")
X = encoder.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#building ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
#add input layer
#classifier.add(tf.keras.Input(shape=11))
#add two hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))
#add output layer
classifier.add(Dense(units=1, kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train, y_train,batch_size=10,epochs=20)
y_pred = (classifier.predict(X_test) > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
