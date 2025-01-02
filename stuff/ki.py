import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

df = pd.read_csv('/Users/jonathan.bach/Documents/UNI/DataLiteracy/kitrainINTCSV.csv', sep = ";")
X = df.drop('Stirbt', axis = 1)

y = df['Stirbt']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=40)

model = Sequential()

model.add(Input(shape=(99,)))
model.add(Dense(100,activation= "sigmoid"))
model.add(Dense(50,activation= "sigmoid"))
model.add(Dense(1,activation= "sigmoid"))

model.compile(optimizer="rmsprop",loss = "binary_crossentropy",metrics = ["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=64)

input_data = [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

input_data = np.array(input_data).reshape(1, -1)

print(model.evaluate(X_test, y_test))
prediction = model.predict(input_data)
print(prediction)

model.save("model.keras")