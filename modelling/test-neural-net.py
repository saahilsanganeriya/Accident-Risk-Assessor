import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = 'https://raw.githubusercontent.com/werowe/logisticRegressionBestModel/master/KidCreative.csv'
data = pd.read_csv(url, delimiter=',')
labels = data['Buy']
features = data.iloc[:,2:16]

X = features
y = np.ravel(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(14,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=8, batch_size=1, verbose=1)

y_pred = model.predict(X_test)
# print(y_pred.shape)
# print(y_test.shape)
# print(y_pred[:10])
# print(y_test[:10])
# score = np.sum(y_pred == y_test[:,np.newaxis]) / y_pred.shape[0]
# print(score)
print(X_test.shape)
print(y_test.shape)
score = model.evaluate(X_test, y_test, verbose=1)
print(score)
