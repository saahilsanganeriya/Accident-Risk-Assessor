import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('../data/balanced_data.csv')

print(data.keys())
X = data.drop('accident binary', axis=1)
y = data['accident binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

model = Sequential()
model.add(Dense(27, activation='tanh', input_shape=(27,)))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)

y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=1)
print(score)