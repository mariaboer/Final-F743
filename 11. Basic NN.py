import tensorflow
print(tensorflow.__version__)

import pandas as pd
import pickle
import os

rootpath = os.path.dirname(__file__)
# datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
# df = pd.read_csv(datafile)

picklefile = os.path.join(rootpath, 'preprocessed.pkl')
with open(picklefile, 'rb') as file:
    df = pickle.load(file)

X = df.drop(['baseFare'], axis=1)
y = df['baseFare']

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from tensorflow.keras import Sequential
myNN = Sequential()

n_x = X.shape[1]

from tensorflow.keras.layers import Dense
myNN.add(Dense(10, activation = 'relu', input_shape = (n_x,)))
myNN.add(Dense(10, activation = 'relu'))
myNN.add(Dense(10, activation = 'relu'))
myNN.add(Dense(1, activation = 'sigmoid'))


myNN.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics=['accuracy'])

from sklearn.model_selection import train_test_split
X_n, X_t, Y_n, Y_t = train_test_split(X, Y, random_state = 100,
                                      test_size = 0.3)

myNN.fit(X_n, Y_n, epochs = 5)

loss, acc = myNN.evaluate(X_t, Y_t, verbose = 0)
print('Accuracy is: ' + str(acc))