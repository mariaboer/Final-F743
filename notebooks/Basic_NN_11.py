import argparse
import os
import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def str_or_none(value):
    return value if value is None else str(x)


def main(rootpath):
    if rootpath is None:
        rootpath = os.path.dirname(__file__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
    # df = pd.read_csv(datafile)

    picklefile = os.path.join(rootpath, 'snapshots', 'preprocessed.pkl')
    with open(picklefile, 'rb') as file:
        df = pickle.load(file)

    X = df.drop(['baseFare'], axis=1)
    y = df['baseFare']

    X = StandardScaler().fit_transform(X)

    myNN = Sequential()
    n_x = X.shape[1]
    myNN.add(Dense(x_n / 2, activation='relu', input_shape=(n_x,)))
    myNN.add(Dense(x_n / 4, activation='relu'))
    myNN.add(Dense(x_n / 2, activation='relu'))
    myNN.add(Dense(1, activation='sigmoid'))

    myNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    X_n, X_t, Y_n, Y_t = train_test_split(X, Y, random_state=100, test_size=0.3)
    checkpoint_filepath = os.path.join(rootpath, 'snapshots', 'Basic_NN.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    myNN.fit(X_n, Y_n, epochs=500, callbacks=[model_checkpoint_callback])
    myNN.load_weights(checkpoint_filepath)

    with open(os.path.join(rootpath, 'snapshots', 'Basic_NN.pkl'), 'wb') as file:
        pickle.dump(myNN, file)

    loss, acc = myNN.evaluate(X_t, Y_t, verbose=0)

    print("Neural Network Model: Return the Loss and Accuracy")
    print('Accuracy is: ' + str(acc))
    print('Loss is: ' + str(loss))
    print("Neural Network completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Neural Network')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    args = parser.parse_args()
    try:
        main()
    except KeyboardInterrupt:
        print('Program terminated by user')
        exit(-1)
    except Exception as e:
        print(e)
        print('Error running the program')
