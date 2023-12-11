import argparse
import os
import pickle

import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def configure_logging(level=logging.INFO, log_path=None):
    if log_path is None:
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_file = os.path.join(log_path, f"{os.path.dirname(os.path.realpath(__file__)).split(os.sep)[-1]}.log")
    if level == logging.INFO or logging.NOTSET:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    elif level == logging.DEBUG or level == logging.ERROR:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(filename)s function:%(funcName)s()\t[%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )


def str_or_none(value):
    return value if value is None else str(value)


def str_or_none(value):
    return value if value is None else str(value)


def main(rootpath, loader):
    # Bring in data to pre-process
    if loader == 'DataFile':
        datafile = os.path.join(rootpath, 'data', 'Atlanta Prices.csv')
        df = pd.read_csv(datafile)
    else:
        pickle_file = os.path.join(rootpath, 'snapshots', 'chunk.pkl')
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)

    X = df.drop(['baseFare'], axis=1)
    y = df['baseFare']
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create transformers for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])
    preprocessed = preprocessor.fit_transform(X)
    X = StandardScaler().fit_transform(preprocessed)

    myNN = Sequential()
    n_x = X.shape[1]
    myNN.add(Dense(x_n / 2, activation='relu', input_shape=(n_x,)))
    myNN.add(Dense(x_n / 4, activation='relu'))
    myNN.add(Dense(x_n / 2, activation='relu'))
    myNN.add(Dense(1, activation='sigmoid'))

    myNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
    checkpoint_filepath = os.path.join(rootpath, 'snapshots', 'Basic_NN.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    myNN.fit(X_train, y_train, epochs=500, callbacks=[model_checkpoint_callback])
    myNN.load_weights(checkpoint_filepath)

    with open(os.path.join(rootpath, 'snapshots', 'Basic_NN.pkl'), 'wb') as file:
        pickle.dump(myNN, file)

    loss, acc = myNN.evaluate(X_t, Y_t, verbose=0)

    print("Neural Network Model: Return the Loss and Accuracy")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Basic Neural Network - Mean Squared Error: {mse}')
    print(f'Basic Neural Network - R-squared: {r2}')
    print('Accuracy is: ' + str(acc))
    print('Loss is: ' + str(loss))

    # Calculate residuals
    residuals = Y_n - Y_t

    # Plotting residual values
    plt.scatter(Y_t, residuals)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot for Basic Neural Network')
    plt.savefig(os.path.join(rootpath, 'outputs', 'Basic Neural Network - Residuals Plot.jpg'), format='jpeg')

    # Plotting actual vs. predicted values
    plt.scatter(Y_t, Y_n, label='Predictions')
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Predicted Base Fare')
    plt.title('Actual vs. Predicted Base Fare using KNN Regression')
    plt.legend()
    plt.savefig(os.path.join(rootpath, 'outputs', 'KNN - Actual vs. Predicted.jpg'), format='jpeg')

    print("Neural Network completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Neural Network')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    parser.add_argument('--loader', type=str_or_none, help='Type of loader to use. Literals "DataFile" or "Memory".Default - Memory', default="Memory")
    args = parser.parse_args()
    if args.root_path is None:
        args.root_path = os.path.dirname(__file__)
    configure_logging(logging.DEBUG, os.path.join(args.root_path, 'logs'))
    if ['DataFile', 'Memory'] not in args.loader:
        logging.warning("Invalid loader. Valid loaders are 'DataFile' or 'Memory'. Defaulting to 'Memory'")
    args.loader = 'Memory'
    try:
        main(args.root_path, args.loader)
    except KeyboardInterrupt:
        loading.warning('Program terminated by user')
        exit(-1)
    except Exception as e:
        loading.error(e)
        loading.error('Error running the program')
