import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def main(rootpath, loader):
    # Bring in data to pre-process
    if loader == 'DataFile':
        datafile = os.path.join(rootpath, 'data', 'Atlanta Prices.csv')
        df = pd.read_csv(datafile)
    else:
        pickle_file = os.path.join(rootpath, 'snapshots', 'chunk.pkl')
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)

    # Assuming you have a DataFrame named 'df' with the necessary columns
    # Extracting features (independent variables) and the target variable
    X = df.drop(['baseFare'], axis=1)
    y = df['baseFare']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    # Separating numeric and categorical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    logging.debug(f"Numeric Features: {numeric_features.columns}")
    logging.debug(f"Categorical Features: {categorical_features.columns}")
    # Creating transformers for numeric and categorical features

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combining transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Combining preprocessing and KNN model into a single pipeline
    logging.debug("Starting Pipeline")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', KNeighborsRegressor(n_neighbors=5))])

    # Fitting the pipeline
    pipeline.fit(X_train, y_train)
    logging.debug("Pipeline Fitted")
    # Predictions on the test set
    y_pred = pipeline.predict(X_test)
    with open(os.path.join(rootpath, 'snapshots', 'knn.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info("K-Nearest Neighbors Regression Model: Return the Mean Squared Error and R-squared")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'R-squared: {r2}')
    # Calculate residuals
    residuals = y_test - y_pred
    logging.debug("Saving residuals plot")
    # Plotting residual values
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot for KNN Regression')
    plt.legend()
    plt.savefig(os.path.join(rootpath, 'outputs', 'KNN - Residuals Plot.jpg'), format='jpeg')

    logging.debug("Saving actual vs. predicted plot")

    plt.clf()

    # Plotting actual vs. predicted values
    plt.scatter(y_test, y_pred, label='Predictions', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Predicted Base Fare')
    plt.title('Actual vs. Predicted Base Fare using Random Forest Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(rootpath, 'outputs', 'Random Forest - Actual vs. Predicted.jpg'), format='jpeg')

    print("Random Forest completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Regression')
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
        print('Program terminated by user')
        exit(-1)
    except Exception as e:
        print(e)
        print('Error running the program')
