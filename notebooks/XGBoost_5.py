import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
        pickle_file = os.path.join(rootpath, 'snapshots', 'preprocessed.pkl')
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)

    # Separate features (X) and target variable (y)
    X = df.drop('baseFare', axis=1)
    y = df['baseFare']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    logging.debg(f"Categorical columns: {categorical_columns}")
    # Create transformers for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # Create the XGBoost Regressor model
    model = xgb.XGBRegressor()

    # Create a pipeline with encoding and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the model
    logging.debug("Fitting the model")
    pipeline.fit(X_train, y_train)
    logging.debug("Model fit complete, saving model")

    with open(os.path.join(rootpath, 'snapshots', 'xgboost.pkl'), 'wb') as file:
        pickle.dump(pipeline, file)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    logging.info("XGBoost Regression Model: Return the Mean Squared Error and R-squared")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    num_trees_to_plot = 5
    r2 = r2_score(y_test, y_pred)

    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'R-squared: {r2}')

    residuals = y_test - y_pred

    # Plotting residual values
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot for KNN Regression')
    plt.savefig(os.path.join(rootpath, 'outputs', 'XGBoost - Residuals Plot.jpg'), format='jpeg')

    plt.clf()

    # Plotting actual vs. predicted values
    plt.scatter(y_test, y_pred, label='Predictions', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Predicted Base Fare')
    plt.title('Actual vs. Predicted Base Fare using XGBoost Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(rootpath, 'outputs', 'XGBoost - Actual vs. Predicted.jpg'), format='jpeg')

    num_trees_to_plot = 2

    plt.figure(figsize=(18, 12))
    for i in range(num_trees_to_plot):
        plt.subplot(1, num_trees_to_plot, i + 1)
        xgb.plot_tree(pipeline.named_steps['model'], num_trees=i, rankdir='LR')
        plt.title(f'Tree {i + 1}')

    plt.savefig(os.path.join(rootpath, 'outputs', 'XGBoost-Tree.png'))

    logging.debug("XGBoost Regression Model: Return the Mean Squared Error and R-squared")
    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'R-squared: {r_squared}')
    logging.info("XGBoost completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost Regression Model')
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
