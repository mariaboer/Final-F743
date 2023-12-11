import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


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

    # This algorithm uses gridsearch to identify the optimal hyperparameters

    # Separate features (X) and target variable (y)
    X = df.drop('baseFare', axis=1)
    y = df['baseFare']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create transformers for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # Create the XGBoost Regressor model
    xgb_model = XGBRegressor()

    # Create a pipeline with encoding and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

    # Define hyperparameters for tuning
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_child_weight': [1, 3, 5],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    with open(os.path.join(rootpath, 'snapshots', 'gridsearch_bm.pkl'), 'wb') as file:
        pickle.dump(best_model, file)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    loading.info(f'Best Hyperparameters: {grid_search.best_params_}')
    loading.info(f'Mean Squared Error: {mse}')
    loading.info(f'R-squared: {r_squared}')

    # Once hyperparameters are identified we run the algorithm again with said parameters and increase the R-squared slightly

    # Separate features (X) and target variable (y)
    X = df.drop('baseFare', axis=1)
    y = df['baseFare']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create transformers for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # Use the best hyperparameters from GridSearchCV
    best_hyperparameters = {'model__colsample_bytree': 0.8,
                            'model__learning_rate': 0.1,
                            'model__max_depth': 7,
                            'model__min_child_weight': 1,
                            'model__n_estimators': 300,
                            'model__subsample': 0.8}

    # Create the XGBoost Regressor model with the best hyperparameters
    xgb_model = XGBRegressor(**best_hyperparameters)

    # Create a pipeline with encoding and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

    # Fit the model
    pipeline.fit(X_train, y_train)

    with open(os.path.join(rootpath, 'snapshots', 'gridsearch_bp.pkl'), 'wb') as file:
        pickle.dump(pipeline, file)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    loading.info("GridsearchCV XGBoost Regression Model: Return the Mean Squared Error and R-squared")

    loading.info(f'Best Hyperparameters: {best_hyperparameters}')
    loading.info(f'Mean Squared Error: {mse}')
    loading.info(f'R-squared: {r_squared}')

    # Calculate residuals
    residuals = y_test - y_pred

    # Plotting residual values
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot for KNN Regression')
    plt.savefig(os.path.join(rootpath, 'outputs', 'GridSearch - Residuals Plot.jpg'), format='jpeg')

    plt.clf()

    # Plotting actual vs. predicted values
    plt.scatter(y_test, y_pred, label='Predictions', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Predicted Base Fare')
    plt.title('Actual vs. Predicted Base Fare using GridSearch Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(rootpath, 'outputs', 'GridSearch - Actual vs. Predicted.jpg'), format='jpeg')

    results = pd.DataFrame(grid_search.cv_results_)
    scores = np.array(results.mean_test_score).reshape(len(param_grid['n_estimators']), len(param_grid['max_depth']))

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=param_grid['max_depth'], yticklabels=param_grid['n_estimators'], cmap="viridis")
    plt.xlabel('Max Depth')
    plt.ylabel('Number of Estimators')
    plt.title('Grid Search Results (Mean Test Score)')
    plt.savefig(os.path.join(rootpath, 'outputs', 'GridSearchResults.png'))

    loading.info("GridsearchCV XGBoost completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GridSearchCV XGBoost Regressio')
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
