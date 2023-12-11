import argparse
import logging
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def configure_logging(level=logging.INFO, log_path=None):
    if log_path is None:
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
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
    model = XGBRegressor()

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

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

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
    configure_logging(logging.DEBUG, args.root_path)
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
