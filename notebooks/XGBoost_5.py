import argparse
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor


def str_or_none(value):
    return value if value is None else str(x)


def main(rootpath):
    if rootpath is None:
        rootpath = os.path.dirname(__file__)
    # datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
    # df = pd.read_csv(datafile)

    picklefile = os.path.join(rootpath, 'snapshots', 'preprocessed.pkl')
    with open(picklefile, 'rb') as file:
        df = pickle.load(file)

    df = df.reset_index()

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
    model = XGBRegressor()

    # Create a pipeline with encoding and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the model
    pipeline.fit(X_train, y_train)

    with open(os.path.join(rootpath, 'snapshots', 'xgboost.pkl'), 'wb') as file:
        pickle.dump(pipeline, file)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    print("XGBoost Regression Model: Return the Mean Squared Error and R-squared")

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r_squared}')

    print("XGBoost completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost Regression Model')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    args = parser.parse_args()
    try:
        main(args.root_path)
    except KeyboardInterrupt:
        print('Program terminated by user')
        exit(-1)
    except Exception as e:
        print(e)
        print('Error running the program')
