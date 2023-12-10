import argparse
import os
import pickle

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Using KNN to make predictions based on the similarity of the datapoints in the dataset.

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

    # Assuming you have a DataFrame named 'df' with the necessary columns
    # Extracting features (independent variables) and the target variable
    X = df.drop(['baseFare'], axis=1)
    y = df['baseFare']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    # Separating numeric and categorical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

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
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', KNeighborsRegressor(n_neighbors=5))])

    # Fitting the pipeline
    pipeline.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = pipeline.predict(X_test)
    with open(os.path.join(rootpath, 'snapshots', 'knn.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("K-Nearest Neighbors Regression Model: Return the Mean Squared Error and R-squared")
    print(f'KNN - Mean Squared Error: {mse}')
    print(f'KNN - R-squared: {r2}')

    # Calculate residuals
    residuals = y_test - y_pred

    # Plotting residual values
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot for KNN Regression')
    plt.savefig(os.path.join(rootpath, 'outputs', 'KNN - Residuals Plot.jpg'), format='jpeg')

    # Plotting actual vs. predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Base Fare')
    plt.ylabel('Predicted Base Fare')
    plt.title('Actual vs. Predicted Base Fare using KNN Regression')
    plt.savefig(os.path.join(rootpath, 'outputs', 'KNN - Actual vs. Predicted.jpg'), format='jpeg')

    print('KNN Regression Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Regression')
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
