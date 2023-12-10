import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

rootpath = os.path.dirname(__file__)
# datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
# df = pd.read_csv(datafile)

picklefile = os.path.join(rootpath, 'preprocessed.pkl')
with open(picklefile, 'rb') as file:
    df = pickle.load(file)

#This algorithm uses gridsearch to identify the optimal hyperparameters

from sklearn.model_selection import GridSearchCV


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

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')


#Once hyperparameters are identified we run the algorithm again with said parameters and increase the R-squared slightly

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

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f'Best Hyperparameters: {best_hyperparameters}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')