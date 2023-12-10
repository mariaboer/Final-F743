from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import pickle
import os
import pandas as pd

rootpath = os.path.dirname(__file__)
# datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
# df = pd.read_csv(datafile)

picklefile = os.path.join(rootpath, 'preprocessed.pkl')
with open(picklefile, 'rb') as file:
    df = pickle.load(file)

# Step 1: One-hot encode non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['int', 'float']).columns

one_hot_encoder = OneHotEncoder(sparse=False) #, drop='first')  # 'drop' is set to 'first' to avoid the dummy variable trap
encoded_columns = pd.DataFrame(one_hot_encoder.fit_transform(df[non_numeric_columns]))
encoded_columns.columns = one_hot_encoder.get_feature_names_out(non_numeric_columns)

df_PCA = pd.concat([df.drop(columns=non_numeric_columns), encoded_columns], axis=1)

# Step 2: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_PCA)

# Step 3: Perform PCA
# You can specify the number of components you want to keep
num_components = 24  # Change this to your desired number of components
pca = PCA(n_components=num_components)
pca_result = pca.fit_transform(scaled_data)

# Now, pca_result contains the transformed data after PCA
# You can convert it back to a DataFrame if needed
pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(num_components)])

# Print the resulting DataFrame
print(f"PCA results from number of components: {num_components}")
print(pca_dataframe.head())

#Perform PCA with automatic determination of components
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Determine the number of components based on a threshold (e.g., 95% variance explained)
cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
num_components = sum(cumulative_variance_ratio >= 0.95)

# Apply PCA with the determined number of components
pca = PCA(n_components=num_components)
pca_result = pca.fit_transform(scaled_data)

# Now, pca_result contains the transformed data after PCA
pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(num_components)])

# Print the resulting DataFrame and explained variance
print(pca_dataframe.head())
print(f"PCA results from number of components: {num_components}")
print(f"Cumulative explained variance: {cumulative_variance_ratio[num_components - 1]:.2%}")

