import argparse
import os
import pickle

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

    # Step 1: One-hot encode non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['int', 'float']).columns

    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_columns = pd.DataFrame(one_hot_encoder.fit_transform(df[non_numeric_columns]))
    encoded_columns.columns = one_hot_encoder.get_feature_names_out(non_numeric_columns)

    df_PCA = pd.concat([df.drop(columns=non_numeric_columns), encoded_columns], axis=1)

    # Step 2: Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_PCA)

    # Step 3: Perform PCA
    # Specify what columns need to stay
    num_components = 24  # Change this to your desired number of components
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(scaled_data)

    # Now, pca_result contains the transformed data after PCA
    # Convert to PCA dataframe
    pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(num_components)])
    print(pca_dataframe.head())

    # Determine the number of components based on a threshold (e.g., 95% variance explained)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    num_components = sum(cumulative_variance_ratio >= 0.95)

    # Apply PCA with the determined number of components
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(scaled_data)

    with open(os.path.join(rootpath, 'snapshots', 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # Now, pca_result contains the transformed data after PCA
    pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(num_components)])

    # Print the resulting DataFrame and explained variance
    print(pca_dataframe.head())
    print(f"PCA results from number of components: {num_components}")
    print(f"Cumulative explained variance: {cumulative_variance_ratio[num_components - 1]:.2%}")

    # Plot individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Component')
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Explained Variance.jpg'), format='jpeg')

    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Cumulative Explained Variance.jpg'), format='jpeg')

    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: Reduced-dimensional Scatter Plot')
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Reduced dimensional Scatter Plot.jpg'), format='jpeg')

    print("PCA completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Principal Component Analysis')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    args = parser.parse_args()
    try:
        main(args.root_path)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        exit(-1)
    except Exception as e:
        print(e)
        exit(1)
