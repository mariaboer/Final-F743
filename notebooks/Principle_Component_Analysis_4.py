import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
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

    # Step 1: One-hot encode non-numeric columns
    logging.debug("Converting non-numerics to one-hot encoded columns")
    non_numeric_columns = df.select_dtypes(exclude=['int', 'float']).columns
    logging.debug(f"Non-numeric columns: {non_numeric_columns}")
    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_columns = pd.DataFrame(one_hot_encoder.fit_transform(df[non_numeric_columns]))
    encoded_columns.columns = one_hot_encoder.get_feature_names_out(non_numeric_columns)
    df_PCA = pd.concat([df.drop(columns=non_numeric_columns), encoded_columns], axis=1)

    # Step 2: Scale the data
    logging.debug("Scaling the data")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_PCA)

    # Step 3: Perform PCA
    # Specify what columns need to stay
    logging.debug("Performing PCA")
    num_components = 24  # Change this to your desired number of components
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(scaled_data)

    # Now, pca_result contains the transformed data after PCA
    # Convert to PCA dataframe
    logging.debug("Applying PCA with variance ratio threshold")
    pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(num_components)])
    logging.debug(pca_dataframe.head())

    # Determine the number of components based on a threshold (e.g., 95% variance explained)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    num_components = sum(cumulative_variance_ratio >= 0.95)
    logging.debug(f"PCA results from number of components: {num_components}")
    # Apply PCA with the determined number of components
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(scaled_data)

    # Save the PCA model
    logging.debug("Saving PCA model")
    with open(os.path.join(rootpath, 'snapshots', 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # Now, pca_result contains the transformed data after PCA
    pca_dataframe = pd.DataFrame(data=pca_result, columns=[f'PC{i + 1}' for i in range(num_components)])

    # Print the resulting DataFrame and explained variance
    logging.debug(pca_dataframe.head())
    logging.info(f"PCA results from number of components: {num_components}")
    logging.info(f"Cumulative explained variance: {cumulative_variance_ratio[num_components - 1]:.2%}")

    # Plot individual explained variance
    logging.debug("Plotting PCA results")
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Component')
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Explained Variance.jpg'), format='jpeg')

    # Plot cumulative explained variance
    logging.debug("Plotting cumulative explained variance")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Cumulative Explained Variance.jpg'), format='jpeg')

    # Plot reduced-dimensional scatter plot
    logging.debug("Plotting reduced-dimensional scatter plot")
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: Reduced-dimensional Scatter Plot')
    plt.savefig(os.path.join(rootpath, 'outputs', 'PCA - Reduced dimensional Scatter Plot.jpg'), format='jpeg')

    logging.info("PCA completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Principal Component Analysis')
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
        print("Keyboard Interrupt")
        exit(-1)
    except Exception as e:
        print(e)
        exit(1)
