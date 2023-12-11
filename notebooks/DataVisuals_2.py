import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    return value if value is None else str(x)


def main(rootpath, loader):
    # Bring in data to pre-process
    if loader == 'DataFile':
        datafile = os.path.join(rootpath, 'data', 'AtlantaPrices_Processed.csv')
        df = pd.read_csv(datafile)
    else:
        pickle_file = os.path.join(rootpath, 'snapshots', 'preprocessed.pkl')
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)

    # Debug Visuals
    logging.debug(df.info())
    logging.debug(df.head())

    # Set the style for seaborn
    sns.set(style="whitegrid")

    logging.debug("Creating Historgram and Box Plot for baseFare")
    # Histogram and Box Plot for baseFare
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['baseFare'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of baseFare')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='baseFare', data=df, color='lightcoral')
    plt.title('Box Plot of baseFare')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_Histogram.jpg')
    plt.savefig(file_name, format='jpeg')

    # Correlation Matrix
    logging.debug("Creating Correlation Matrix")
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.select_dtypes(include='number').corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_correlation.jpg')
    plt.savefig(file_name, format='jpeg')

    # Scatter Plots
    logging.debug("Creating Scatter Plots")
    scatter_features = ['totalTravelDuration', 'seatsRemaining']
    scatter_labels = ['Travel Duration vs baseFare', 'Seats Remaining vs baseFare']

    plt.figure(figsize=(16, 5))
    for i in range(len(scatter_features)):
        plt.subplot(1, len(scatter_features), i + 1)
        sns.scatterplot(x=scatter_features[i], y='baseFare', data=df, color='darkorange', alpha=0.5)
        plt.title(scatter_labels[i])

    file_name = os.path.join(rootpath, 'outputs', 'Flight_Scatter.jpg')
    plt.savefig(file_name, format='jpeg')

    # Categorical Plots
    logging.debug("Creating Categorical Plots")
    categorical_features = ['isBasicEconomy', 'isNonStop', 'time_of_departure', 'time_of_arrival']

    plt.figure(figsize=(16, 5))
    for i in range(len(categorical_features)):
        plt.subplot(1, len(categorical_features), i + 1)
        sns.boxplot(x=categorical_features[i], y='baseFare', data=df, palette='Set2')
        plt.title(f'{categorical_features[i]} vs baseFare')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_BoxPlot.jpg')
    plt.savefig(file_name, format='jpeg')

    # Time Series Plot
    logging.debug("Creating Time Series Plot")
    time_series_df = df.set_index('weeknum')['baseFare']

    plt.figure(figsize=(14, 6))
    sns.lineplot(x=time_series_df.index, y=time_series_df.values, color='mediumseagreen')
    plt.title('Time Series Plot of baseFare')
    plt.xlabel('Week Number')
    plt.ylabel('Base Fare')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_TimeSeries.jpg')
    plt.savefig(file_name, format='jpeg')

    logging.info('Data Visuals Completed')


if __name__ == '__main__':
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
