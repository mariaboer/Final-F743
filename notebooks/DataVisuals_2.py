import argparse
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns


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

    # Debug Visuals
    # df.info()
    # df.head()

    # Set the style for seaborn
    sns.set(style="whitegrid")

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

    # plt.show()

    # Correlation Matrix
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.select_dtypes(include='number').corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_correlation.jpg')
    plt.savefig(file_name, format='jpeg')

    # plt.show()

    # Scatter Plots
    scatter_features = ['totalTravelDuration', 'totalTravelDistance', 'seatsRemaining']
    scatter_labels = ['Travel Duration vs baseFare', 'Total Travel Distance vs baseFare', 'Seats Remaining vs baseFare']

    plt.figure(figsize=(16, 5))
    for i in range(len(scatter_features)):
        plt.subplot(1, len(scatter_features), i + 1)
        sns.scatterplot(x=scatter_features[i], y='baseFare', data=df, color='darkorange', alpha=0.5)
        plt.title(scatter_labels[i])

    file_name = os.path.join(rootpath, 'outputs', 'Flight_Scatter.jpg')
    plt.savefig(file_name, format='jpeg')

    # plt.show()

    # Categorical Plots
    categorical_features = ['isBasicEconomy', 'isNonStop', 'time_of_departure', 'time_of_arrival']

    plt.figure(figsize=(16, 5))
    for i in range(len(categorical_features)):
        plt.subplot(1, len(categorical_features), i + 1)
        sns.boxplot(x=categorical_features[i], y='baseFare', data=df, palette='Set2')
        plt.title(f'{categorical_features[i]} vs baseFare')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_BoxPlot.jpg')
    plt.savefig(file_name, format='jpeg')

    # plt.show()

    # Time Series Plot
    time_series_df = df.set_index('weeknum')['baseFare']

    plt.figure(figsize=(14, 6))
    sns.lineplot(x=time_series_df.index, y=time_series_df.values, color='mediumseagreen')
    plt.title('Time Series Plot of baseFare')
    plt.xlabel('Week Number')
    plt.ylabel('Base Fare')

    file_name = os.path.join(rootpath, 'outputs', 'Flight_TimeSeries.jpg')
    plt.savefig(file_name, format='jpeg')

    print('Data Visuals Completed')

    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Visuals')
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
