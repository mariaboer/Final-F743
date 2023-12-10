import argparse
import os
import pickle

import pandas as pd


def str_or_none(value):
    return value if value is None else str(x)


def process_chunk(df):
    df = df.query('startingAirport == "ATL"')
    print(df.shape)
    return df


def main(rootpath):
    if rootpath is None:
        rootpath = os.path.dirname(__file__)
    datafile = os.path.join(rootpath, 'itineraries.csv')

    chunks = pd.read_csv(datafile, chunksize=100000)

    chunk_list = []  # used for storing dataframes
    for chunk in chunks:  # each chunk is a dataframe
        # perform data filtering
        filtered_chunk = process_chunk(chunk)

        # Once the data filtering is done, append the filtered chunk to list
        chunk_list.append(filtered_chunk)
    df_concat = pd.concat(chunk_list)
    df_concat.reset_index(inplace=True)
    # Save dataframe off in case of error
    datafile = os.path.join(rootpath, 'Atlanta Prices.csv')
    df_concat.to_csv(datafile, index=False)

    ##Use Pickle to create a binary hierarchy that can be translated later
    with open(os.path.join(rootpath, 'snapshots', 'chunk.pkl'), 'wb') as file:
        pickle.dump(df_concat, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Regression')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    args = parser.parse_args()
    try:
        main(args.root_path)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        exit(-1)
    except Exception as e:
        print(e)
        exit(1)
