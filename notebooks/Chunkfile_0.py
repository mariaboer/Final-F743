import argparse
import logging
import os
import pickle

import pandas as pd


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


def process_chunk(df):
    df = df[df['startingAirport'] == "ATL"]
    logging.debug(f"Chunk processed. Shape: {df.shape}")
    logging.debug(df.info())
    return df


def main(rootpath):
    datafile = os.path.join(rootpath, 'data', 'itineraries.csv')

    chunks = pd.read_csv(datafile, chunksize=100000, engine='c')
    datafile = os.path.join(rootpath, 'data', 'Atlanta Prices.csv')
    for chunk in chunks:  # each chunk is a dataframe
        # perform data filtering
        filtered_chunk = process_chunk(chunk)
        filtered_chunk.to_csv(datafile, mode='a', index=False, header=False)
    df_concat = pd.read_csv(datafile)
    df_concat.reset_index(inplace=True)
    df_concat.columns = chunks.columns
    df_concat.to_csv(datafile, index=False)
    # Save dataframe off in case of error

    # Use Pickle to create a binary hierarchy that can be translated later
    if not os.path.exists(os.path.join(rootpath, 'snapshots')):
        os.mkdir(os.path.join(rootpath, 'snapshots'))

    with open(os.path.join(rootpath, 'snapshots', 'chunk.pkl'), 'wb') as file:
        pickle.dump(df_concat, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chunking Processor')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    args = parser.parse_args()
    if args.root_path is None:
        args.root_path = os.path.realpath(__file__)
    else:
        if not os.path.exists(args.root_path):
            args.root_path = os.path.realpath(__file__)
    configure_logging(logging.DEBUG, os.path.join(args.root_path, 'logs'))
    try:
        main(args.root_path)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        exit(-1)
    except Exception as e:
        print(e)
        exit(1)
