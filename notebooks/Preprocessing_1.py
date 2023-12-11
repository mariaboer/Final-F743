# Bring in the correct packages
import argparse
import logging
import os
import pickle
import re

import numpy as np
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


def convert_duration(duration_str):
    # Pattern for 'PT#H#M' and 'P#DT#H#M'
    pattern = re.compile(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?')

    match = pattern.match(duration_str)
    if match:
        days = int(match.group(1) or 0)
        hours = int(match.group(2) or 0)
        minutes = int(match.group(3) or 0)
        total_minutes = days * 24 * 60 + hours * 60 + minutes
        return total_minutes
    else:
        return None


# Create split function for flights with more than one stop - not used as it created too many columns
def split_columns(row, column_name):
    values = row[column_name].split("||")
    return values + [0] * (4 - len(values))


# Create calculation for unique values in columns
def count_unique_values_in_row(row):
    values = str(row).split("||") if pd.notna(row) else ['']
    unique_values = set(filter(lambda x: x != 0, values))
    return len(unique_values)


# Create calculation for time of day
def convert_time_period(datetime_str):
    values = datetime_str.split('T')[1].split('.')[0]
    values = pd.to_datetime(values, format='%H:%M:%S').hour * 60 + pd.to_datetime(values, format='%H:%M:%S').minute
    return np.where(values <= (8 * 60), "Morning", np.where(values <= (16 * 60), "Mid-Day", "Night"))


# Create calculation for week number
def convert_week_number(datetime_str):
    dt_object = pd.to_datetime(datetime_str, format='%Y-%m-%dT%H:%M:%S.%f%z')
    iso_week = dt_object.isocalendar().week
    return iso_week


def main(rootpath, loader):
    # Bring in data to pre-process
    if loader == 'DataFile':
        datafile = os.path.join(rootpath, 'data', 'Atlanta Prices.csv')
        df = pd.read_csv(datafile)
    else:
        pickle_file = os.path.join(rootpath, 'snapshots', 'chunk.pkl')
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)

    # Preprocess Data
    logging.debug("Starting preprocessing")
    df = df.dropna()
    # Drop Unwanted columns in  Dataframe
    df = df[['destinationAirport', 'elapsedDays', 'isBasicEconomy', 'isNonStop', 'baseFare', 'totalFare', 'seatsRemaining',
             'totalTravelDistance', 'totalTravelDuration', 'weeknum', 'time_of_departure',
             'time_of_arrival', 'no_airlines', 'no_layovers', 'no_equipment', 'no_cabin_changes']]

    # Rename columns
    df['totalTravelDuration'] = df['travelDuration'].apply(lambda x: convert_duration(x))
    df['weeknum'] = df['segmentsDepartureTimeRaw'].str[:29].apply(lambda x: convert_week_number(x))
    df['time_of_departure'] = df['segmentsDepartureTimeRaw'].str[:29].apply(lambda x: convert_time_period(x))
    df['time_of_arrival'] = df['segmentsArrivalTimeRaw'].str[-29:].apply(lambda x: convert_time_period(x))
    df['no_layovers'] = df['segmentsArrivalAirportCode'].apply(count_unique_values_in_row)
    df['no_airlines'] = df['segmentsAirlineCode'].apply(count_unique_values_in_row)
    df['no_equipment'] = df['segmentsEquipmentDescription'].apply(count_unique_values_in_row)
    df['no_cabin_changes'] = df['segmentsCabinCode'].apply(count_unique_values_in_row)

    logging.debug("Replacing null values")
    # Check for sparse data - will impact models
    df.replace('', np.nan, inplace=True, regex=True)
    df.replace(' ', np.nan, inplace=True, regex=True)
    df.replace('null', np.nan, inplace=True, regex=True)
    df.replace('NULL', np.nan, inplace=True, regex=True)
    df.replace(None, np.nan, inplace=True, regex=True)
    df = df.dropna()
    logging.debug(f"Pre-Sparse drop:{df.head()}")
    sparse_columns = df.select_dtypes(include=pd.SparseDtype()).columns

    if 'baseFare' in sparse_columns:
        sparse_columns = sparse_columns[sparse_columns != 'baseFare']

    if not sparse_columns.empty:
        logging.warning(f"The following columns have sparse dtype: {sparse_columns}")

    df = df.drop(columns=sparse_columns)
    df = df.reset_index()
    logging.debug(f"Post sparse drop:\n{df.head()}")

    # Save dataframe off in case of error
    datafile = os.path.join(rootpath, 'data', 'AtlantaPrices_Processed.csv')
    df.to_csv(datafile, index=False)

    # Use Pickle to create a binary hierarchy that can be translated later
    with open(os.path.join(root_path, 'snapshots', 'preprocessed.pkl'), 'wb') as file:
        pickle.dump(df, file)
    logging.info("Preprocessing Complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--root_path', type=str_or_none, help='Root path of the project. Should have the snapshots/data folder', default=None)
    parser.add_argument('--loader', type=str_or_none, help='Type of loader to use. Literals "DataFile" or "Memory".Default - Memory', default="Memory")
    args = parser.parse_args()
    if args.root_path is None:
        args.root_path = os.path.dirname(__file__)
    configure_logging(logging.DEBUG, os.path.join(args.root_path,'logs'))
    if ['DataFile', 'Memory'] not in args.loader:
        logging.warning("Invalid loader. Valid loaders are 'DataFile' or 'Memory'. Defaulting to 'Memory'")
    args.loader = 'Memory'
    try:
        main(args.root_path, args.loader)
    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt")
        exit(-1)
    except Exception as e:
        logging.error(e)
        exit(1)
