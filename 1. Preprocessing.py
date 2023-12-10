# Bring in the correct packages
import pandas as pd
import numpy as np
import re
import pickle
import os


# The Atlanta Prices is made up of concatenated columns up to 4 different values that consists of date/datetime,
# strings, and integers. Def will be created to process those columns

# Create custom equation for working with date and datetime items
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


# Create split function for flights with more than one stop
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


# Bring in data to pre-process
rootpath = os.path.dirname(__file__)
datafile = os.path.join(rootpath, 'Atlanta Prices.csv')
df = pd.read_csv(datafile)

# picklefile = os.path.join(rootpath, 'chunk.pkl')
# with open(picklefile, 'rb') as file:
#    df = pickle.load(file)

# Preprocess Data

df = df.dropna()

df['totalTravelDuration'] = df['travelDuration'].apply(lambda x: convert_duration(x))

df['weeknum'] = df['segmentsDepartureTimeRaw'].str[:29].apply(lambda x: convert_week_number(x))

df['time_of_departure'] = df['segmentsDepartureTimeRaw'].str[:29].apply(lambda x: convert_time_period(x))

df['time_of_arrival'] = df['segmentsArrivalTimeRaw'].str[-29:].apply(lambda x: convert_time_period(x))

df['no_layovers'] = df['segmentsArrivalAirportCode'].apply(count_unique_values_in_row)

df['no_airlines'] = df['segmentsAirlineCode'].apply(count_unique_values_in_row)

df['no_equipment'] = df['segmentsEquipmentDescription'].apply(count_unique_values_in_row)

df['no_cabin_changes'] = df['segmentsCabinCode'].apply(count_unique_values_in_row)

# Update Dataframe
df = df[
    ['destinationAirport', 'fareBasisCode', 'travelDuration', 'elapsedDays', 'isBasicEconomy', 'isNonStop', 'baseFare',
    'totalFare', 'seatsRemaining', 'totalTravelDistance', 'totalTravelDuration', 'weeknum', 'time_of_departure',
    'time_of_arrival', 'no_airlines', 'no_layovers', 'no_equipment', 'no_cabin_changes']
]

# Check for sparse data - will impact models
df = df.dropna()
sparse_columns = df.select_dtypes(include=pd.SparseDtype()).columns

if not sparse_columns.empty:
    print(f"The following columns have sparse dtype: {sparse_columns}")

if 'baseFare' in sparse_columns:
    sparse_columns = sparse_columns[sparse_columns != 'baseFare']

df = df.drop(columns=sparse_columns)

# Save dataframe off in case of error
datafile = os.path.join(rootpath, 'AtlantaPrices_Processed.csv')
df.to_csv(datafile, index=False)

##Use Pickle to create a binary hierarchy that can be translated later
with open('preprocessed.pkl', 'wb') as file:
    pickle.dump(df, file)