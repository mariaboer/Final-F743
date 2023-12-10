# import time
import pandas as pd
import os

def process_chunk(df):
    df = df.query('startingAirport == "ATL"')
    print(df.shape)
    return df

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



# Save dataframe off in case of error
rootpath = os.path.dirname(__file__)
datafile = os.path.join(rootpath, 'Atlanta Prices.csv')
df_concat.to_csv(datafile, index=False)

##Use Pickle to create a binary hierarchy that can be translated later
with open('chunk.pkl', 'wb') as file:
    pickle.dump(df_concat, file)



