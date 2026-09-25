#%%

import pandas as pd

# Now read the 2019 parquet file to do data analysis
TRIPS_COLUMNS = [
    "tripduration",
    "starttime",
    "stoptime",
    "startstationid",
    "endstationid",
    "bikeid",
    "usertype",
    "birthyear",
    "gender",
]

trips = pd.read_parquet('data/pqt/2019/201901-citibike-tripdata.parquet', columns=TRIPS_COLUMNS,
                 engine='pyarrow').reset_index()

trips.drop(trips.columns[0], axis=1, inplace=True)  # drop the dask index
trips["starttime"] = trips["starttime"].astype("datetime64")
trips["stoptime"] = trips["stoptime"].astype("datetime64")

display(trips.head(5))

# %%

hourly_trips = (trips.groupby([trips['starttime'].dt.date,
                              trips['starttime'].dt.hour])['tripduration']
                .count())

display(hourly_trips)

#%%

hourly_trips = hourly_trips.reset_index()

# %%

#hourly_trips.index.names(['startday','starthour'])

hourly_trips.index.names = [None,None]

# %%
print(hourly_trips.index.names)
#display(hourly_trips)
# %%

hourly_trips = hourly_trips.reset_index()
# %%

display(hourly_trips)
# %%
