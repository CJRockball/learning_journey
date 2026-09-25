import requests, zipfile, io, os, tarfile
import pyarrow as pa
import pandas as pd
import dask.dataframe as dd
from urllib.request import urlopen
import pathlib
import os
import numpy as np

PARQUET_EXTENSION = '.parquet'
ROOT = pathlib.Path(__file__).resolve().parent

def dl_fcn():
    """
    Download citi bike data save as csv on disk
    """
    year_list = [2014, 2015, 2016, 2017, 2018, 2019]
    for year_ in year_list:
        print(f'Processing year {year_}')
        CSV_YEAR_FOLDER = ROOT / 'data/bike/csv' / str(year_)
        
        if not os.path.exists(CSV_YEAR_FOLDER):
            CSV_YEAR_FOLDER.mkdir(parents=True, exist_ok=True)
            for i in range (1,13):

                CSV_FNAME = str(year_) + str(i).zfill(2) + '-citibike-tripdata.csv'
                if year_ < 2017:
                    url = 'https://s3.amazonaws.com/tripdata/' + str(year_) + str(i).zfill(2) + '-citibike-tripdata.zip'
                else:
                    url = 'https://s3.amazonaws.com/tripdata/' + str(year_) + str(i).zfill(2) + '-citibike-tripdata.csv.zip'

                print('getting file', url)
                r = requests.get(url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                
                for zinfo in z.infolist():
                    zname = zinfo.filename
                
                z.extractall(CSV_YEAR_FOLDER)
                
                if zname != CSV_FNAME:
                    old_file = os.path.join(str(CSV_YEAR_FOLDER), zname)
                    new_file = os.path.join(str(CSV_YEAR_FOLDER), CSV_FNAME)
                    os.rename(old_file, new_file)
                                    
                print(f'{year_} month {i} extracted') 
                
        else: print(f'Year {year_} already downloaded')
    
    return


def min_size():
    """ Minimize df size by modding dtypes.
        Save to parquet"""
    year_list = [2016, 2017, 2018, 2019] #[2014, 2015, 2016, 2017, 2018, 2019]
    
    for year_ in year_list:
        
        print(f'Processing year {year_}')
        CSV_YEAR_FOLDER = ROOT / 'data/bike/csv' / str(year_)

        def col_type_mod(df_mod):
            for col in int_list:
                df_mod[col] = pd.to_numeric(df_mod[col], downcast='integer')

            for col in float_list:
                df_mod[col] = pd.to_numeric(df_mod[col], downcast='float')
                
            for col in date_cols:
                df_mod[col] = pd.to_datetime(df_mod[col])

            for col in cat_cols:
                df_mod[col] = df_mod[col].astype('category')
            
            return df_mod
        
        org_size = []
        for i in range (1,13):
            print(f'processing month {i}')
            col_names = ['tripduration', 'starttime', 'stoptime', 'start station id',
             'start station name', 'start station latitude','start station longitude',
             'end station id', 'end station name', 'end station longitude',
             'end station latitude', 'bikeid','usertype','birth year','gender']
            CSV_FNAME = str(year_) + str(i).zfill(2) + '-citibike-tripdata.csv'
            df = pd.read_csv(CSV_YEAR_FOLDER / CSV_FNAME, names=col_names, dtype=str)
            
            org_size.append(df.memory_usage(deep=True).sum()/10**9)  

            #int_cols = df.select_dtypes(include=['int']).columns.to_list()
            #float_cols = df.select_dtypes(include=['float']).columns.to_list()
            float_list = ['start station latitude','start station longitude','end station latitude','end station longitude']
            int_list = ['tripduration', 'start station id', 'end station id','bikeid', 'birth year', 'gender']
            date_cols = ['starttime', 'stoptime']
            cat_cols = ['usertype', 'start station name', 'end station name']
    
            if df.isna().sum().sum() != 0:
                df = df.dropna()
      
            df = df[df.iloc[:,13].apply(lambda x: str(x).isnumeric())]
            print(df.shape)    
            
            df_mod = col_type_mod(df)

            if i == 1:
                df_full = df_mod.copy()
            else:
                df_full = pd.concat([df_full,df_mod], axis=0, ignore_index=True)   
                print(f'df_comp size: {df.memory_usage(deep=True).sum()/10**9}')
        new_size = df_full.memory_usage(deep=True).sum()/10**9       
        
        
        for col in cat_cols:
            df_full[col] = df_full[col].astype('category')
        new_size2 = df_full.memory_usage(deep=True).sum()/10**9       
        print(f'original size: {sum(org_size)}, new size {new_size}, new size2 {new_size2}')
                    
        df_full.to_parquet(os.path.join(CSV_YEAR_FOLDER, str(year_) + '_full.parquet'))

    print('Done')            


def merge_yearly_trips():
    """
    Inter year parquet file merge
    """
    year_list = [2014, 2015, 2016, 2017, 2018, 2019]
    #month_files = sorted([directory + f for f in os.listdir(directory) if f.endswith(PARQUET_EXTENSION)])
    
    parquet_ddfs: list[dd.DataFrame] = []
    for year_ in year_list:
        CSV_YEAR_FOLDER = ROOT / 'data/bike/csv' / str(year_)
        CSV_FNAME = str(year_) + '_full.parquet'
        ddf = dd.read_parquet(CSV_YEAR_FOLDER / CSV_FNAME)
        print(ddf.memory_usage(deep=True).sum()/10**9, 'GB')
        parquet_ddfs.append(ddf)

    all_trips = dd.concat(parquet_ddfs)
    print(all_trips.memory_usage(deep=True).sum()/10**9, 'GB')
    fname = 'all_years.parquet'
    all_trips.to_parquet(
        'data/bike/' + fname,
        engine="pyarrow"
    )


def convert_to_parquet(directory_from: str, directory_to: str):
    """
    _summary_
    Convert csv in directory from to parquet in directory to
    Args:
        directory_from (str): Directory that holds csv
        directory_to (str): Directory to save parquet
    """
    csv_files = sorted([directory_from + f for f in os.listdir(directory_from)])
    i = 0
    for csv_file in csv_files:
        i += 1
        filename = csv_file.split('/')[-1][:-4]
        newfilepath = directory_to + filename + ".parquet"
        
        print('Open file')
        csv_df = pd.read_csv(csv_file)
        csv_df.columns = [col.lower().replace(' ', '') for col in csv_df.columns]
        csv_df.dropna(inplace=True)
        csv_df['birthyear'] = csv_df['birthyear'].replace(r"\\N","0", regex=True)
        print(f'Save file {filename}')
        csv_df.to_parquet(newfilepath)



# schema for parquet files in
TRIPDATA_COLUMN_DTYPES = {
    "tripduration": "int32",
    "starttime": "datetime64",
    "stoptime": "datetime64",
    "startstationid": "category",
    "startstationname": "category",
    "startstationlatitude": "category",
    "startstationlongitude": "category",
    "endstationid": "category",
    "endstationname": "category",
    "endstationlatitude": "category",
    "endstationlongitude": "category",
    "bikeid": "category",
    "usertype": "category",
    "birthyear": "category",
    "gender": "category",
}

def merge_monthly_trips(year, directory:str):
    """
    Inter year parquet file merge

    Args:
        year (int): Which year to merge
        directory (str): Which directory to merge in
    """
    month_files = sorted([directory + f for f in os.listdir(directory) if f.endswith(PARQUET_EXTENSION)])
    print(month_files)

    parquet_ddfs: list[dd.DataFrame] = []
    for month_file in month_files:
        if os.path.exists(month_file):
            ddf = dd.read_parquet(month_file)
            ddf.astype(TRIPDATA_COLUMN_DTYPES)
            parquet_ddfs.append(ddf)
    
    all_trips = dd.concat(parquet_ddfs)
    filename = str(year) if year else 'alltrips'
    all_trips.to_parquet(
        directory + filename + PARQUET_EXTENSION,
        engine="pyarrow"
    )




def dl_weather(year='2019'):
    fname = year + '.tar.gz'
    url = "https://www.ncei.noaa.gov/data/global-hourly/archive/csv/" + fname
    
    print('getting file', url)
    r = requests.get(url, stream=True)
    tar = tarfile.open(name=None, fileobj=io.BytesIO(r.content)) #, "r:gz")
    print('extracting zip file')
    tar.extractall('data/weather/csv/' + year)  
    tar.close()
    return   
    

if __name__ == '__main__':
    #dl_fcn()
    #print('download done')
    #min_size()
    merge_yearly_trips()
    
    # dir_from = 'data/bike/csv/2018/'
    # dir_to = 'data/bike/pqt/2018/'
    # convert_to_parquet(dir_from, dir_to)

    # merge_monthly_trips(2018, 'data/bike/pqt/2018/')

    #dl_weather()
