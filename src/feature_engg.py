import argparse
import numpy as np
import dask, dask_cudf
from math import cos, sin, asin, sqrt, pi

from dask_client import dask_client
from load_data import load_data

def jfk_distance(dropoff_latitude, dropoff_longitude, jfk_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_jfk = pi/180 * 40.6413
        y_jfk = pi/180 * -73.7781
        
        dlon = y_jfk - y_1
        dlat = x_jfk - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_jfk) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        jfk_distance[i] = c * r
        
def lga_distance(dropoff_latitude, dropoff_longitude, lga_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_lga = pi/180 * 40.7769
        y_lga = pi/180 * -73.8740
        
        dlon = y_lga - y_1
        dlat = x_lga - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_lga) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        lga_distance[i] = c * r
        
def ewr_distance(dropoff_latitude, dropoff_longitude, ewr_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_ewr = pi/180 * 40.6895
        y_ewr = pi/180 * -74.1745
        
        dlon = y_ewr - y_1
        dlat = x_ewr - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_ewr) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        ewr_distance[i] = c * r
        
def tsq_distance(dropoff_latitude, dropoff_longitude, tsq_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_tsq = pi/180 * 40.7580
        y_tsq = pi/180 * -73.9855
        
        dlon = y_tsq - y_1
        dlat = x_tsq - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_tsq) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        tsq_distance[i] = c * r
        
def met_distance(dropoff_latitude, dropoff_longitude, met_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_met = pi/180 * 40.7794
        y_met = pi/180 * -73.9632
        
        dlon = y_met - y_1
        dlat = x_met - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_met) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        met_distance[i] = c * r
        
def wtc_distance(dropoff_latitude, dropoff_longitude, wtc_distance):
    for i, (x_1, y_1) in enumerate(zip(dropoff_latitude, dropoff_longitude)):
        x_1 = pi/180 * x_1
        y_1 = pi/180 * y_1
        x_wtc = pi/180 * 40.7126
        y_wtc = pi/180 * -74.0099
        
        dlon = y_wtc - y_1
        dlat = x_wtc - x_1
        a = sin(dlat/2)**2 + cos(x_1) * cos(x_wtc) * sin(dlon/2)**2
        
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        
        wtc_distance[i] = c * r

def dtype_conversion(df):
    df['key'] = df['key'].astype('datetime64[ns]')
    df['fare_amount'] = df ['fare_amount'].astype('float32')
    df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
    df['pickup_longitude'] = df ['pickup_longitude'].astype('float32')
    df['pickup_latitude'] = df ['pickup_latitude'].astype('float32')
    df['dropoff_longitude'] = df ['dropoff_longitude'].astype('float32')
    df['dropoff_latitude'] = df ['dropoff_latitude'].astype('float32')
    df['passenger_count'] = df ['passenger_count'].astype('uint8')
    return df

def filter_outliers_missing_vals(df):
    query_frags = [
    'fare_amount >= 2.5 and fare_amount < 500',
    'passenger_count > 0 and passenger_count < 6',
    'pickup_longitude > -75 and pickup_longitude < -73',
    'dropoff_longitude > -75 and dropoff_longitude < -73',
    'pickup_latitude > 40 and pickup_latitude < 42',
    'dropoff_latitude > 40 and dropoff_latitude < 42'
    ]
    df = df.query(' and '.join(query_frags))
    return df

def feature_addition(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['weekday'] = df['pickup_datetime'].dt.weekday
    
    df = df.apply_rows(jfk_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(jfk_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(lga_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(lga_distance=np.float32), kwargs=dict())
        
    df = df.apply_rows(ewr_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(ewr_distance=np.float32), kwargs=dict())
            
    df = df.apply_rows(tsq_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(tsq_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(met_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(met_distance=np.float32), kwargs=dict())
    
    df = df.apply_rows(wtc_distance, incols=['dropoff_latitude', 'dropoff_longitude'],
                       outcols=dict(wtc_distance=np.float32), kwargs=dict())
    
    df = df.drop(['pickup_datetime','key'], axis=1)
    return df

def feature_engg(df):
    df = dtype_conversion(df)
    df = filter_outliers_missing_vals(df)
    parts = [dask.delayed(feature_addition)(part) for part in df.to_delayed()]
    df = dask_cudf.from_delayed(parts)
    return df
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    client = dask_client(config_path=parsed_args.config)
    df = load_data(config_path=parsed_args.config)
    _ = feature_engg(df)
    client.close()