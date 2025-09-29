# src/data_loader.py

import wrds
import os
import pandas as pd
from pathlib import Path


# This function needs the db connection too!
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def give_secid_data(ticker, db_connection): # <-- Add db_connection parameter
    secid_query = f"""SELECT secid FROM optionm.indexd WHERE ticker = '{ticker}'"""
    secid_data = db_connection.raw_sql(secid_query) # <-- Use the passed-in connection
    return secid_data

# Make db_connection a required argument by removing the default value
def give_raw_data(ticker, db_connection, end_date = None, start_date = None) -> pd.DataFrame:
    if start_date and end_date and start_date>=end_date:
        raise ValueError("End Date must be greater than Start Date")
    base_query =  f"""
            SELECT
            t2.date,
            t2.close,
            t2.open,
            t2.high,
            t2.low
        FROM
            optionm.secprd AS t2
        JOIN
            optionm.indexd AS t1 ON t1.secid = t2.secid
        WHERE 
            t1.ticker = '{ticker}'
    """
    date_filter_sql = ""
    if start_date and end_date:
        date_filter_sql = f"AND t2.date BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter_sql = f"AND t2.date >= '{start_date}'"
    elif end_date:
        date_filter_sql = f"AND t2.date <= '{end_date}'"

    final_query = f"{base_query} {date_filter_sql} ORDER BY t2.date"
    
    # Use the connection object that was passed into the function
    data_df = db_connection.raw_sql(final_query)
    return data_df

def give_raw_vvix_data():
    raw_vvix_data = pd.read_csv('/Users/pustak/Downloads/VVIX_History.csv')
    return raw_vvix_data

vvix_daily_raw_data = give_raw_vvix_data()

def store_raw_data(data:pd.DataFrame, name:str):
    project_folder_path =  PROJECT_ROOT
    folder_path = os.path.join(project_folder_path, 'data/raw')
    file_name = name
    file_path = os.path.join(folder_path, file_name)
    data.to_csv(file_path, index = False)

def load_raw_data(file_name: str):
    if "." not in file_name:
        raise KeyError('No file name exists: please enter full file name with extension')
    if not isinstance(file_name, str):
        raise KeyError('File name must be a string')
    data_containing_file_path = os.path.join(PROJECT_ROOT, "data", "raw", file_name)
    raw_data = pd.read_csv(data_containing_file_path)
    return raw_data
