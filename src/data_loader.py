import wrds
import os
import pandas as pd
print("Connecting to WRDS...")
db = wrds.Connection()
print("Connection Succesfull")

def give_secid_data(ticker):
    secid_query = f"""SELECT secid FROM optionm.indexd WHERE ticker = '{ticker}'"""
    secid_data = db.raw_sql(secid_query)
    return secid_data
#Why are we retrieving secid_data dataframe instead of secid directly?
# =>  Because there can be instances where manual checking is required.

def give_raw_data(ticker, db_connection = db, end_date = None, start_date = None) -> pd.DataFrame:
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
    # Dynamically add the date filter part
    date_filter_sql = ""
    if start_date and end_date:
        date_filter_sql = f"AND t2.date BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter_sql = f"AND t2.date >= '{start_date}'"
    elif end_date:
        date_filter_sql = f"AND t2.date <= '{end_date}'"

    final_query = f"{base_query} {date_filter_sql} ORDER BY t2.date"
    
    data_df = db.raw_sql(final_query)
    return data_df

def give_refined_data(data_df: pd.DataFrame) -> pd.DataFrame:
    refined_data = data_df.copy()
    refined_data.set_index(pd.to_datetime(refined_data['date']), inplace=True)
    refined_data.drop(columns=['date'], inplace=True)
    
    # Create full calendar range
    full_range = pd.date_range(refined_data.index.min(), refined_data.index.max())
    refined_data = refined_data.reindex(full_range)
    return refined_data


# VVIX historical data was sourced directly from the CBOE website:  
# https://www.cboe.com/tradable_products/vix/vix_historical_data/  
# The indexed table in Optionm only contained data after 2012-04-19.  
# Using the CBOE data, which extends back to mid-2006, helped avoid a major  
# bottleneck in training the logistic regression model.  

def give_raw_vvix_data():
    raw_vvix_data = pd.read_csv('/Users/pustak/Downloads/VVIX_History.csv')
    return raw_vvix_data

vvix_daily_raw = give_raw_vvix_data()

#vix_daily_raw = give_raw_data("VIX")
#spx_daily_raw = give_raw_data("SPX")

# I am commenting these two for the time-being.
# Because they make code execution slower, and thus slower testing.

#Note: Full data in the variable above means:
#  data from the inception of the index to the latest date in WRDS universe.

def store_raw_full_data(data:pd.DataFrame, name:str):
    project_folder_path = os.getcwd()
    folder_path = os.path.join(project_folder_path, 'data/raw')
    file_name = name
    file_path = os.path.join(folder_path, file_name)
    data.to_csv(file_path, index = False)
# I know I could have easily just copy pasted the data - but completing due-dilligence.

