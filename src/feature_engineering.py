import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Function to load relevant raw data
def load_raw_data(file_name: str):
    if "." not in file_name:
        raise KeyError('No file name exists: please enter full file name with extension')
    if not isinstance(file_name, str):
        raise KeyError('File name must be a string')
    data_containing_file_path = os.path.join(PROJECT_ROOT, "data", "raw", file_name)
    raw_data = pd.read_csv(data_containing_file_path)
    return raw_data

#Calculation of rolling daily_volatility and conversion into annualized_volatility
def calculate_realized_volatility(close_prices = pd.Series, window:int = 21) -> pd.Series:
    TRADING_DAYS_PER_YEAR = 252
    log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    realized_volatility = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    realized_volatility.name = 'realized_volatility'
    realized_volatility.index.name = 'date'
    return realized_volatility




