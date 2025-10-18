import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import percentileofscore
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


def calculate_realized_volatility(close_prices = pd.Series, window:int = 21) -> pd.Series:
    """Calculate the realized volatility over a specified rolling window."""
    TRADING_DAYS_PER_YEAR = 252
    log_returns = np.log(close_prices / close_prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    realized_volatility = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    realized_volatility.name = 'realized_volatility'
    realized_volatility.index.name = 'date'
    return realized_volatility


def calculate_momentum(data_series: pd.Series, window: int) -> pd.Series:
    momentum = data_series.pct_change(periods=window)
    return momentum

def calculate_rolling_percentile(data_series: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calculate the rolling percentile of the latest value within a specified window."""
    min_period_for_cal = max(3, int(window * 0.1))
    rolling_percentile = pd.DataFrame(index=data_series.index)
    rolling_percentile['value'] = data_series
    rolling_percentile.index.name = 'date'
    rolling_percentile[f'rolling_{window}d_percentile'] = data_series.rolling(window=window, min_periods=min_period_for_cal).apply(
        lambda x: percentileofscore(x, x.iloc[-1])/100
    )
    return rolling_percentile

def calculate_spx_drawdown_labels(window:int, drawdown_pct:float, price_series):
    """Generate drawdown labels for the S&P 500 index, to be used as a label for supervised logistic regression."""
    future_min = price_series.rolling(window = window).min().shift(periods = -window)
    threshold = price_series * (1 - drawdown_pct)
    drawdown_label = (future_min <= threshold).astype(float)
    drawdown_label[future_min.isna()] = np.nan
    drawdown_label.name = f'spx_drawdown_{int(drawdown_pct*100)}pct_{window}d'
    return drawdown_label

if __name__ == "__main__":
    pd.set_option('display.min_rows', 600)
    final_dataframe = pd.read_csv("data/processed_1/final_labeled_dataset.csv", parse_dates=['Date'], index_col='Date')
    vix_percentile_1y = calculate_rolling_percentile(final_dataframe['vix_level'], window = 252)
    spx_rv_percentile_1y = calculate_rolling_percentile(final_dataframe['spx_realized_vol_21d'], window = 252)
    vvix_percentile_1y = calculate_rolling_percentile(final_dataframe['vvix_level'], window = 252)
    credit_spread_percentile_1y = calculate_rolling_percentile(final_dataframe['credit_spread_high_yield'], window = 252)
    yield_curve_slope_percentile_1y = calculate_rolling_percentile(final_dataframe['yield_curve_slope_10y_2y'], window = 252)
    print(vix_percentile_1y)