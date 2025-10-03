import requests
import re
import json
import pandas as pd
from pathlib import Path

from .vix_extract_web_scrapper import read_csv_with_auto_skip

def refine_vix_raw_data(raw_folder, refined_folder):
    """
    Processes raw VIX futures data: adds an 'Expiration' column based on the last
    trade date and saves the cleaned file to a new directory.
    """
    raw_path = Path(raw_folder)
    refined_path = Path(refined_folder)
    refined_path.mkdir(parents=True, exist_ok=True)
    
    raw_files = list(raw_path.glob('*.csv'))
    if not raw_files:
        print(f"No raw CSV files found in '{raw_path}'.")
        return

    print(f"\nStarting refining process for {len(raw_files)} files...")
    
    for file in raw_files:
        try:
            df = read_csv_with_auto_skip(file)
            df['Trade Date'] = pd.to_datetime(df['Trade Date'])
            expiration_date = df['Trade Date'].iloc[-1]
            df['Expiration Date'] = expiration_date
            
            save_path = refined_path / file.name
            df.to_csv(save_path, index=False)
            
            print(f"  Refined and saved {file.name}")
        except Exception as e:
            print(f"  Failed to process {file.name}. Error: {e}")
            
    print("\nData refining process finished.")

def create_final_dataframe(refined_futures_folder, spot_vix_filepath):
    """
    Builds the complete, analysis-ready DataFrame with M1/M2 prices, expirations, DTE,
    and merges it with spot VIX data to calculate final features.
    """
    #Process Refined Futures Data 
    refined_path = Path(refined_futures_folder)
    if not refined_path.exists():
        print(f"Error: Refined futures folder not found at '{refined_path}'"); return pd.DataFrame()

    all_files = list(refined_path.glob('*.csv'))
    print(f"Found {len(all_files)} refined CSV files to process...")

    all_contracts_df = [read_csv_with_auto_skip(file) for file in all_files]
    master_df = pd.concat(all_contracts_df, ignore_index=True)

    master_df['Trade Date'] = pd.to_datetime(master_df['Trade Date'])
    master_df['Expiration Date'] = pd.to_datetime(master_df['Expiration Date'])
    master_df = master_df.sort_values(by=['Trade Date', 'Expiration Date']).reset_index(drop=True)
    
    term_structure_records = []
    for trade_date, daily_group in master_df.groupby('Trade Date'):
        active_contracts = daily_group[daily_group['Expiration Date'] > trade_date]
        if len(active_contracts) >= 2:
            m1 = active_contracts.iloc[0]; m2 = active_contracts.iloc[1]
            m1_dte = (m1['Expiration Date'] - trade_date).days
            m2_dte = (m2['Expiration Date'] - trade_date).days
            
            term_structure_records.append({
                'Date': trade_date, 'M1_Price': m1['Settle'], 'M2_Price': m2['Settle'],
                'M1_Expiration': m1['Expiration Date'], 'M2_Expiration': m2['Expiration Date'],
                'M1_DTE': m1_dte, 'M2_DTE': m2_dte
            })

    term_structure_df = pd.DataFrame(term_structure_records).set_index('Date')
    print("Futures term structure successfully built.")

    #Load Spot VIX Data
    spot_path = Path(spot_vix_filepath)
    if not spot_path.exists():
        print(f"Error: Spot VIX file not found at '{spot_path}'"); return pd.DataFrame()
    
    spot_vix_df = pd.read_csv(spot_vix_filepath, usecols=['date', 'close'])
    spot_vix_df.rename(columns={'date': 'Date', 'close': 'VIX_Close'}, inplace=True)
    spot_vix_df['Date'] = pd.to_datetime(spot_vix_df['Date'])
    spot_vix_df.set_index('Date', inplace=True)
    print("Spot VIX data loaded.")

    #Merge, Clean, and Calculate Final Features ---
    final_df = term_structure_df.join(spot_vix_df)

    #Normalization Step
    cutoff_date = '2007-03-26'
    historical_mask = final_df.index < cutoff_date
    if historical_mask.any():
        print(f"Normalizing historical data: Dividing M1/M2 prices by 10 for dates before {cutoff_date}...")
        final_df.loc[historical_mask, ['M1_Price', 'M2_Price']] /= 10
    
    initial_rows = len(final_df)
    final_df = final_df[(final_df['M1_Price'] > 0) & (final_df['M2_Price'] > 0)].copy()
    rows_removed = initial_rows - len(final_df)
    if rows_removed > 0:
        print(f"Data Cleaning: Removed {rows_removed} rows with non-positive M1/M2 settlement prices.")
    
    final_df.dropna(inplace=True)

    final_df['Term_Structure_Ratio_1_0'] = final_df['M1_Price'] / final_df['VIX_Close']
    final_df['Term_Structure_Slope_2_1'] = final_df['M2_Price'] - final_df['M1_Price']
    


    print("Final DataFrame created with all features.")
    return final_df