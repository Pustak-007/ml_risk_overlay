"""import pandas as pd
from pathlib import Path

def build_final_dataset(term_structure_path, raw_data_folder):
    '''
    Loads all relevant data sources, joins them by date, and creates the
    final, analysis-ready feature DataFrame.
    '''
    dfs_to_join = []



    # 1. VIX Change 5D
    vix_change_path = Path(raw_data_folder) / "vix_change_5d.csv"
    if vix_change_path.exists():
        print(f"Loading data from {vix_change_path} ...")
        vix_change_df = pd.read_csv(vix_change_path)
        vix_change_df.rename(columns={
            'date': 'Date',
            'VIX Change 5D': 'vix_change_5d'
        }, inplace=True)
        vix_change_df['Date'] = pd.to_datetime(vix_change_df['Date'])
        vix_change_df.set_index('Date', inplace=True)
        dfs_to_join.append(vix_change_df)


    #2. SPX Realized Volatility 21D
    spx_realized_volatility_21d_path = Path(raw_data_folder) / "spx_realized_vol_21d.csv"
    if spx_realized_volatility_21d_path.exists():
        print(f"Loading data from {spx_realized_volatility_21d_path} ...")
        spx_realized_volatility_21d_df = pd.read_csv(spx_realized_volatility_21d_path)
        spx_realized_volatility_21d_df.rename(columns = {'date':'Date', 'SPX Realized Volatility 21D':'spx_realized_volatility_21d'}, inplace = True)
        spx_realized_volatility_21d_df['Date'] = pd.to_datetime(spx_realized_volatility_21d_df['Date'])
        spx_realized_volatility_21d_df.set_index('Date', inplace=True)
        dfs_to_join.append(spx_realized_volatility_21d_df)
    
    #3. VVIX Level
    vvix_path = Path(raw_data_folder) / "vvix_daily_raw.csv"
    if vvix_path.exists():
        print(f"Loading data from {vvix_path} ...")
        vvix_df = pd.read_csv(vvix_path)
        vvix_df.rename(columns={
            'DATE': 'Date',
            'VVIX': 'vvix_level'
        }, inplace=True)
        vvix_df['Date'] = pd.to_datetime(vvix_df['Date'])
        vvix_df.set_index('Date', inplace=True)
        dfs_to_join.append(vvix_df)

    # 4. Credit Spread High Yield
    credit_spread_path = Path(raw_data_folder) / "credit_spread_high_yield.csv"
    if credit_spread_path.exists():
        print(f"Loading data from {credit_spread_path} ...")
        credit_spread_df = pd.read_csv(credit_spread_path)
        credit_spread_df.rename(columns={
            'date': 'Date',
            'BofA US High Yield Index Spread': 'credit_spread_high_yield'
        }, inplace=True)
        credit_spread_df['Date'] = pd.to_datetime(credit_spread_df['Date'])
        credit_spread_df.set_index('Date', inplace=True)
        dfs_to_join.append(credit_spread_df)

    # 5. Credit Spread Change 21D
    credit_spread_change_path = Path(raw_data_folder) / "credit_spread_high_yield_change_21d.csv"
    if credit_spread_change_path.exists():
        print(f"Loading data from {credit_spread_change_path} ...")
        credit_spread_change_df = pd.read_csv(credit_spread_change_path)
        credit_spread_change_df.rename(columns={
            'date': 'Date',
            'Credit Spread High Yield Change 21D': 'credit_spread_change_21d'
        }, inplace=True)
        credit_spread_change_df['Date'] = pd.to_datetime(credit_spread_change_df['Date'])
        credit_spread_change_df.set_index('Date', inplace=True)
        dfs_to_join.append(credit_spread_change_df)


    # 6. Yield Curve Slope 10Y/2Y
    yield_curve_slope_path = Path(raw_data_folder) / "yield_curve_slope_10y_2y.csv"
    if yield_curve_slope_path.exists():
        print(f"Loading data from {yield_curve_slope_path} ...")
        yield_curve_df = pd.read_csv(yield_curve_slope_path)
        yield_curve_df.rename(columns = {
            'date':'Date', 
            'Yield Curve Slope 10Y/2Y':'yield_curve_slope_10y_2y'
        }, inplace = True)
        yield_curve_df['Date'] = pd.to_datetime(yield_curve_df['Date'])
        yield_curve_df.set_index('Date', inplace=True)
        dfs_to_join.append(yield_curve_df)

    # 7. Spot VIX
    spot_vix_path = Path(raw_data_folder) / "vix_daily_raw.csv"
    if spot_vix_path.exists():
        print(f"Loading data from {spot_vix_path} ...")
        spot_vix_df = pd.read_csv(spot_vix_path)
        spot_vix_df = spot_vix_df[['date', 'close']]
        spot_vix_df.rename(columns = {
            'date':'Date', 
            'close':'vix_level'
        }, inplace = True)
        spot_vix_df['Date'] = pd.to_datetime(spot_vix_df['Date'])
        spot_vix_df.set_index('Date', inplace=True)
        #no appending for spot_vix as spot_vix_df is going to be the base_df

    #8.Term_Structure_Ratio_1_0 and 9.Term_Structure_Slope_2_1
    vix_term_structure_path = Path(term_structure_path)
    if vix_term_structure_path.exists():
        print(f"Loading data from {vix_term_structure_path} ...")
        term_structure_df = pd.read_csv(vix_term_structure_path)
        
        # Select all the columns you want from this file at once
        term_structure_df = term_structure_df[['Date', 'Term_Structure_Ratio_1_0', 'Term_Structure_Slope_2_1']]
        
        term_structure_df.rename(columns={
            'Term_Structure_Ratio_1_0': 'term_structure_ratio_1_0',
            'Term_Structure_Slope_2_1': 'term_structure_slope_2_1'
        }, inplace=True)
        term_structure_df['Date'] = pd.to_datetime(term_structure_df['Date'])
        term_structure_df.set_index('Date', inplace=True)
        dfs_to_join.append(term_structure_df)

    base_df = spot_vix_df
    final_df = base_df
    print('\nJoining all the data sources...')
    for feature_df in dfs_to_join:
        final_df = final_df.join(feature_df)

    # --- Final Cleanup ---
    # Remove any rows that have missing data after the join
    # This ensures every row in your dataset is complete.
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    rows_removed = initial_rows - len(final_df)
    if rows_removed > 0:
        print(f"Cleanup: Removed {rows_removed} rows with missing data after joining.")
        
    print("\nFinal dataset successfully assembled.")
    return final_df
  """

import pandas as pd
from pathlib import Path

def load_feature_df(file_path, use_cols, rename_dict):
    """A helper function to load, select, rename, and index a feature CSV."""
    if not file_path.exists():
        print(f"  - Warning: File not found, skipping: {file_path.name}")
        return None
    
    print(f"Loading data from {file_path.name} ...")
    df = pd.read_csv(file_path, usecols=use_cols)
    df.rename(columns=rename_dict, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def build_final_dataset(term_structure_path, raw_data_folder):
    """
    Loads all relevant data sources in a specific order, joins them by date, 
    and creates the final, analysis-ready feature DataFrame.
    """
    # --- 1. Load the Base DataFrame: Spot VIX ---
    base_df = load_feature_df(
        file_path=Path(raw_data_folder) / "vix_daily_raw.csv",
        use_cols=['date', 'close'],
        rename_dict={'date': 'Date', 'close': 'vix_level'}
    )
    if base_df is None:
        print("Error: Base VIX file is missing. Cannot proceed.")
        return pd.DataFrame()

    # --- 2. Define all other features to load and join in chronological order ---
    features_to_load = [
        {
            "filename": "vix_change_5d.csv",
            "use_cols": ['date', 'VIX Change 5D'],
            "rename_dict": {'date': 'Date', 'VIX Change 5D': 'vix_change_5d'}
        },
        {
            "filename": "spx_realized_vol_21d.csv",
            "use_cols": ['date', 'SPX Realized Volatility 21D'],
            "rename_dict": {'date': 'Date', 'SPX Realized Volatility 21D': 'spx_realized_vol_21d'}
        },
        {
            "filename": "vvix_daily_raw.csv",
            "use_cols": ['DATE', 'VVIX'],
            "rename_dict": {'DATE': 'Date', 'VVIX': 'vvix_level'}
        },
        {
            "filename": "credit_spread_high_yield.csv",
            "use_cols": ['date', 'BofA US High Yield Index Spread'],
            "rename_dict": {'date': 'Date', 'BofA US High Yield Index Spread': 'credit_spread_high_yield'}
        },
        {
            "filename": "credit_spread_high_yield_change_21d.csv",
            "use_cols": ['date', 'Credit Spread High Yield Change 21D'],
            "rename_dict": {'date': 'Date', 'Credit Spread High Yield Change 21D': 'credit_spread_change_21d'}
        },
        {
            "filename": "yield_curve_slope_10y_2y.csv",
            "use_cols": ['date', 'Yield Curve Slope 10Y/2Y'],
            "rename_dict": {'date': 'Date', 'Yield Curve Slope 10Y/2Y': 'yield_curve_slope_10y_2y'}
        },
        {
            "filename": "vix_term_structure_final.csv",
            "use_cols": ['Date', 'Term_Structure_Ratio_1_0', 'Term_Structure_Slope_2_1'],
            "rename_dict": {'Term_Structure_Ratio_1_0': 'term_structure_ratio_1_0', 'Term_Structure_Slope_2_1': 'term_structure_slope_2_1'}
        }
    ]

    # --- 3. Join all the DataFrames ---
    final_df = base_df
    print('\nJoining all other data sources...')
    for feature_info in features_to_load:
        # Handle the special case for the term structure file path
        is_term_structure = feature_info["filename"] == "vix_term_structure_final.csv"
        file_path = Path(term_structure_path) if is_term_structure else Path(raw_data_folder) / feature_info["filename"]
        
        feature_df = load_feature_df(
            file_path=file_path,
            use_cols=feature_info["use_cols"],
            rename_dict=feature_info["rename_dict"]
        )
        if feature_df is not None:
            final_df = final_df.join(feature_df)
        
    #Normalize Units:
    print("\nNormalizing units for percentage-based columns...")
    cols_to_normalize = ['credit_spread_high_yield', 'yield_curve_slope_10y_2y']
    for col in cols_to_normalize:
        if col in final_df.columns:
            final_df[col] = final_df[col] / 100
            print(f"  - Converted '{col}' to decimal format.")



    # --- 4. Final Cleanup ---
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    if (rows_removed := initial_rows - len(final_df)) > 0:
        print(f"Cleanup: Removed {rows_removed} rows with missing data after joining.")
        
    print("\nFinal dataset successfully assembled.")
    return final_df

def add_target_label(df, look_forward_days=21, threshold=20.0):
    """
    Creates the 'Panic_Imminent_21d' target label based on a future VIX spike.
    """
    print(f"\nCreating target label 'Panic_Imminent_21d'...")

    # 1. Create a rolling 21-day max (looks backwards)
    # 2. Shift the result series UP by 21 days to make it a "look forward"
    df['future_max_vix'] = df['vix_level'].rolling(window=look_forward_days, min_periods=1).max().shift(-look_forward_days)
    
    # Apply the rule: 1 if the future max VIX is > threshold, else 0
    df['Panic_Imminent_21d'] = (df['future_max_vix'] > threshold).astype(int)
    
    # Clean up the intermediate column
    df.drop(columns=['future_max_vix'], inplace=True)
    
    # Remove rows at the end where the label can't be calculated
    # (because their future window is incomplete)
    initial_rows = len(df)
    df.dropna(subset=['Panic_Imminent_21d'], inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Labeling Cleanup: Removed latest {rows_removed} rows that have an incomplete future window.")

    return df