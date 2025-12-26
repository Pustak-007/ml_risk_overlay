#In this module we are training the data on the new processed dataset (i.e. processed_2)
# The label is based on future (21 days onwards) spx drawdown. 
# - Based on whether that drawdown is greater than 5% or not.

""" The Definition: For every specific date (let's say, today), the code looks forward at the next 21 trading days (approx. 1 calendar month). It calculates the lowest price the S&P 500 hits during that period.
The Logic:
Label = 1 (DANGER): If the S&P 500 drops 5% or more from today's price at any point in the next month.
Label = 0 (SAFE): If the S&P 500 never drops more than 5% in the next month.
"""

"""We are currently using a simple logistic regression model, and we will be deciding whether this model is 
 feasible or not based on the metrics like mcc, accuracy, precision - across all folds and on average.
 """
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
   classification_report,
   recall_score,
   precision_score,
   matthews_corrcoef,
   roc_auc_score
)

def train_and_evaluate_display_only():
   """
   Runs a rolling window cross-validation and displays all metrics directly in the terminal.
   """

   processed_path = Path("data/processed_2/final_labeled_dataset_v2.csv")
   df = pd.read_csv(processed_path, index_col='Date', parse_dates=True)

   cutoff_date = '2019-12-31'
   development_df = df[df.index <= cutoff_date]
   X_dev = development_df.drop(columns=['spx_drawdown_5pct_21d'])
   y_dev = development_df['spx_drawdown_5pct_21d']

   n_days_in_year, train_years, test_years, embargo_days = 252, 4, 1, 21
   train_size, test_size, step_size = train_years * n_days_in_year, test_years * n_days_in_year, test_years * n_days_in_year
   n_samples = len(X_dev)
  
   n_folds = ((n_samples - train_size - embargo_days - test_size) // step_size) + 1
  
   print(f"\nSetting up a rolling window with {n_folds} folds on {n_samples} development samples...")
   print("Using StandardScaler for feature scaling based on experimental results.")

   all_fold_metrics = []

   for i in range(n_folds):
       train_start = i * step_size
       train_end = train_start + train_size
       test_start = train_end + embargo_days
       test_end = test_start + test_size
      
       if test_end > n_samples:
           break

       X_train, X_test = X_dev.iloc[train_start:train_end], X_dev.iloc[test_start:test_end]
       y_train, y_test = y_dev.iloc[train_start:train_end], y_dev.iloc[test_start:test_end]

       fold_num = i + 1
       print(f"\n----- Fold {fold_num}/{n_folds} -----")
       print("Date Ranges:")
       print(f"  Training on: {X_train.index.min().date()} to {X_train.index.max().date()}")
       print(f"  Testing on:  {X_test.index.min().date()} to {X_test.index.max().date()}")
      
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)
       #
       model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
       model.fit(X_train_scaled, y_train)
      
       y_pred = model.predict(X_test_scaled)
       y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
      
       print("\nClassification Report:")
       print(classification_report(y_test, y_pred, zero_division=0))
      
       metrics = {
           'recall_1': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
           'precision_1': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
           'mcc': matthews_corrcoef(y_test, y_pred),
           'roc_auc': roc_auc_score(y_test, y_pred_proba)
       }
       all_fold_metrics.append(metrics)
       print(f"Matthews Correlation Coefficient (MCC): {metrics['mcc']:.3f}")

   summary_df = pd.DataFrame(all_fold_metrics)
   print("\n\n--- Cross-Validation Summary ---")
   print("Averaged performance across all folds:")
   print(summary_df.mean().round(3))

if __name__ == "__main__":
   train_and_evaluate_display_only()
