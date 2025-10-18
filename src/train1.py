import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score
)
import numpy as np
import matplotlib.pyplot as plt

def run_and_log_evaluation():
    """
    Executes the cross-validation exactly as specified and logs all textual
    and visual outputs to a single, versioned directory.
    """
    # --- 1. Define and Create Output Paths ---
    # Follows the exact structure: model_outputs/logistic_regression/1/
    base_output_dir = Path("../model_outputs")
    experiment_dir = base_output_dir / "logistic_regression" / "run_1"
    figures_dir = experiment_dir / "figures"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"All outputs will be saved in: '{experiment_dir}'")
    
    # This list will capture every line of text for the final report
    report_lines = []

    # --- 2. Load Data and Perform the Hold-Out Split ---
    processed_path = Path("../data/processed_1/final_labeled_dataset.csv")
    df = pd.read_csv(processed_path, index_col='Date', parse_dates=True)

    cutoff_date = '2019-12-31'
    development_df = df[df.index <= cutoff_date]
    X_dev = development_df.drop(columns=['Panic_Imminent_21d'])
    y_dev = development_df['Panic_Imminent_21d']

    # --- 3. Set up Rolling Window Parameters and Correct Fold Calculation ---
    n_days_in_year, train_years, test_years, embargo_days = 252, 4, 1, 21
    train_size, test_size, step_size = train_years * n_days_in_year, test_years * n_days_in_year, test_years * n_days_in_year
    n_samples = len(X_dev)
    
    n_folds = ((n_samples - train_size - embargo_days - test_size) // step_size) + 1
    report_lines.append(f"For the sake of assurance, I want to confirm that this is the first logistic regression evaluation script.")
    
    report_lines.append(f"\nSetting up a rolling window with {n_folds} folds on {n_samples} development samples...")

    all_fold_metrics = []
    scaler = StandardScaler()
    if isinstance(scaler, StandardScaler):
        report_lines.append("  Using StandardScaler for feature scaling.")
    elif isinstance(scaler, MinMaxScaler):
        report_lines.append("  Using MinMaxScaler for feature scaling.")
    else:
        report_lines.append("  Using an unknown scaler for feature scaling.")
    # --- 4. Perform the Rolling Window Loop ---
    for i in range(n_folds):
        train_start, train_end = i * step_size, i * step_size + train_size
        test_start, test_end = train_end + embargo_days, train_end + embargo_days + test_size
        
        if test_end > n_samples: break

        X_train, X_test = X_dev.iloc[train_start:train_end], X_dev.iloc[test_start:test_end]
        y_train, y_test = y_dev.iloc[train_start:train_end], y_dev.iloc[test_start:test_end]

        fold_num = i + 1
        report_lines.append(f"\n----- Fold {fold_num}/{n_folds} -----")
        report_lines.append("Date Ranges:")
        report_lines.append(f"  Training on: {X_train.index.min().date()} to {X_train.index.max().date()}")
        report_lines.append(f"  Testing on:  {X_test.index.min().date()} to {X_test.index.max().date()}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Capture the classification report string
        report_lines.append("\n  Classification Report:")
        report_lines.append(classification_report(y_test, y_pred, zero_division=0))
        
        metrics = {
            'recall_1': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'precision_1': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall_0': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            'precision_0': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        all_fold_metrics.append(metrics)
        report_lines.append(f"  Matthews Corr. Coef. (MCC): {metrics['mcc']:.3f}")

        # Save the visualization for this fold
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold_num}', fontsize=16)
        plt.legend(loc="lower right")
        plt.savefig(figures_dir / f"roc_curve_fold_{fold_num}.png", dpi=400)
        plt.close()

    # --- 5. Capture Final Summary and Save the Single Text File ---
    summary_df = pd.DataFrame(all_fold_metrics)
    report_lines.append("\n\n--- Cross-Validation Summary ---")
    report_lines.append("Averaged performance across all folds:")
    # Convert the pandas Series to a string to append it
    report_lines.append(summary_df.mean().round(3).to_string())
    
    report_lines.append("\n--- Evaluation Script Finished ---")
    
    # Join all captured lines and write to a single file
    final_report_string = "\n".join(report_lines)
    report_path = experiment_dir / "full_evaluation_log.txt"
    with open(report_path, "w") as f:
        f.write(final_report_string)
        
    print(f"\nFull textual output successfully saved to: {report_path}")

if __name__ == "__main__":
    run_and_log_evaluation()