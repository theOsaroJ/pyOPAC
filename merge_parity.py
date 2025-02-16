# pyOPAC/merge_parity.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def merge_predictions_actuals(predictions_file, actual_file, output_merged="merged_predictions.csv"):
    """
    Merge predictions and actual CSV files on 'mol_id' and save the merged DataFrame.
    """
    df_pred = pd.read_csv(predictions_file)
    df_act = pd.read_csv(actual_file)
    df_merged = pd.merge(df_act, df_pred, on='mol_id')
    df_merged.to_csv(output_merged, index=False)
    return df_merged

def create_parity_plot(df_merged, property_mapping):
    """
    Create parity plots for each property pair.
    
    Parameters:
      df_merged (pd.DataFrame): Merged DataFrame.
      property_mapping (dict): Mapping of predicted column names to actual column names,
                               e.g., {"Predicted_Property_1": "Property1", "Predicted_Property_2": "Property2"}
    """
    for pred_col, act_col in property_mapping.items():
        if pred_col not in df_merged.columns or act_col not in df_merged.columns:
            print(f"Warning: Missing columns for {pred_col} or {act_col}. Skipping parity plot.")
            continue
        
        property_name = act_col  # e.g., 'Property1'
        actual = df_merged[act_col]
        predicted = df_merged[pred_col]

        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.5, label='Data Points')
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Parity')
        plt.xlabel(f"Actual {property_name}")
        plt.ylabel(f"Predicted {property_name}")
        plt.title(f"Parity Plot: {property_name}")

        metrics_text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f} eV\nRMSE: {rmse:.3f} eV"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
        plt.legend()

        plot_filename = f"parity_{property_name}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved parity plot for {property_name} as {plot_filename}")
