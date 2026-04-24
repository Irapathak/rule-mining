import pandas as pd
import os
import numpy as np
import re
import json
import argparse
import sys

# --- Configuration ---
try:
    with open('../configs/config.json', 'r') as f:
        config = json.load(f)
    CHOSEN_DS = config["CHOSEN_DATASET"]
except FileNotFoundError:
    print("Warning: '../configs/config.json' not found. Defaulting to 'german'.")
    CHOSEN_DS = "german"
    config = {"DATASETS": {"german": {"EPSILONS": []}}} # Fallback

RESULTS_FILE = f"{CHOSEN_DS}_homogeneity_results.xlsx"
OUTPUT_FILE = f"{CHOSEN_DS}_homogeneity_metrics_summary.xlsx"


def to_boolean(val):
    """Robust boolean converter."""
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)): return val != 0
    s = str(val).strip().lower()
    return s in ['true', 't', '1', '1.0', 'yes']


def calculate_summary_metrics():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Calculate homogeneity metrics.')
    parser.add_argument('-c', '--config', action='store_true', 
                        help='Config mode: Only process epsilons specified in config.json')
    args = parser.parse_args()

    print("--- Calculating Summarized Metrics (All Algos in One Sheet) ---")
    if args.config:
        print("   [Mode: Config-restricted (filtering by config.json epsilons)]")
    else:
        print("   [Mode: Auto-detect (processing all epsilons in results file)]")

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    # Try reading as Excel, fallback to CSV if necessary
    try:
        df = pd.read_excel(RESULTS_FILE)
    except Exception:
        df = pd.read_csv(RESULTS_FILE.replace('.xlsx', '.csv'))

    if 'algorithm' in df.columns:
        df['algorithm'] = df['algorithm'].astype(str).str.strip()

    # --- 1. Identify Ground Truth (GT) ---
    GT_NAME = "Brute-Force Algorithm"
    unique_algos = df['algorithm'].unique()

    # Dynamic GT Detection
    if GT_NAME not in unique_algos:
        matches = [x for x in unique_algos if 'Brute' in x or 'Apriori' in x or 'FPGrowth' in x]
        if matches:
            GT_NAME = matches[0]
            print(f"Ground Truth Algorithm identified as: {GT_NAME}")
        else:
            print("Error: Could not identify Ground Truth (Brute/Apriori).")
            return

    # Identify Test Algorithms (Everyone else)
    test_algorithms = [algo for algo in unique_algos if algo != GT_NAME]

    # Get unique configs available in the data
    params = df[['delta', 'epsilon']].drop_duplicates().sort_values(['delta', 'epsilon'])

    # --- FILTERING LOGIC FOR -c FLAG ---
    if args.config:
        try:
            # Get target epsilons from config
            target_epsilons = config['DATASETS'][CHOSEN_DS].get('EPSILONS', [])
            # Handle case where it might be a single float instead of a list
            if isinstance(target_epsilons, (int, float)):
                target_epsilons = [target_epsilons]
            
            if not target_epsilons:
                print("Warning: Config mode enabled but no EPSILONS found in config.json.")
            
            # Filter params
            original_count = len(params)
            params = params[params['epsilon'].isin(target_epsilons)]
            print(f"   > Filtered epsilons: {len(params)} kept out of {original_count} found in file.")
            
        except KeyError:
            print(f"Error: Could not find EPSILONS for dataset {CHOSEN_DS} in config.")
            return

    final_rows = []

    # --- 2. Iterate Configs and Algorithms ---
    for _, row in params.iterrows():
        delta = row['delta']
        epsilon = row['epsilon']

        # Get Ground Truth for this config
        gt_subset = df[
            (df['algorithm'] == GT_NAME) & 
            (df['delta'] == delta) & 
            (df['epsilon'] == epsilon)
        ]

        # --- Explicitly add the Ground Truth row as "Brute Force" ---
        if not gt_subset.empty:
            final_rows.append({
                "Algorithm": "Brute Force",  # Renaming as requested
                "Delta": delta,
                "Epsilon": epsilon,
                "Precision (%)": 100.0,  # Hardcoded 100%
                "Specificity (%)": 100.0,  # Hardcoded 100%
                "Total Runs Checked": len(gt_subset)
            })

        # Build GT Lookup for comparison with other algorithms
        gt_lookup = {}
        for _, g_row in gt_subset.iterrows():
            k = (g_row['treatment'], g_row['condition'])
            gt_lookup[k] = to_boolean(g_row['homogeneity_status'])

        if not gt_lookup:
            continue

        # Process each test algorithm against this GT
        for algo_name in test_algorithms:
            algo_subset = df[
                (df['algorithm'] == algo_name) & 
                (df['delta'] == delta) & 
                (df['epsilon'] == epsilon)
            ]

            if algo_subset.empty: continue

            run_precisions = []
            run_specificities = []

            for _, run_row in algo_subset.iterrows():
                key = (run_row['treatment'], run_row['condition'])
                if key not in gt_lookup: continue

                gt_val = gt_lookup[key]
                pred_val = to_boolean(run_row['homogeneity_status'])

                # Metrics Calculation (Scientific Per-Run Method)
                tp = 1 if (pred_val and gt_val) else 0
                fp = 1 if (pred_val and not gt_val) else 0
                tn = 1 if (not pred_val and not gt_val) else 0

                # Precision
                if (tp + fp) > 0:
                    p = tp / (tp + fp)
                else:
                    p = 1.0  # Silence is Golden

                # Specificity
                if (tn + fp) > 0:
                    s = tn / (tn + fp)
                else:
                    s = 1.0  # No Negatives existed or Perfect TN

                run_precisions.append(p)
                run_specificities.append(s)

            # Average
            if run_precisions:
                avg_p = np.mean(run_precisions) * 100
                avg_s = np.mean(run_specificities) * 100
                total_runs = len(run_precisions)
            else:
                avg_p, avg_s, total_runs = 0, 0, 0

            final_rows.append({
                "Algorithm": algo_name,
                "Delta": delta,
                "Epsilon": epsilon,
                "Precision (%)": round(avg_p, 2),
                "Specificity (%)": round(avg_s, 2),
                "Total Runs Checked": total_runs
            })

    # --- 3. Output ---
    if final_rows:
        summary_df = pd.DataFrame(final_rows)
        # Sort: Delta -> Epsilon -> Put "Brute Force" first -> Then alphabetical
        summary_df['sorter'] = summary_df['Algorithm'].apply(lambda x: 0 if x == 'Brute Force' else 1)

        summary_df = summary_df.sort_values(['Delta', 'Epsilon', 'sorter', 'Algorithm'])
        summary_df = summary_df.drop(columns=['sorter'])

        print(f"Writing summary to {OUTPUT_FILE}...")
        summary_df.to_excel(OUTPUT_FILE, index=False)
        print("Done.")
        print(f"\nTotal unique epsilon values: {summary_df['Epsilon'].nunique()}")
        print(f"Epsilon values: {sorted(summary_df['Epsilon'].unique())}")
        print("\nResults (grouped by algorithm, delta, epsilon):")
        print(summary_df)
    else:
        print("No results found.")

if __name__ == "__main__":
    calculate_summary_metrics()