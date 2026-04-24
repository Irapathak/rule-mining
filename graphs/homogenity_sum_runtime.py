import os
import json
import pandas as pd

# --- Configuration ---
try:
    with open('../configs/config.json', 'r') as f:
        config = json.load(f)
    CHOSEN_DS = config["CHOSEN_DATASET"]
except FileNotFoundError:
    print("Warning: '../configs/config.json' not found. Defaulting to 'german'.")
    CHOSEN_DS = "german"

INPUT_FILE = f"{CHOSEN_DS}_homogeneity_results.xlsx"
OUTPUT_FILE = f"{CHOSEN_DS}_homogeneity_runtime_summary.xlsx"

def summarize_runtimes(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    print(f"--- Calculating Runtime Statistics for {CHOSEN_DS} ---")

    # 1. Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    print(f"Reading data from {input_file}...")

    try:
        # Load the data (Try Excel, fallback to CSV)
        try:
            df = pd.read_excel(input_file)
        except Exception:
            df = pd.read_csv(input_file.replace('.xlsx', '.csv'))

        # Ensure necessary standard columns exist
        required_columns = ['algorithm', 'delta', 'epsilon', 'run_time_seconds']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input file is missing one of the required columns: {required_columns}")
            return
        
        # Clean algorithm names (strip whitespace)
        if 'algorithm' in df.columns:
            df['algorithm'] = df['algorithm'].astype(str).str.strip()

        # --- RENAME STEP: Change FPGrowth/Apriori to Brute Force ---
        # This checks if "FPGrowth" or "Apriori" is in the name (case-insensitive) and renames it
        mask = df['algorithm'].str.contains('FPGrowth|Apriori', case=False, na=False)
        if mask.any():
            print(f"  > Renaming {mask.sum()} rows from 'FPGrowth/Apriori' to 'Brute Force'")
            df.loc[mask, 'algorithm'] = 'Brute Force'

        # Handle Num Subgroups column
        if 'num_subgroups' not in df.columns:
            df['num_subgroups'] = 0
        else:
            df['num_subgroups'] = pd.to_numeric(df['num_subgroups'], errors='coerce').fillna(0)

        # 2. Group by Algorithm, Delta, and Epsilon
        # This ensures we capture EVERY unique combination present in the file
        print("Calculating averages, variance, and standard deviation...")

        summary_df = df.groupby(['algorithm', 'delta', 'epsilon']).agg(
            Runs_Count=('run_time_seconds', 'count'),
            Avg_Runtime_Sec=('run_time_seconds', 'mean'),
            Variance_Runtime=('run_time_seconds', 'var'),
            Std_Dev_Runtime=('run_time_seconds', 'std'),
            Avg_Num_Subgroups=('num_subgroups', 'mean')
        ).reset_index()

        # Fill NaN values (happens if only 1 run exists, std/var become NaN)
        cols_to_fill = ['Std_Dev_Runtime', 'Variance_Runtime', 'Avg_Num_Subgroups']
        summary_df[cols_to_fill] = summary_df[cols_to_fill].fillna(0)

        # Round for cleaner output
        round_cols = ['Avg_Runtime_Sec', 'Variance_Runtime', 'Std_Dev_Runtime', 'Avg_Num_Subgroups']
        for col in round_cols:
            summary_df[col] = summary_df[col].round(4)

        # Sort for better readability: 
        # Delta -> Epsilon -> Put "Brute Force" first -> Then alphabetical
        summary_df['sorter'] = summary_df['algorithm'].apply(lambda x: 0 if x == 'Brute Force' else 1)
        summary_df = summary_df.sort_values(['delta', 'epsilon', 'sorter', 'algorithm'])
        summary_df = summary_df.drop(columns=['sorter'])

        # 3. Write to Excel (Single Sheet)
        print(f"Writing results to {output_file}...")
        summary_df.to_excel(output_file, index=False, sheet_name="Runtime_Summary")

        print("Done! Summary file created successfully.")
        print(f"\nTotal unique epsilon values: {summary_df['epsilon'].nunique()}")
        print(f"Epsilon values: {sorted(summary_df['epsilon'].unique())}")
        print("\nPreview of Results (grouped by algorithm, delta, epsilon):")
        print(summary_df.to_string())

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    summarize_runtimes()