import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import random
import time

# Import your existing module
import all_subgroups_loop as main_script

# --- Configuration ---
SCALABILITY_ALGORITHMS = ["RW", "FPGrowth"]
REPEATS_PER_CONFIG = 3
ROW_PERCENTAGES = [x / 10 for x in range(1, 11)]

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_scalability_rows():
    print(f"{Colors.HEADER}🧪 STARTING SCALABILITY TEST SUITE (ROWS){Colors.ENDC}")
    print(f"Dataset: {main_script.CHOSEN_DS}")

    # 1. Load Rules & Config
    with open(main_script.RULES_FILE, "r") as f:
        rules_list = [json.loads(line) for line in f]

    ds_config = main_script.ds_config
    max_attributes = ds_config.get("MAX_SCALABILITY_ATTRIBUTES", 10)
    target_col = main_script.TARGET_COLUMN_NAME

    print(f"Max Attributes Configured: {max_attributes}")

    # --- CLEAN FILES BEFORE STARTING ---
    #print(f"{Colors.YELLOW}🧹 Cleaning old scalability results (Rows)...{Colors.ENDC}")
    #main_script.clean_results_files(2) # Clean Row Scalability CSV
    # ---------------------------------

    # 2. Load Full Dataset & Clean (Parity with mode 0)
    print(f"{Colors.BLUE}📂 Loading full dataset: {main_script.FULL_DATASET_PATH}{Colors.ENDC}")
    full_df = pd.read_csv(main_script.FULL_DATASET_PATH)
    
    # --- CRITICAL CLEANING STEP ---
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]
    full_df = full_df[~full_df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    # ------------------------------

    # ==========================================
    # MODE 2: ROW SCALABILITY (10% -> 100%)
    # ==========================================
    print(f"\n{Colors.BOLD}" + "=" * 60)
    print("📊 MODE 2: ROW SCALABILITY TEST")
    print("=" * 60 + f"{Colors.ENDC}")

    for p in ROW_PERCENTAGES:
        percent_str = f"{int(p * 100)}%"
        print(f"\n{Colors.YELLOW}>>> Testing Dataset Fraction: {percent_str}{Colors.ENDC}")

        for r in range(REPEATS_PER_CONFIG):
            seed = 42 + int(p * 100) + r
            print(f"   🔄 Repeat {r + 1}/{REPEATS_PER_CONFIG} (Seed {seed})...")

            if p == 1.0:
                sampled_df = full_df # Use original order, no shuffling
            else:
                sampled_df = full_df.sample(frac=p, random_state=seed)

            for algo_name in SCALABILITY_ALGORITHMS:
                t0 = time.time()

                # --- SIMULATE MAIN LOOP (ALL RULES) ---
                for i, rule in enumerate(rules_list):
                    if i not in [7,8,9] or algo_name != "FPGrowth":
                        print(f"{Colors.YELLOW} skipping rule {i+1}...{Colors.ENDC}")
                        continue
                    # Progress Indicator
                    print(f"      [Rule {i + 1}/{len(rules_list)}] Processing...", end="\r")

                    main_script.process_dataset_dynamic(
                        i, rule, full_df, chosen_mode=2,
                        chosen_algorithm_name=algo_name, tgtO=target_col,
                        override_df=sampled_df, metric_value=p
                    )
                # --------------------------------------

                print(f"      👉 {algo_name} (All Rules) done ({time.time() - t0:.2f}s)    ")
    
    print(f"\n{Colors.GREEN}✅ Row Tests Complete. Generating Graphs...{Colors.ENDC}")
    generate_graphs()

def calculate_rw_accuracy(df_data, metric_col):
    """
    Calculates accuracy of RW against FPGrowth (Ground Truth).
    Returns a DataFrame: [metric_value, accuracy]
    Includes Logic: If FPGrowth TIMEOUT -> Assume Homogeneous.
    """
    # Create a unique key for each rule run
    df_data['run_key'] = df_data['chosen_condition'].astype(str) + "|" + df_data['chosen_treatment'].astype(str)
    
    accuracy_results = []
    
    # Iterate through each distinct configuration (e.g., 100%, 90%...)
    metrics = df_data[metric_col].unique()
    
    for metric in metrics:
        subset = df_data[df_data[metric_col] == metric]
        
        # Split into Ground Truth (FPGrowth) and Prediction (RW)
        ground_truth = subset[subset['algorithm'] == 'FPGrowth'].set_index('run_key')['status']
        predictions = subset[subset['algorithm'] == 'RW']
        
        if ground_truth.empty or predictions.empty:
            continue
            
        correct_count = 0
        total_count = 0
        
        for _, row in predictions.iterrows():
            key = row['run_key']
            rw_status = row['status']
            
            if key in ground_truth.index:
                # Compare statuses (Homogeneous vs NOT Homogeneous)
                # Note: Ground truth might have duplicates due to repeats, take mode or first
                gt_status = ground_truth.loc[key]
                if isinstance(gt_status, pd.Series):
                    gt_status = gt_status.iloc[0]
                
                # --- LOGIC: Treat TIMEOUT as Homogeneous ---
                if gt_status == "TIMEOUT":
                    gt_status = "Homogeneous"
                # -------------------------------------------
                
                if rw_status == gt_status:
                    correct_count += 1
                total_count += 1
        
        accuracy = (correct_count / total_count) if total_count > 0 else 0.0
        accuracy_results.append({metric_col: metric, 'accuracy': accuracy})
        
    return pd.DataFrame(accuracy_results)

def generate_graphs():
    results_dir = main_script._REPO_ROOT / "graphs"

    # --- 1. Rows Processing ---
    row_file = results_dir / f"{main_script.CHOSEN_DS}_scalability_rows.csv"
    if row_file.exists():
        df_rows = pd.read_csv(row_file)
        
        # Rename FPGrowth -> Brute Force
        df_rows['algorithm'] = df_rows['algorithm'].replace({'FPGrowth': 'Brute Force'})
        
        # Runtime Graph
        df_rows_agg = df_rows.groupby(['algorithm', 'dataset_percentage'])['run_time_seconds'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_rows_agg, x='dataset_percentage', y='run_time_seconds', hue='algorithm', style='algorithm',
                     markers=True, markersize=8)
        plt.title(f'Scalability: Rows ({main_script.CHOSEN_DS})')
        plt.xlabel('Dataset Percentage')
        plt.ylabel('Avg Runtime (s)')
        plt.xticks(ROW_PERCENTAGES, [f"{int(x * 100)}%" for x in ROW_PERCENTAGES])
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / f"{main_script.CHOSEN_DS}_scalability_rows_graph.png")
        plt.close()
        
        # Accuracy Graph (RW vs FPGrowth)
        # Reload original to separate algo names for logic
        df_raw = pd.read_csv(row_file) 
        acc_df = calculate_rw_accuracy(df_raw, 'dataset_percentage')
        
        if not acc_df.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=acc_df, x='dataset_percentage', y='accuracy', markers=True, marker='o', color='green')
            plt.title(f'RW Accuracy vs Rows ({main_script.CHOSEN_DS})')
            plt.xlabel('Dataset Percentage')
            plt.ylabel('Accuracy (1.0 = Match with Brute Force)')
            plt.ylim(-0.1, 1.1)
            plt.xticks(ROW_PERCENTAGES, [f"{int(x * 100)}%" for x in ROW_PERCENTAGES])
            plt.grid(True, alpha=0.3)
            plt.savefig(results_dir / f"{main_script.CHOSEN_DS}_scalability_rows_accuracy_graph.png")
            plt.close()

if __name__ == "__main__":
    run_scalability_rows()