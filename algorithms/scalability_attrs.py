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

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
def run_attr_scalability():
    print(f"{Colors.HEADER}🧪 STARTING MODE 3: ATTRIBUTE SCALABILITY{Colors.ENDC}")
    print(f"Dataset: {main_script.CHOSEN_DS}")
    
    # 1. Load Rules & Config
    with open(main_script.RULES_FILE, "r") as f:
        rules_list = [json.loads(line) for line in f]
    
    ds_config = main_script.ds_config
    max_attributes = ds_config.get("MAX_SCALABILITY_ATTRIBUTES", 10)
    target_col = main_script.TARGET_COLUMN_NAME
    
    # --- CLEAN FILES (Mode 3 only) ---
    print(f"{Colors.YELLOW}🧹 Cleaning old attribute scalability results...{Colors.ENDC}")
    main_script.clean_results_files(3)
    
    # 2. Load Full Dataset
    print(f"{Colors.BLUE}📂 Loading full dataset...{Colors.ENDC}")
    full_df = pd.read_csv(main_script.FULL_DATASET_PATH)
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]
    full_df = full_df[~full_df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)

    all_columns = [c for c in full_df.columns if c != target_col]
    max_features_to_add = max(0, max_attributes - 3)
    
    print(f"Max Attributes: {max_attributes} (Adding up to {max_features_to_add} extras)")

    # 3. Execution Loop
    for n_features in range(0, max_features_to_add + 1, 2):
        total_cols = n_features + 3
        print(f"\n{Colors.YELLOW}>>> Testing Feature Count: {n_features} (Total Cols: {total_cols}){Colors.ENDC}")

        for r in range(REPEATS_PER_CONFIG):
            print(f"   🔄 Repeat {r + 1}/{REPEATS_PER_CONFIG}...")
            
            for algo_name in SCALABILITY_ALGORITHMS:
                t0 = time.time()
                for i, rule in enumerate(rules_list):
                    print(f"      [Rule {i + 1}/{len(rules_list)}] Processing...", end="\r")

                    cond_col = list(rule["condition"].keys())[0]
                    treat_col = list(rule["treatment"].keys())[0]
                    mandatory = {cond_col, treat_col, target_col}

                    remaining_pool = [c for c in all_columns if c not in mandatory]
                    count_to_pick = min(n_features, len(remaining_pool))

                    random.seed(42 + i + n_features + r)
                    selected_features = random.sample(remaining_pool, count_to_pick)
                    final_cols = list(mandatory) + selected_features
                    attr_filtered_df = full_df[final_cols].copy()

                    main_script.process_dataset_dynamic(
                        i, rule, full_df, chosen_mode=3,
                        chosen_algorithm_name=algo_name, tgtO=target_col,
                        override_df=attr_filtered_df, metric_value=n_features
                    )
                print(f"      👉 {algo_name} done ({time.time() - t0:.2f}s)    ")

    print(f"\n{Colors.GREEN}✅ Attribute Tests Complete. Generating Graphs...{Colors.ENDC}")
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

    # --- 2. Attributes Processing ---
    attr_file = results_dir / f"{main_script.CHOSEN_DS}_scalability_attributes.csv"
    if attr_file.exists():
        df_attrs = pd.read_csv(attr_file)
        
        # Rename FPGrowth -> Brute Force
        df_attrs['algorithm'] = df_attrs['algorithm'].replace({'FPGrowth': 'Brute Force'})
        
        # Runtime Graph
        df_attrs_agg = df_attrs.groupby(['algorithm', 'num_attributes'])['run_time_seconds'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_attrs_agg, x='num_attributes', y='run_time_seconds', hue='algorithm', style='algorithm',
                     markers=True, markersize=8)
        plt.title(f'Scalability: Attributes ({main_script.CHOSEN_DS})')
        plt.xlabel('Number of Additional Features')
        plt.ylabel('Avg Runtime (s)')
        # Sort X axis to ensure clean lines
        plt.xticks(sorted(df_attrs_agg['num_attributes'].unique()))
        plt.grid(True, alpha=0.3)
        plt.savefig(results_dir / f"{main_script.CHOSEN_DS}_scalability_attributes_graph.png")
        plt.close()

        # Accuracy Graph (RW vs FPGrowth)
        df_raw_attr = pd.read_csv(attr_file)
        acc_df_attr = calculate_rw_accuracy(df_raw_attr, 'num_attributes')
        
        if not acc_df_attr.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=acc_df_attr, x='num_attributes', y='accuracy', markers=True, marker='o', color='purple')
            plt.title(f'RW Accuracy vs Attributes ({main_script.CHOSEN_DS})')
            plt.xlabel('Number of Additional Features')
            plt.ylabel('Accuracy (1.0 = Match with Brute Force)')
            plt.ylim(-0.1, 1.1)
            plt.xticks(sorted(acc_df_attr['num_attributes'].unique()))
            plt.grid(True, alpha=0.3)
            plt.savefig(results_dir / f"{main_script.CHOSEN_DS}_scalability_attributes_accuracy_graph.png")
            plt.close()
            

if __name__ == "__main__":
    run_attr_scalability()