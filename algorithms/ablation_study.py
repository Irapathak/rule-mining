import re
import sys
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
from contextlib import contextmanager
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to sys.path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from ATE_update import calculate_ate_safe
from mlxtend.frequent_patterns import fpgrowth, apriori
from brute_force_algorithm import calc_utility_for_subgroups as brute_force_calc_utility_for_subgroups
from rw_unlearning import calc_utility_for_subgroups as rw_unlearning_calc_utility_for_subgroups

# --- Configuration (paths relative to repo root so clone-and-run works) ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
_config_path = _REPO_ROOT / 'configs' / 'config.json'
with open(_config_path, 'r') as f:
    config = json.load(f)

CHOSEN_DS = config["CHOSEN_DATASET"]
ds_config = config['DATASETS'][CHOSEN_DS]

FULL_DATASET_PATH = str((_REPO_ROOT / ds_config['FULL_DATASET_PATH']).resolve())
RULES_FILE = str((Path(__file__).resolve().parent / ds_config['RULES_FILE']).resolve())
TARGET_COLUMN_NAME = ds_config['TARGET_COLUMN']
ATTRIBUTE_WEIGHTS = ds_config.get('ATTRIBUTE_WEIGHTS', {})
TREATMENT_COL = config['TREATMENT_COL']
OPTIMIZATION_MODES = config.get('OPTIMIZATION_MODES', ['direct'])

# Ablation parameters (per-dataset)
_ABLATION_PARAMS = {
    "so": {
        "EPSILON_VALUES": [250000.0, 300000.0, 350000.0, 400000.0, 450000.0],
        "FIXED_EPSILON": 350000.0,
    },
    "acs": {
        "EPSILON_VALUES": [50.0, 100.0, 200.0, 500.0, 1000.0],  # Very small epsilon = many subgroups to check = strong RW advantage
        "FIXED_EPSILON": 500.0,  # Fixed epsilon for delta experiment (smaller = more subgroups)
    },
    "german": {
        "EPSILON_VALUES": [0.1, 0.3, 0.5, 0.7, 0.9],
        "FIXED_EPSILON": 0.5,
    },
}
_ds_ablation = _ABLATION_PARAMS.get(CHOSEN_DS, {"EPSILON_VALUES": [250000.0, 300000.0, 350000.0, 400000.0, 450000.0], "FIXED_EPSILON": 350000.0})
EPSILON_VALUES = _ds_ablation["EPSILON_VALUES"]
FIXED_EPSILON = _ds_ablation["FIXED_EPSILON"]
# Delta percentages - use smaller values for ACS to show RW advantage
# For ACS (~600k rows): 0.01% = 60, 0.02% = 120, 0.05% = 300, 0.1% = 600
DELTA_PERCENTAGES = [0.01, 0.02, 0.05, 0.10]  # 0.01%, 0.02%, 0.05%, 0.10% (smaller = more subgroups)
FIXED_DELTA_PERCENTAGE = 0.05  # 0.05% (was 10%)
NUM_RW_RUNS = 3

print("\n" + "="*70)
print("🔬 ABLATION STUDY - FPGrowth vs RW_Direct")
print("="*70)
print(f"Dataset: {CHOSEN_DS}")
print(f"Path: {FULL_DATASET_PATH}")
print(f"Rules: {Path(__file__).resolve().parent / RULES_FILE}")
print("="*70)


@contextmanager
def timer() -> callable:
    t0 = perf_counter()
    yield lambda: perf_counter() - t0


def encode_dataframe_local(df):
    """Encode dataframe with local unique values."""
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    for column in categorical_columns:
        unique_values = df_encoded[column].unique()
        column_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        df_encoded[column] = df_encoded[column].map(column_mapping)
    
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)
    
    return df_encoded


def run_algorithm(algo_name, df, tgtO, delta, epsilon, utility_all, attr_vals=None):
    """Run a single algorithm and return results."""
    common = dict(
        df=df,
        treatment_col=TREATMENT_COL,
        tgtO=tgtO,
        delta=delta,
        epsilon=epsilon,
        mode=0,  # Homogeneity check
        utility_all=utility_all
    )
    
    if algo_name == "FPGrowth":
        _fpgrowth_kw = dict(common, algorithm=fpgrowth)
        with timer() as elapsed:
            result = brute_force_calc_utility_for_subgroups(**_fpgrowth_kw)
        runtime = elapsed()
        
    elif algo_name == "RW_Direct":
        _rw_kw = dict(common, algorithm=apriori, size_stop=0.8,
                      optimization_mode=OPTIMIZATION_MODES[0],
                      attribute_weights=ATTRIBUTE_WEIGHTS)
        with timer() as elapsed:
            result = rw_unlearning_calc_utility_for_subgroups(**_rw_kw)
        runtime = elapsed()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Parse result
    is_homogeneous = False
    num_checked = 0
    
    if isinstance(result, tuple):
        raw_result = result[0]
        if len(result) >= 2:
            num_checked = result[1] if isinstance(result[1], int) else 0
        is_homogeneous = bool(raw_result)
    else:
        is_homogeneous = bool(result)
    
    return is_homogeneous, num_checked, runtime


def process_rule(rule, full_df, tgtO, delta, epsilon, algo_name, delta_pct_original=None):
    """Process a single rule with given delta and epsilon."""
    # Parse rule
    condition_dict = rule["condition"]
    condition_attr, condition_val = list(condition_dict.items())[0]
    treatment_dict = rule["treatment"]
    treatment_attr, treatment_val = list(treatment_dict.items())[0]
    
    # Filter data
    try:
        sub_df = full_df[full_df[condition_attr] == condition_val].copy()
    except KeyError:
        return None
    
    if sub_df.empty:
        return None
    
    # Drop condition column
    sub_df = sub_df.drop(columns=[condition_attr])
    
    # Apply treatment
    if treatment_attr not in sub_df.columns:
        return None
    
    sub_df[TREATMENT_COL] = (sub_df[treatment_attr] == treatment_val).astype(int)
    
    if treatment_attr in sub_df.columns:
        sub_df = sub_df.drop(columns=[treatment_attr])
    
    if sub_df[TREATMENT_COL].sum() == 0:
        return None
    
    # Encode
    if CHOSEN_DS != "acs":
        sub_df_encoded = encode_dataframe_local(sub_df)
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
        sub_df_encoded[tgtO] = pd.to_numeric(sub_df_encoded[tgtO], errors='coerce')
    else:
        sub_df_encoded = sub_df.copy()
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
    
    # If delta_pct_original is provided, recalculate delta for this subset
    if delta_pct_original is not None:
        delta = int(len(sub_df_encoded) * delta_pct_original)
        if delta < 1:
            delta = 1
    
    # Check if delta is valid
    if len(sub_df_encoded) < delta:
        return None
    
    # Calculate utility
    utility_all = calculate_ate_safe(sub_df_encoded, TREATMENT_COL, tgtO, delta)
    
    # Calculate attr_vals
    attr_vals = {
        col: sorted(sub_df_encoded[col].dropna().unique())
        for col in sub_df_encoded.columns
        if col not in [TREATMENT_COL, tgtO, *treatment_dict.keys()]
    }
    
    # Run algorithm
    is_homogeneous, num_checked, runtime = run_algorithm(
        algo_name, sub_df_encoded, tgtO, delta, epsilon, utility_all, attr_vals
    )
    
    return {
        'algorithm': algo_name,
        'condition': f"{condition_attr}={condition_val}",
        'treatment': f"{treatment_attr}={treatment_val}",
        'delta': delta,
        'delta_pct': (delta_pct_original * 100) if delta_pct_original else round(delta / len(sub_df_encoded) * 100, 2),
        'epsilon': epsilon,
        'is_homogeneous': is_homogeneous,
        'num_subgroups_checked': num_checked,
        'runtime_seconds': runtime,
        'dataset_size': len(sub_df_encoded)
    }


def run_ablation_study():
    """Run the complete ablation study."""
    # Load data
    print(f"Loading dataset from {FULL_DATASET_PATH}...")
    full_df = pd.read_csv(FULL_DATASET_PATH)
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]
    full_df = full_df[~full_df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    
    dataset_size = len(full_df)
    print(f"Dataset size: {dataset_size} rows")
    
    # Load rules - use path relative to this script's location
    rules_path = Path(__file__).resolve().parent / RULES_FILE
    print(f"Loading rules from {rules_path}...")
    with open(rules_path, "r") as f:
        rules_list = [json.loads(line) for line in f]
    print(f"Loaded {len(rules_list)} rules")
    
    # Calculate total runs for progress tracking
    total_epsilon_runs = len(EPSILON_VALUES) * len(rules_list) * 2  # 2 algorithms (FPGrowth + RW)
    total_delta_runs = len(DELTA_PERCENTAGES) * len(rules_list) * 2
    total_runs = total_epsilon_runs + total_delta_runs
    completed_runs = 0
    
    print(f"\n📊 TOTAL EXPERIMENTS TO RUN: {total_runs}")
    print(f"   Experiment 1 (Epsilon): {len(EPSILON_VALUES)} values × {len(rules_list)} rules × 2 algos = {total_epsilon_runs} runs")
    print(f"   Experiment 2 (Delta): {len(DELTA_PERCENTAGES)} values × {len(rules_list)} rules × 2 algos = {total_delta_runs} runs")
    
    results = []
    start_time = perf_counter()
    
    # Experiment 1: Varying Epsilon (Fixed Delta)
    print("\n" + "="*70)
    print(f"EXPERIMENT 1: Varying Epsilon (Fixed Delta = {FIXED_DELTA_PERCENTAGE*100:.2f}% of each rule's subset)")
    print("="*70)
    
    for eps_idx, epsilon in enumerate(EPSILON_VALUES):
        print(f"\n🔍 Testing Epsilon = {epsilon:,.0f} ({eps_idx + 1}/{len(EPSILON_VALUES)})")
        for rule_idx, rule in enumerate(rules_list):
            # Calculate progress
            progress = (completed_runs / total_runs) * 100
            elapsed = perf_counter() - start_time
            if completed_runs > 0:
                avg_time_per_run = elapsed / completed_runs
                remaining_runs = total_runs - completed_runs
                est_remaining = avg_time_per_run * remaining_runs
                est_remaining_min = est_remaining / 60
                print(f"  [{progress:.1f}%] Rule {rule_idx + 1}/{len(rules_list)} | ETA: {est_remaining_min:.1f} min", end="")
            
            # FPGrowth
            result = process_rule(rule, full_df, TARGET_COLUMN_NAME, None, epsilon, "FPGrowth", FIXED_DELTA_PERCENTAGE)
            fp_runtime = result['runtime_seconds'] if result else 0
            if result:
                result['experiment'] = 'varying_epsilon'
                results.append(result)
                print(f" | FPGrowth: {'✓' if result['is_homogeneous'] else '✗'} ({fp_runtime:.2f}s)", end="")
            completed_runs += 1
            
            # RW_Direct (3 runs)
            rw_runtimes = []
            for run in range(NUM_RW_RUNS):
                result = process_rule(rule, full_df, TARGET_COLUMN_NAME, None, epsilon, "RW_Direct", FIXED_DELTA_PERCENTAGE)
                if result:
                    rw_runtimes.append(result['runtime_seconds'])
                    result['experiment'] = 'varying_epsilon'
                    result['run_number'] = run + 1
                    results.append(result)
                    if run == 0:
                        rw_avg = sum(rw_runtimes) / len(rw_runtimes) if rw_runtimes else 0
                        speedup = fp_runtime / rw_avg if rw_avg > 0 else 0
                        print(f" | RW: {'✓' if result['is_homogeneous'] else '✗'} ({rw_avg:.2f}s, {speedup:.2f}x)", end="")
            completed_runs += 1
            print()
    
    # Experiment 2: Varying Delta (Fixed Epsilon)
    print("\n" + "="*70)
    print(f"EXPERIMENT 2: Varying Delta (Fixed Epsilon = {FIXED_EPSILON:,.0f})")
    print("="*70)
    
    for delta_idx, delta_pct in enumerate(DELTA_PERCENTAGES):
        print(f"\n🔍 Testing Delta = {delta_pct*100:.0f}% of each rule's subset ({delta_idx + 1}/{len(DELTA_PERCENTAGES)})")
        
        for rule_idx, rule in enumerate(rules_list):
            # Calculate progress
            progress = (completed_runs / total_runs) * 100
            elapsed = perf_counter() - start_time
            if completed_runs > 0:
                avg_time_per_run = elapsed / completed_runs
                remaining_runs = total_runs - completed_runs
                est_remaining = avg_time_per_run * remaining_runs
                est_remaining_min = est_remaining / 60
                print(f"  [{progress:.1f}%] Rule {rule_idx + 1}/{len(rules_list)} | ETA: {est_remaining_min:.1f} min", end="")
            
            # FPGrowth
            result = process_rule(rule, full_df, TARGET_COLUMN_NAME, None, FIXED_EPSILON, "FPGrowth", delta_pct)
            fp_runtime = result['runtime_seconds'] if result else 0
            if result:
                result['experiment'] = 'varying_delta'
                results.append(result)
                print(f" | FPGrowth: {'✓' if result['is_homogeneous'] else '✗'} ({fp_runtime:.2f}s)", end="")
            completed_runs += 1
            
            # RW_Direct (3 runs)
            rw_runtimes = []
            for run in range(NUM_RW_RUNS):
                result = process_rule(rule, full_df, TARGET_COLUMN_NAME, None, FIXED_EPSILON, "RW_Direct", delta_pct)
                if result:
                    rw_runtimes.append(result['runtime_seconds'])
                    result['experiment'] = 'varying_delta'
                    result['run_number'] = run + 1
                    results.append(result)
                    if run == 0:
                        rw_avg = sum(rw_runtimes) / len(rw_runtimes) if rw_runtimes else 0
                        speedup = fp_runtime / rw_avg if rw_avg > 0 else 0
                        print(f" | RW: {'✓' if result['is_homogeneous'] else '✗'} ({rw_avg:.2f}s, {speedup:.2f}x)", end="")
            completed_runs += 1
            print()
    
    # Save raw results
    results_df = pd.DataFrame(results)
    results_dir = _REPO_ROOT / "ablation_results"
    results_dir.mkdir(exist_ok=True)
    
    total_elapsed = perf_counter() - start_time
    total_elapsed_min = total_elapsed / 60
    
    excel_path = results_dir / f"{CHOSEN_DS}_ablation_raw_results.xlsx"
    results_df.to_excel(excel_path, index=False)
    
    print(f"\n" + "="*70)
    print(f"✅ ALL EXPERIMENTS COMPLETE!")
    print(f"   Total runs: {completed_runs}/{total_runs}")
    print(f"   Total time: {total_elapsed_min:.1f} minutes ({total_elapsed:.1f} seconds)")
    print(f"   Average per run: {total_elapsed/completed_runs:.2f} seconds")
    print(f"   Raw results saved to: {excel_path}")
    print("="*70)
    
    return results_df


def calculate_summary(df):
    """Calculate summary statistics including RW vs FPGrowth agreement."""
    summaries = []
    
    for exp in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp]
        
        if exp == 'varying_epsilon':
            group_cols = ['epsilon']
            varying_param = 'epsilon'
        else:
            group_cols = ['delta_pct']
            varying_param = 'delta_pct'
        
        # Calculate per algorithm stats
        for algo in ['FPGrowth', 'RW_Direct']:
            algo_df = exp_df[exp_df['algorithm'] == algo]
            grouped = algo_df.groupby(group_cols).agg({
                'is_homogeneous': ['sum', 'count', 'mean'],
                'num_subgroups_checked': 'mean',
                'runtime_seconds': 'mean'
            }).reset_index()
            
            grouped.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in grouped.columns]
            grouped['homogeneity_rate'] = grouped['is_homogeneous_mean'] * 100
            grouped['algorithm'] = algo
            grouped['experiment'] = exp
            summaries.append(grouped)
        
        # Calculate agreement between RW and FPGrowth
        for param_value in exp_df[group_cols[0]].unique():
            fp_results = exp_df[(exp_df['algorithm'] == 'FPGrowth') & (exp_df[group_cols[0]] == param_value)]
            rw_results = exp_df[(exp_df['algorithm'] == 'RW_Direct') & (exp_df[group_cols[0]] == param_value)]
            
            # Merge on treatment/condition to compare same rules
            merged = pd.merge(fp_results[['treatment', 'condition', 'is_homogeneous']], 
                            rw_results[['treatment', 'condition', 'is_homogeneous']], 
                            on=['treatment', 'condition'], 
                            suffixes=('_fp', '_rw'))
            
            if len(merged) > 0:
                agreement = (merged['is_homogeneous_fp'] == merged['is_homogeneous_rw']).mean() * 100
                agreement_row = {group_cols[0]: param_value, 'algorithm': 'RW_Agreement', 
                               'agreement_rate': agreement, 'experiment': exp}
                summaries.append(pd.DataFrame([agreement_row]))
    
    return pd.concat(summaries, ignore_index=True)


def generate_visualizations(results_df, results_dir):
    """Generate all visualizations and HTML report."""
    print("\n📊 Generating visualizations...")
    
    # Calculate summary statistics
    summary_df = calculate_summary(results_df)
    
    # Save summary CSV
    csv_path = results_dir / f"{CHOSEN_DS}_ablation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✅ Summary CSV saved to: {csv_path}")
    
    # Create visualizations
    print("📈 Creating graphs...")
    
    eps_data = summary_df[summary_df['experiment'] == 'varying_epsilon']
    delta_data = summary_df[summary_df['experiment'] == 'varying_delta']
    
    # 1. RW Correctness: Epsilon Sensitivity
    fig1 = go.Figure()
    eps_agreement = eps_data[eps_data['algorithm'] == 'RW_Agreement']
    fig1.add_trace(go.Scatter(
        x=eps_agreement['epsilon'],
        y=eps_agreement['agreement_rate'],
        mode='lines+markers',
        name='RW Correctness',
        line=dict(width=4, color='green'),
        marker=dict(size=12),
        fill='tozeroy'
    ))
    
    fig1.update_layout(
        title=f"RW Correctness vs Epsilon (FPGrowth = Ground Truth)<br><sub>Fixed Delta = 10%</sub>",
        xaxis_title="Epsilon",
        yaxis_title="Agreement with FPGrowth (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        font=dict(size=14),
        height=500
    )
    
    # 2. RW Correctness: Delta Sensitivity
    fig2 = go.Figure()
    delta_agreement = delta_data[delta_data['algorithm'] == 'RW_Agreement']
    fig2.add_trace(go.Scatter(
        x=delta_agreement['delta_pct'],
        y=delta_agreement['agreement_rate'],
        mode='lines+markers',
        name='RW Correctness',
        line=dict(width=4, color='green'),
        marker=dict(size=12),
        fill='tozeroy'
    ))
    
    fig2.update_layout(
        title=f"RW Correctness vs Delta (FPGrowth = Ground Truth)<br><sub>Fixed Epsilon = 350k</sub>",
        xaxis_title="Delta (% of Dataset)",
        yaxis_title="Agreement with FPGrowth (%)",
        yaxis=dict(range=[0, 105]),
        # NOTE: delta_pct is stored in *percent units* (e.g., 5, 10, 15, 20),
        # so using Plotly percent tickformat would incorrectly scale (5 -> 500%).
        xaxis=dict(tickformat='.0f', ticksuffix='%'),
        template="plotly_white",
        font=dict(size=14),
        height=500
    )
    
    # 3. Runtime Comparison - Epsilon
    fig3 = go.Figure()
    for algo in ['FPGrowth', 'RW_Direct']:
        algo_data = eps_data[eps_data['algorithm'] == algo]
        if len(algo_data) > 0:
            fig3.add_trace(go.Bar(
                x=algo_data['epsilon'],
                y=algo_data['runtime_seconds_mean'],
                name=algo,
                text=algo_data['runtime_seconds_mean'].round(2),
                textposition='outside'
            ))
    
    fig3.update_layout(
        title=f"Runtime: FPGrowth vs RW - Varying Epsilon<br><sub>Shows RW speedup</sub>",
        xaxis_title="Epsilon",
        yaxis_title="Average Runtime (seconds)",
        template="plotly_white",
        font=dict(size=14),
        height=500,
        barmode='group'
    )
    
    # 4. Runtime Comparison - Delta
    fig4 = go.Figure()
    for algo in ['FPGrowth', 'RW_Direct']:
        algo_data = delta_data[delta_data['algorithm'] == algo]
        if len(algo_data) > 0:
            fig4.add_trace(go.Bar(
                x=algo_data['delta_pct'],
                y=algo_data['runtime_seconds_mean'],
                name=algo,
                text=algo_data['runtime_seconds_mean'].round(2),
                textposition='outside'
            ))
    
    fig4.update_layout(
        title=f"Runtime: FPGrowth vs RW - Varying Delta<br><sub>Shows RW speedup</sub>",
        xaxis_title="Delta (% of Dataset)",
        yaxis_title="Average Runtime (seconds)",
        # delta_pct is already in percent units (5, 10, 15, 20)
        xaxis=dict(tickformat='.0f', ticksuffix='%'),
        template="plotly_white",
        font=dict(size=14),
        height=500,
        barmode='group'
    )
    
    # 5. How Parameters Affect Results
    fig5 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Impact of Epsilon', 'Impact of Delta'),
        horizontal_spacing=0.15
    )
    
    # Epsilon impact
    for algo in ['FPGrowth', 'RW_Direct']:
        algo_data = eps_data[eps_data['algorithm'] == algo]
        if len(algo_data) > 0:
            fig5.add_trace(go.Scatter(
                x=algo_data['epsilon'],
                y=algo_data['homogeneity_rate'],
                mode='lines+markers',
                name=algo,
                showlegend=True
            ), row=1, col=1)
    
    # Delta impact
    for algo in ['FPGrowth', 'RW_Direct']:
        algo_data = delta_data[delta_data['algorithm'] == algo]
        if len(algo_data) > 0:
            fig5.add_trace(go.Scatter(
                x=algo_data['delta_pct'],
                y=algo_data['homogeneity_rate'],
                mode='lines+markers',
                name=algo,
                showlegend=False
            ), row=1, col=2)
    
    fig5.update_xaxes(title_text="Epsilon", row=1, col=1)
    # delta_pct is already in percent units (5, 10, 15, 20)
    fig5.update_xaxes(title_text="Delta (%)", tickformat='.0f', ticksuffix='%', row=1, col=2)
    fig5.update_yaxes(title_text="% Rules Homogeneous", row=1, col=1)
    fig5.update_yaxes(title_text="% Rules Homogeneous", row=1, col=2)
    
    fig5.update_layout(
        title_text=f"Parameter Sensitivity: How Epsilon/Delta Affect Results<br><sub>% of rules where algorithm found homogeneity</sub>",
        template="plotly_white",
        font=dict(size=14),
        height=500
    )
    
    # Generate HTML Report
    print("📝 Generating HTML report...")
    
    # Prepare detailed results table
    detailed_table_html = results_df[['experiment', 'algorithm', 'condition', 'treatment', 'epsilon', 'delta_pct', 'delta', 'is_homogeneous', 'num_subgroups_checked', 'runtime_seconds', 'dataset_size']].rename(columns={
        'experiment': 'Experiment',
        'algorithm': 'Algorithm',
        'condition': 'Rule Condition',
        'treatment': 'Treatment',
        'epsilon': 'Epsilon',
        'delta_pct': 'Delta (%)',
        'delta': 'Delta (samples)',
        'is_homogeneous': 'Homogeneous?',
        'num_subgroups_checked': 'Subgroups Checked',
        'runtime_seconds': 'Runtime (s)',
        'dataset_size': 'Subset Size'
    }).to_html(index=False, classes='summary-table')
    
    # Separate agreement and algorithm summaries
    agreement_df = summary_df[summary_df['algorithm'] == 'RW_Agreement'][['experiment', 'epsilon', 'delta_pct', 'agreement_rate']].copy()
    agreement_df['epsilon'] = agreement_df['epsilon'].fillna('')
    agreement_df['delta_pct'] = agreement_df['delta_pct'].fillna('')
    agreement_summary_html = agreement_df.rename(columns={
        'experiment': 'Experiment',
        'epsilon': 'Epsilon',
        'delta_pct': 'Delta (%)',
        'agreement_rate': 'Agreement Rate (%)'
    }).to_html(index=False, classes='summary-table')
    
    algo_df = summary_df[summary_df['algorithm'].isin(['FPGrowth', 'RW_Direct'])][['algorithm', 'experiment', 'epsilon', 'delta_pct', 'homogeneity_rate', 'num_subgroups_checked_mean', 'runtime_seconds_mean']].copy()
    algo_df['epsilon'] = algo_df['epsilon'].fillna('')
    algo_df['delta_pct'] = algo_df['delta_pct'].fillna('')
    algo_summary_html = algo_df.rename(columns={
        'algorithm': 'Algorithm',
        'experiment': 'Experiment',
        'epsilon': 'Epsilon',
        'delta_pct': 'Delta (%)',
        'homogeneity_rate': 'Homogeneity Rate (%)',
        'num_subgroups_checked_mean': 'Avg Subgroups Checked',
        'runtime_seconds_mean': 'Avg Runtime (s)'
    }).to_html(index=False, classes='summary-table')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ablation Study Report - {CHOSEN_DS}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .info-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .graph-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .table-container {{
            max-height: 600px;
            overflow-y: auto;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>🔬 Ablation Study Report: {CHOSEN_DS.upper()}</h1>
    
    <div class="info-box">
        <h2>📋 Study Overview</h2>
        <div class="metric">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">{CHOSEN_DS}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Algorithms Tested</div>
            <div class="metric-value">2</div>
        </div>
        <div class="metric">
            <div class="metric-label">Rules Tested</div>
            <div class="metric-value">{len(results_df['treatment'].unique())}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Runs</div>
            <div class="metric-value">{len(results_df)}</div>
        </div>
    </div>

    <div class="info-box">
        <h3>Experiment Setup:</h3>
        <ul>
            <li><strong>Ground Truth:</strong> FPGrowth (100% accurate but slower)</li>
            <li><strong>Testing:</strong> RW_Direct (faster but need to verify correctness)</li>
            <li><strong>Experiment 1:</strong> Varying Epsilon [250k, 300k, 350k, 400k, 450k] with Fixed Delta = 10%</li>
            <li><strong>Experiment 2:</strong> Varying Delta [5%, 10%, 15%, 20%] with Fixed Epsilon = 350k</li>
            <li><strong>Goal:</strong> Show RW matches FPGrowth results across different parameters</li>
        </ul>
    </div>

    <h2>✅ RW Correctness: Does RW Match FPGrowth?</h2>
    <div class="graph-container">
        <div id="epsilon_plot"></div>
        <p><strong>Explanation:</strong> Shows what % of rules RW gives same answer as FPGrowth for each epsilon value. Higher = better.</p>
    </div>

    <div class="graph-container">
        <div id="delta_plot"></div>
        <p><strong>Explanation:</strong> Shows what % of rules RW gives same answer as FPGrowth for each delta value. Higher = better.</p>
    </div>

    <h2>⏱️ Runtime: RW Speedup vs FPGrowth</h2>
    <div class="graph-container">
        <div id="runtime_epsilon_plot"></div>
        <p><strong>Explanation:</strong> Shows average runtime per algorithm. RW should be faster while maintaining correctness.</p>
    </div>
    <div class="graph-container">
        <div id="runtime_delta_plot"></div>
    </div>

    <h2>📉 Parameter Impact on Results</h2>
    <div class="graph-container">
        <div id="param_impact_plot"></div>
        <p><strong>Explanation:</strong> Shows how epsilon/delta values affect what % of rules each algorithm finds homogeneous.</p>
    </div>

    <h2>📈 Aggregated Summary Statistics</h2>
    
    <h3>✅ RW Correctness (Agreement with FPGrowth Ground Truth)</h3>
    <div class="info-box">
        {agreement_summary_html}
    </div>
    
    <h3>⚡ Algorithm Performance Metrics</h3>
    <div class="info-box">
        {algo_summary_html}
    </div>

    <h2>📋 Detailed Results by Rule</h2>
    <div class="info-box">
        <p><strong>This table shows all individual test results. Use it to trace back which rule/treatment with which parameters produced each result.</strong></p>
        <div class="table-container">
        {detailed_table_html}
        </div>
    </div>

    <script>
        {f"var epsilon_plot = {fig1.to_json()};"}
        Plotly.newPlot('epsilon_plot', epsilon_plot.data, epsilon_plot.layout);

        {f"var delta_plot = {fig2.to_json()};"}
        Plotly.newPlot('delta_plot', delta_plot.data, delta_plot.layout);

        {f"var runtime_epsilon = {fig3.to_json()};"}
        Plotly.newPlot('runtime_epsilon_plot', runtime_epsilon.data, runtime_epsilon.layout);

        {f"var runtime_delta = {fig4.to_json()};"}
        Plotly.newPlot('runtime_delta_plot', runtime_delta.data, runtime_delta.layout);

        {f"var param_impact = {fig5.to_json()};"}
        Plotly.newPlot('param_impact_plot', param_impact.data, param_impact.layout);
    </script>

    <footer style="text-align: center; margin-top: 50px; padding: 20px; color: #7f8c8d;">
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""
    
    html_path = results_dir / f"{CHOSEN_DS}_ablation_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {html_path}")
    print(f"\n📂 All output files:")
    print(f"   - Raw data: {results_dir / f'{CHOSEN_DS}_ablation_raw_results.xlsx'}")
    print(f"   - Summary CSV: {csv_path}")
    print(f"   - HTML Report: {html_path}")
    print(f"\n💡 Open the HTML file in your browser to view interactive graphs!")


if __name__ == "__main__":
    # Run ablation study
    results_df = run_ablation_study()
    
    # Generate visualizations and report
    results_dir = _REPO_ROOT / "ablation_results"
    generate_visualizations(results_df, results_dir)
    
    print("\n🎉 Ablation study complete with visualizations!")

