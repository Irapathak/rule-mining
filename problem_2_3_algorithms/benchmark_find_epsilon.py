"""
Benchmark for Find Smallest Epsilon Algorithm.

Tests the binary search algorithm across multiple rules and delta values
to evaluate efficiency and performance.

Outputs:
- Excel file with detailed results
- HTML report with beautiful visualizations
- PNG plots
"""
import sys
import json
import time
import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import math
import re

# seaborn is optional; we fall back to matplotlib-only plots if it's not installed.
try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sns = None

# Add project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']


def load_treatments(treatment_file: str = None) -> List[Dict]:
    """Load treatment-condition pairs from JSON file."""
    if treatment_file is None:
        # Use the local treatments file in problem_2_3_algorithms/
        treatment_file = Path(__file__).resolve().parent / "Chosen10Treatments.json"
    
    treatments = []
    with open(treatment_file, "r") as f:
        for line in f:
            treatments.append(json.loads(line))
    return treatments


def _prepare_rule_df(base_df, rule, outcome_col):
    """Prepare a filtered DataFrame for one rule on-the-fly (no pre-computed CSV).

    Used when a base encoded dataset exists (e.g. ACS) instead of pre-computed
    per-rule CSVs (SO path).

    Returns
    -------
    (df, actual_outcome_col)  or  (None, outcome_col) if the rule is empty.
    """
    condition = rule['condition']
    treatment_dict = rule['treatment']

    cond_attr, cond_val = list(condition.items())[0]
    treat_attr, treat_val = list(treatment_dict.items())[0]

    sub_df = base_df[base_df[cond_attr] == cond_val].copy()
    if sub_df.empty:
        return None, outcome_col

    # Drop condition column (invariant after filtering)
    sub_df = sub_df.drop(columns=[cond_attr])

    # Create binary treatment column
    sub_df[TREATMENT_COL] = (sub_df[treat_attr] == treat_val).astype(int)

    # Drop treatment source column
    if treat_attr in sub_df.columns:
        sub_df = sub_df.drop(columns=[treat_attr])

    if sub_df[TREATMENT_COL].sum() == 0:
        return None, outcome_col

    # Clean column names (special chars like commas cause FPGrowth issues)
    sub_df = sub_df.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
    actual_outcome_col = re.sub(r'[,:\[\]\{\}"]', '_', outcome_col)

    # Ensure outcome is numeric
    sub_df[actual_outcome_col] = pd.to_numeric(sub_df[actual_outcome_col], errors='coerce')

    return sub_df, actual_outcome_col


def run_benchmark(
    num_rules: int = 5,
    delta_values: List[int] = None,
    epsilon_0: float = None,
    epsilon_max_cap: float = 1_000_000_000.0,
    output_dir: str = "benchmark_results",
    verbose: bool = False,
    # ── Dataset-specific parameters ──
    treatment_file: str = None,
    outcome_col: str = 'ConvertedSalary',
    base_dataset_path: str = None,
    delta_percentages: List[float] = None,
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across multiple rules and delta values.
    
    Args:
        num_rules: Number of rules to test
        delta_values: List of absolute delta values to test (SO path).
        epsilon_0: Initial ε₀ for Phase 1 exponential search (None = auto, default 1000)
        epsilon_max_cap: Maximum epsilon cap to prevent infinite search
        output_dir: Directory for output files
        verbose: Print detailed progress
        treatment_file: Path to treatments JSON (None → default SO file)
        outcome_col: Name of the outcome column
        base_dataset_path: Path to encoded base CSV for on-the-fly prep.
                           If None, pre-computed per-rule SO CSVs are used.
        delta_percentages: If set, overrides *delta_values* with percentages of
                           each rule's per-rule dataset size (e.g. [5, 10, 15, 20]).
    """
    use_pct = delta_percentages is not None
    if delta_values is None and not use_pct:
        delta_values = [500, 1000, 1500, 2000, 2500, 3000]
    
    # Create output directory
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("BENCHMARK: Two-Phase Find Smallest Epsilon Algorithm")
    print("="*80)
    if use_pct:
        n_delta = len(delta_percentages)
        print(f"Testing {num_rules} rules with {n_delta} delta-% values each")
        print(f"Total experiments: {num_rules * n_delta}")
        print(f"Delta percentages: {delta_percentages}")
    else:
        n_delta = len(delta_values)
        print(f"Testing {num_rules} rules with {n_delta} delta values each")
        print(f"Total experiments: {num_rules * n_delta}")
        print(f"Delta values: {delta_values}")
    if epsilon_0 is None:
        print(f"Phase 1: Exponential search to find ε_max (starting from ε₀ = 1000)")
    else:
        print(f"Phase 1: Exponential search to find ε_max (starting from ε₀ = {epsilon_0:,.0f})")
    print(f"Phase 2: Binary search on [0, ε_max]")
    print(f"Maximum epsilon cap: {epsilon_max_cap:,.0f}")
    print("="*80)
    
    # Load treatments
    treatments = load_treatments(treatment_file)[:num_rules]
    
    # Pre-load base dataset once for on-the-fly mode (ACS path)
    base_df = None
    if base_dataset_path is not None:
        print(f"\nLoading base dataset: {base_dataset_path}")
        base_df = pd.read_csv(base_dataset_path)
        # Remove Unnamed columns (same as ablation study - line 220)
        base_df = base_df.loc[:, ~base_df.columns.str.startswith('Unnamed')]
        print(f"  → {len(base_df)} rows, {len(base_df.columns)} columns")
    
    # Results storage
    results = []
    
    # Counter for progress
    total_experiments = num_rules * (len(delta_percentages) if use_pct else len(delta_values))
    current_experiment = 0
    
    # Run benchmark
    for rule_idx, treatment_data in enumerate(treatments, 1):
        condition = treatment_data['condition']
        treatment = treatment_data['treatment']
        
        # ── Prepare dataset for this rule ──
        if base_df is not None:
            df, actual_outcome_col = _prepare_rule_df(base_df, treatment_data, outcome_col)
            if df is None:
                print(f"\n⚠️  Warning: Rule {rule_idx} produced empty dataset. Skipping.")
                continue
        else:
            dataset_path = Path(f'../stackoverflow/processed_db/so_countries_treatment_{rule_idx}_encoded.csv')
            if not dataset_path.exists():
                print(f"\n⚠️  Warning: Dataset for rule {rule_idx} not found. Skipping.")
                continue
            df = pd.read_csv(dataset_path)
            actual_outcome_col = outcome_col
        
        # ── Resolve delta values for this rule ──
        if use_pct:
            rule_deltas = [max(1, int(len(df) * pct / 100)) for pct in delta_percentages]
        else:
            rule_deltas = delta_values
        
        print(f"\n{'='*80}")
        print(f"RULE {rule_idx}/{num_rules}")
        print(f"Condition: {condition}")
        print(f"Treatment: {treatment}")
        print(f"Dataset size: {len(df)} rows")
        if use_pct:
            print(f"Delta values (from %): {rule_deltas}")
        print(f"{'='*80}")
        
        for delta in rule_deltas:
            current_experiment += 1
            
            if len(df) < delta:
                print(f"\n[{current_experiment}/{total_experiments}] ⚠️  Skipping delta={delta} (dataset too small)")
                continue
            
            print(f"\n[{current_experiment}/{total_experiments}] Testing delta = {delta:,}")
            
            # Run algorithm and measure time
            start_time = time.time()
            
            smallest_epsilon, oracle_calls, violation_info, utility_all, phase1_calls, phase2_calls = find_smallest_epsilon_achieving_homogeneity(
                df=df,
                treatment_col=TREATMENT_COL,
                outcome_col=actual_outcome_col,
                delta=delta,
                epsilon_0=epsilon_0,
                epsilon_max_cap=epsilon_max_cap,
                verbose=False  # Suppress detailed output
            )
            
            elapsed_time = time.time() - start_time
            # NOTE:
            # If smallest_epsilon is None, it means epsilon_max was not homogeneous, so ε* does not exist
            # within [0, epsilon_max]. We keep it as 'None' and keep violation_info as a witness at ε_max.
            
            # Extract violation details
            violating_subgroup = str(violation_info['subgroup']) if violation_info else 'N/A'
            subgroup_size = violation_info['size'] if violation_info else 'N/A'
            subgroup_utility = violation_info['utility'] if violation_info else 'N/A'
            utility_diff = violation_info['utility_diff'] if violation_info else 'N/A'
            abs_diff = violation_info['abs_diff'] if violation_info else 'N/A'
            
            # Store results
            result = {
                'Rule_ID': rule_idx,
                'Condition': str(condition),
                'Treatment': str(treatment),
                'Delta': delta,
                'Smallest_Epsilon_Homogeneous': smallest_epsilon if smallest_epsilon is not None else 'None',
                'Oracle_Calls': oracle_calls,
                'Phase1_Calls': phase1_calls,
                'Phase2_Calls': phase2_calls,
                'Runtime_Seconds': round(elapsed_time, 3),
                'Runtime_Minutes': round(elapsed_time / 60, 3),
                'Dataset_Size': len(df),
                'Population_Utility': round(utility_all, 2) if utility_all else 'N/A',
                'Violating_Subgroup': violating_subgroup,
                'Subgroup_Size': subgroup_size,
                'Subgroup_Utility': round(float(subgroup_utility), 2) if subgroup_utility != 'N/A' else 'N/A',
                'Utility_Difference': round(float(utility_diff), 2) if utility_diff != 'N/A' else 'N/A',
                'Abs_Utility_Difference': round(float(abs_diff), 2) if abs_diff != 'N/A' else 'N/A'
            }
            results.append(result)
            
            # Print summary
            eps_str = f"{smallest_epsilon:,.0f}" if smallest_epsilon is not None else "Not found"
            print(f"   ✓ Result: ε* = {eps_str}")
            print(f"   ✓ Oracle calls: {oracle_calls}")
            print(f"   ✓ Runtime: {elapsed_time:.2f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    return results_df


def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from benchmark results."""
    summary = pd.DataFrame({
        'Metric': [
            'Total Experiments',
            'Avg Runtime (seconds)',
            'Avg Oracle Calls',
            'Min Oracle Calls',
            'Max Oracle Calls',
            'Total Runtime (minutes)'
        ],
        'Value': [
            len(results_df),
            results_df['Runtime_Seconds'].mean(),
            results_df['Oracle_Calls'].mean(),
            results_df['Oracle_Calls'].min(),
            results_df['Oracle_Calls'].max(),
            results_df['Runtime_Seconds'].sum() / 60
        ]
        })
    summary['Value'] = summary['Value'].round(3)
    return summary


def create_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Create visualization plots for benchmark results."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Oracle Calls vs Delta
    ax1 = axes[0, 0]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        ax1.plot(rule_data['Delta'], rule_data['Oracle_Calls'], 
                marker='o', label=f'Rule {rule_id}', linewidth=2)
    ax1.set_xlabel('Delta (δ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Oracle Calls (Iterations)', fontsize=12, fontweight='bold')
    ax1.set_title('Search Efficiency: Oracle Calls vs Delta', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs Delta
    ax2 = axes[0, 1]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        ax2.plot(rule_data['Delta'], rule_data['Runtime_Seconds'], 
                marker='s', label=f'Rule {rule_id}', linewidth=2)
    ax2.set_xlabel('Delta (δ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Runtime vs Delta', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Found Epsilon vs Delta
    ax3 = axes[1, 0]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        valid_data = rule_data[rule_data['Smallest_Epsilon_Homogeneous'] != 'None'].copy()
        if not valid_data.empty:
            valid_data['Smallest_Epsilon_Homogeneous'] = pd.to_numeric(valid_data['Smallest_Epsilon_Homogeneous'])
            ax3.plot(valid_data['Delta'], valid_data['Smallest_Epsilon_Homogeneous'], 
                    marker='^', label=f'Rule {rule_id}', linewidth=2)
    ax3.set_xlabel('Delta (δ)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Smallest ε (Homogeneous)', fontsize=12, fontweight='bold')
    ax3.set_title('Found ε* vs Delta', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Oracle Calls Distribution
    ax4 = axes[1, 1]
    ax4.hist(results_df['Oracle_Calls'], bins=15, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax4.axvline(results_df['Oracle_Calls'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {results_df["Oracle_Calls"].mean():.1f}')
    ax4.set_xlabel('Oracle Calls', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Oracle Calls Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_path / 'benchmark_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Visualization saved: {plot_path}")
    plt.close()
    
    # Heatmap (seaborn if available; otherwise matplotlib imshow)
    pivot_oracle = results_df.pivot(index='Rule_ID', columns='Delta', values='Oracle_Calls')

    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.heatmap(pivot_oracle, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Oracle Calls'})
    else:
        import numpy as np
        data = pivot_oracle.values.astype(float)
        im = plt.imshow(data, aspect='auto')
        plt.colorbar(im, label='Oracle Calls')
        plt.xticks(range(len(pivot_oracle.columns)), [f"{int(x):,}" for x in pivot_oracle.columns], rotation=45)
        plt.yticks(range(len(pivot_oracle.index)), [str(int(x)) for x in pivot_oracle.index])
        # annotate values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{int(val)}", ha='center', va='center', fontsize=8, color='black')

    plt.title('Oracle Calls Heatmap: Rule vs Delta', fontsize=14, fontweight='bold')
    plt.xlabel('Delta (δ)', fontsize=12, fontweight='bold')
    plt.ylabel('Rule ID', fontsize=12, fontweight='bold')
    plt.tight_layout()
    heatmap_path = output_path / 'oracle_calls_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Heatmap saved: {heatmap_path}")
    plt.close()


def generate_html_report(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """Generate beautiful HTML report."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    html_path = output_path / 'benchmark_report.html'
    
    total_exp = len(results_df)
    avg_runtime = results_df['Runtime_Seconds'].mean()
    avg_oracle = results_df['Oracle_Calls'].mean()
    min_oracle = results_df['Oracle_Calls'].min()
    max_oracle = results_df['Oracle_Calls'].max()
    total_time = results_df['Runtime_Seconds'].sum() / 60
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Find Smallest Epsilon - Benchmark Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .table-scroll {{
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            margin: 15px 0;
        }}
        .table-scroll table {{
            width: max-content;
            min-width: 100%;
            margin: 0;
        }}
        
        .insight-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .insight-box h3 {{ margin-bottom: 15px; font-size: 1.3em; }}
        .insight-box .formula {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            font-size: 1.1em;
            font-family: 'Courier New', monospace;
            margin-top: 10px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .summary-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }}
        .summary-card h3 {{
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .summary-card .value {{
            font-size: 2.2em;
            font-weight: bold;
        }}
        
        h2 {{
            color: #f5576c;
            border-bottom: 3px solid #f5576c;
            padding-bottom: 10px;
            margin: 40px 0 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: visible;
            font-size: 0.9em;
        }}
        thead {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        th {{
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 0.5px;
        }}
        tbody tr {{ transition: all 0.2s; }}
        tbody tr:nth-child(even) {{ background: #f8f9fa; }}
        tbody tr:hover {{
            background: #ffe0f0;
            transform: scale(1.005);
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-homogeneous {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-heterogeneous {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-runtime {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        .badge-oracle {{
            background: #e2e3e5;
            color: #383d41;
            font-weight: bold;
        }}
        
        .highlight {{
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        
        .rule-group {{
            margin: 30px 0;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .rule-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            font-weight: bold;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .emoji {{ font-size: 1.2em; }}
        
        .glossary {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
        }}
        .glossary h3 {{
            color: #495057;
            margin-bottom: 20px;
        }}
        .glossary-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .glossary-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
            line-height: 1.6;
        }}
        .glossary-table tr:last-child td {{
            border-bottom: none;
        }}
        .glossary-table .metric-name {{
            font-weight: bold;
            color: #495057;
            width: 30%;
            background: #f8f9fa;
        }}
        .glossary-table .metric-desc {{
            color: #6c757d;
        }}
        .glossary-table .formula-code {{
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="emoji">🎯</span> Find Smallest Epsilon - Benchmark Results</h1>
            <p class="subtitle">Two-Phase Search Algorithm (Exponential + Binary)</p>
            <p class="subtitle" style="font-size: 0.9em; margin-top: 10px;">
                {total_exp} Experiments | {len(results_df['Rule_ID'].unique())} Rules | {len(results_df['Delta'].unique())} Delta Values | Runtime: {total_time:.2f} minutes
            </p>
        </header>
        
        <div class="content">
            <div class="insight-box">
                <h3><span class="emoji">💡</span> Algorithm Overview</h3>
                <p><strong>Phase 1: Exponential Search</strong> - Quickly find upper bound by doubling epsilon</p>
                <p><strong>Phase 2: Binary Search</strong> - Refine to find exact smallest epsilon</p>
                <div class="formula">
                    <strong>largest ε (heterogeneous)</strong> = <strong>smallest ε (homogeneous)</strong> - 1
                </div>
                <p style="margin-top: 10px;">
                    Above ε*, the rule is homogeneous. At ε* - 1, violations exist.
                </p>
            </div>
            
            <div class="glossary">
                <h3><span class="emoji">📖</span> Metrics Glossary - How Each Column is Calculated</h3>
                <div class="table-scroll">
                <table class="glossary-table">
                    <tr>
                        <td class="metric-name">Delta (δ)</td>
                        <td class="metric-desc">
                            <strong>Fixed Input Parameter.</strong> The minimum subgroup size threshold. 
                            All candidate subgroups must contain at least δ individuals to be evaluated.
                            <br><span class="formula-code">Input: User-specified value</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Smallest ε (Homogeneous)</td>
                        <td class="metric-desc">
                            <strong>Algorithm Output.</strong> The smallest epsilon threshold where the rule becomes homogeneous 
                            (no subgroups violate the threshold). This is the primary result we're searching for.
                            <br><span class="formula-code">Found via: Exponential search + Binary search</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Largest ε (Heterogeneous)</td>
                        <td class="metric-desc">
                            <strong>Derived Metric.</strong> The largest epsilon value where violations still exist. 
                            Due to monotonicity, this is exactly one less than the smallest homogeneous epsilon.
                            <br><span class="formula-code">Calculation: smallest_ε_homogeneous - 1</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Oracle Calls</td>
                        <td class="metric-desc">
                            <strong>Efficiency Metric.</strong> Total number of times we invoked the homogeneity oracle 
                            (FPGrowth algorithm) to check if a rule is homogeneous at a given epsilon. 
                            Lower is better; indicates search efficiency.
                            <br><span class="formula-code">Count: Total calls to FPGrowth oracle during binary search (plus one upper-bound check)</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Runtime (seconds)</td>
                        <td class="metric-desc">
                            <strong>Performance Metric.</strong> Wall-clock time (in seconds) to complete the entire 
                            binary search for this specific rule and delta combination (ε ∈ [0, ε_max]).
                            <br><span class="formula-code">Measurement: end_time - start_time</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Dataset Size</td>
                        <td class="metric-desc">
                            <strong>Context Information.</strong> The total number of rows (individuals) in the dataset 
                            for this particular rule. Larger datasets may require more computation per oracle call.
                            <br><span class="formula-code">Value: len(dataframe)</span>
                        </td>
                    </tr>
                    <tr>
                        <td class="metric-name">Rule ID</td>
                        <td class="metric-desc">
                            <strong>Identifier.</strong> Unique numerical identifier for each treatment-condition pair being tested.
                            <br><span class="formula-code">Value: 1, 2, 3, ...</span>
                        </td>
                    </tr>
                </table>
                </div>
            </div>
            
            <h2><span class="emoji">📊</span> Performance Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Experiments</h3>
                    <div class="value">{total_exp}</div>
                </div>
                <div class="summary-card">
                    <h3>Avg Runtime</h3>
                    <div class="value">{avg_runtime:.2f}s</div>
                </div>
                <div class="summary-card">
                    <h3>Avg Oracle Calls</h3>
                    <div class="value">{avg_oracle:.1f}</div>
                </div>
                <div class="summary-card">
                    <h3>Min Oracle Calls</h3>
                    <div class="value">{min_oracle}</div>
                </div>
                <div class="summary-card">
                    <h3>Max Oracle Calls</h3>
                    <div class="value">{max_oracle}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Runtime</h3>
                    <div class="value">{total_time:.1f}m</div>
                </div>
            </div>
            
            <h2><span class="emoji">📋</span> Detailed Results</h2>
"""
    
    # Group by rule
    for rule_id in sorted(results_df['Rule_ID'].unique()):
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        condition = rule_data.iloc[0]['Condition']
        treatment = rule_data.iloc[0]['Treatment']
        
        html += f"""
            <div class="rule-group">
                <div class="rule-header">
                    <span class="emoji">📌</span> Rule {rule_id}: {condition} → {treatment}
                </div>
                <div class="table-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Delta (δ)</th>
                            <th>Smallest ε*<br/>(Achieves Homogeneity)</th>
                            <th>Violating Subgroup<br/>(Reason for Boundary)</th>
                            <th>Phase 1<br/>(Bracketing)</th>
                            <th>Phase 2<br/>(Binary Search)</th>
                            <th>Total<br/>Oracle Calls</th>
                            <th>Runtime</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for _, row in rule_data.iterrows():
            delta = f"{int(row['Delta']):,}"
            smallest = row['Smallest_Epsilon_Homogeneous']
            
            if smallest != 'None' and not pd.isna(smallest):
                smallest = f"{int(float(smallest)):,}"
            else:
                smallest = 'N/A'
                
            oracle = int(row['Oracle_Calls'])
            phase1 = int(row['Phase1_Calls'])
            phase2 = int(row['Phase2_Calls'])
            runtime = row['Runtime_Seconds']
            
            # Format violation details
            if 'Violating_Subgroup' in row and row['Violating_Subgroup'] != 'N/A':
                subgroup = str(row['Violating_Subgroup'])[:50] + '...' if len(str(row['Violating_Subgroup'])) > 50 else str(row['Violating_Subgroup'])
                pop_util = row['Population_Utility']
                sub_util = row['Subgroup_Utility']
                abs_diff = row['Abs_Utility_Difference']
                violation_details = f"""
                    <div style="font-size: 0.85em; line-height: 1.4;">
                        <strong>{subgroup}</strong><br/>
                        Size: {row['Subgroup_Size']}<br/>
                        Pop ATE: {pop_util:.2f} | Sub ATE: {sub_util:.2f}<br/>
                        |Diff|: <strong>{abs_diff:.2f}</strong> (needs ε ≥ {smallest})
                    </div>
                """
            else:
                # Most common reason: epsilon_max was too low (algorithm returned None and has no subgroup info).
                # Keep messaging accurate.
                if smallest == 'N/A':
                    violation_details = '<span style="color: #999;">Not found within ε_max (increase --epsilon_max)</span>'
                else:
                    violation_details = '<span style="color: #999;">No violation found</span>'
            
            html += f"""
                        <tr>
                            <td><strong>{delta}</strong></td>
                            <td><span class="badge badge-homogeneous">{smallest}</span></td>
                            <td>{violation_details}</td>
                            <td><span class="badge badge-oracle">{phase1}</span></td>
                            <td><span class="badge badge-oracle">{phase2}</span></td>
                            <td><span class="badge badge-oracle">{oracle}</span></td>
                            <td><span class="badge badge-runtime">{runtime:.2f}s</span></td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
                </div>
            </div>
"""
    
    html += f"""
            <h2><span class="emoji">🎯</span> Key Insights</h2>
            <ul style="line-height: 2; font-size: 1.05em;">
                <li><span class="emoji">⚡</span> Binary search is <span class="highlight">highly efficient</span>: avg {avg_oracle:.1f} oracle calls</li>
                <li><span class="emoji">🔍</span> Binary search finds the smallest ε* with log₂(n) oracle calls (plus one upper-bound check)</li>
                <li><span class="emoji">✅</span> Algorithm scales well across different delta values</li>
            </ul>
        </div>
        
        <footer>
            <p><span class="emoji">🕐</span> Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Algorithm: Binary Search with FPGrowth Oracle | Dataset: Stack Overflow</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"   ✓ HTML report saved: {html_path}")


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """Save results to Excel and CSV."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    excel_path = output_path / 'find_epsilon_benchmark_results.xlsx'
    
    # Excel is optional (requires openpyxl)
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            pivot_oracle = results_df.pivot(index='Rule_ID', columns='Delta', values='Oracle_Calls')
            pivot_oracle.to_excel(writer, sheet_name='Oracle_Calls_by_Rule')
            
            pivot_runtime = results_df.pivot(index='Rule_ID', columns='Delta', values='Runtime_Seconds')
            pivot_runtime.to_excel(writer, sheet_name='Runtime_by_Rule')
            
            pivot_epsilon = results_df.pivot(index='Rule_ID', columns='Delta', values='Smallest_Epsilon_Homogeneous')
            pivot_epsilon.to_excel(writer, sheet_name='Found_Epsilon_by_Rule')
        
        print(f"   ✓ Results saved: {excel_path}")
    except ModuleNotFoundError:
        print("   ℹ️  openpyxl not installed; skipping Excel output (find_epsilon_benchmark_results.xlsx)")
    
    csv_path = output_path / 'find_epsilon_benchmark_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV saved: {csv_path}")
    
    # Generate HTML report
    generate_html_report(results_df, summary_df, output_dir)


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark Two-Phase Find Smallest Epsilon Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Automatic (recommended - Phase 1 starts from ε₀ = 1000)
  python benchmark_find_epsilon.py --rules 10 --deltas "300,500,1000"
  
  # Custom ε₀ for Phase 1
  python benchmark_find_epsilon.py --rules 5 --epsilon_0 5000 --epsilon_max 10000000
  
Algorithm:
  Phase 1: Find ε_max by testing ε₀, 2ε₀, 4ε₀, ... until homogeneous
  Phase 2: Binary search on [0, ε_max] to find smallest ε*
        """
    )
    parser.add_argument('--rules', type=int, default=5, 
                       help='Number of rules to test (default: 5)')
    parser.add_argument('--deltas', type=str, default='500,1000,1500,2000,2500,3000',
                       help='Comma-separated delta values (default: 500,1000,1500,2000,2500,3000)')
    parser.add_argument('--epsilon_0', type=float, default=None,
                       help='Initial ε₀ for Phase 1 exponential search (default: None = auto 1000)')
    parser.add_argument('--epsilon_max', type=float, default=1_000_000_000.0,
                       help='Maximum epsilon cap (default: 1 billion)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory (default: benchmark_results)')
    parser.add_argument('--dataset', type=str, default='so', choices=['so', 'acs'],
                       help='Dataset to benchmark (default: so)')
    
    args = parser.parse_args()
    
    delta_values = [int(x.strip()) for x in args.deltas.split(',')]
    
    # ── Dataset-specific configuration ──
    ds_kwargs = {}
    if args.dataset == 'acs':
        proj_root = Path(__file__).resolve().parent.parent
        ds_kwargs = dict(
            treatment_file=str(proj_root / "algorithms" / "ACSChosen10Treatments.json"),
            outcome_col="Wages or salary income past 12 months",
            base_dataset_path=str(proj_root / "acs" / "acs_encoded.csv"),
            delta_percentages=[5, 10, 15, 20],
        )
    
    print("\n🚀 Starting benchmark...\n")
    start_total = time.time()
    
    results_df = run_benchmark(
        num_rules=args.rules,
        delta_values=delta_values,
        epsilon_0=args.epsilon_0,
        epsilon_max_cap=args.epsilon_max,
        output_dir=args.output,
        **ds_kwargs
    )
    
    total_time = time.time() - start_total
    
    print("\n📊 Generating summary statistics...")
    summary_df = generate_summary_statistics(results_df)
    
    print("\n📈 Creating visualizations...")
    create_visualizations(results_df, args.output)
    
    print("\n💾 Saving results...")
    save_results(results_df, summary_df, args.output)
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\nTotal benchmark time: {total_time/60:.2f} minutes")
    print("="*80)
    print(f"\n✅ Benchmark complete! Results saved to: {args.output}/")
    print(f"   - 🌐 HTML Report: benchmark_report.html (open in browser!)")
    print(f"   - 📊 Excel: find_epsilon_benchmark_results.xlsx")
    print(f"   - 📄 CSV: find_epsilon_benchmark_results.csv")
    print(f"   - 📈 Plots: benchmark_visualization.png, oracle_calls_heatmap.png")
    print("="*80)


if __name__ == "__main__":
    main()

