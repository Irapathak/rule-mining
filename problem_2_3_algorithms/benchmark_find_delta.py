"""
Benchmark for Find Largest Delta Algorithm.

Tests the binary search algorithm across multiple rules (treatment-condition pairs)
and multiple epsilon values to evaluate efficiency and performance.

Outputs:
- Excel file with detailed results table
- Summary statistics
- Visualization plots
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import re

# seaborn is optional; fall back to matplotlib-only plots if it's not installed.
try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sns = None

# Add project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from find_largest_delta import find_largest_delta_breaking_homogeneity

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
    epsilon_values: List[float] = None,
    delta_min: int = 100,
    delta_max: int = 10000,
    output_dir: str = "benchmark_results",
    # ── Dataset-specific parameters ──
    treatment_file: str = None,
    outcome_col: str = 'ConvertedSalary',
    base_dataset_path: str = None,
    delta_min_pct: float = None,
    delta_max_pct: float = None,
) -> pd.DataFrame:
    """
    Run comprehensive benchmark across multiple rules and epsilon values.
    
    Args:
        num_rules: Number of rules to test
        epsilon_values: List of epsilon values to test
        delta_min: Minimum delta for search (absolute, used for SO)
        delta_max: Maximum delta for search (absolute, used for SO)
        output_dir: Directory to save results
        treatment_file: Path to treatments JSON (None → default SO file)
        outcome_col: Name of the outcome column in the dataset
        base_dataset_path: Path to the encoded base CSV for on-the-fly prep.
                           If None, pre-computed per-rule SO CSVs are used.
        delta_min_pct: Override *delta_min* with a % of the per-rule dataset.
        delta_max_pct: Override *delta_max* with a % of the per-rule dataset.
        
    Returns:
        DataFrame with benchmark results
    """
    if epsilon_values is None:
        epsilon_values = [10000, 20000, 30000, 40000, 50000, 60000]
    
    # Create output directory
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("BENCHMARK: Find Largest Delta Algorithm")
    print("="*80)
    print(f"Testing {num_rules} rules with {len(epsilon_values)} epsilon values each")
    print(f"Total experiments: {num_rules * len(epsilon_values)}")
    print(f"Epsilon values: {epsilon_values}")
    if delta_min_pct is not None:
        print(f"Delta range: [{delta_min_pct}% .. {delta_max_pct}%] of per-rule dataset")
    else:
        print(f"Delta range: [{delta_min}, {delta_max}]")
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
    total_experiments = num_rules * len(epsilon_values)
    current_experiment = 0
    
    # Run benchmark
    for rule_idx, treatment_data in enumerate(treatments, 1):
        condition = treatment_data['condition']
        treatment = treatment_data['treatment']
        
        # ── Prepare dataset for this rule ──
        if base_df is not None:
            # On-the-fly preparation (ACS / generic path)
            df, actual_outcome_col = _prepare_rule_df(base_df, treatment_data, outcome_col)
            if df is None:
                print(f"\n⚠️  Warning: Rule {rule_idx} produced empty dataset. Skipping.")
                continue
        else:
            # Pre-computed per-rule CSVs (SO path)
            dataset_path = Path(f'../stackoverflow/processed_db/so_countries_treatment_{rule_idx}_encoded.csv')
            if not dataset_path.exists():
                print(f"\n⚠️  Warning: Dataset for rule {rule_idx} not found. Skipping.")
                continue
            df = pd.read_csv(dataset_path)
            actual_outcome_col = outcome_col
        
        # ── Per-rule delta bounds ──
        rule_delta_min = max(1, int(len(df) * delta_min_pct / 100)) if delta_min_pct is not None else delta_min
        rule_delta_max = int(len(df) * delta_max_pct / 100) if delta_max_pct is not None else delta_max
        
        print(f"\n{'='*80}")
        print(f"RULE {rule_idx}/{num_rules}")
        print(f"Condition: {condition}")
        print(f"Treatment: {treatment}")
        print(f"Dataset size: {len(df)} rows")
        print(f"Delta range for this rule: [{rule_delta_min}, {rule_delta_max}]")
        print(f"{'='*80}")
        
        for epsilon in epsilon_values:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Testing epsilon = {epsilon:,}")
            
            # Run algorithm and measure time
            start_time = time.time()
            
            largest_delta, oracle_calls, violation_info, utility_all = find_largest_delta_breaking_homogeneity(
                df=df,
                treatment_col=TREATMENT_COL,
                outcome_col=actual_outcome_col,
                epsilon=epsilon,
                delta_min=rule_delta_min,
                delta_max=rule_delta_max,
                verbose=False  # Suppress detailed output for benchmark
            )
            
            elapsed_time = time.time() - start_time
            
            # Calculate theoretical maximum iterations (log2 of range)
            import math
            theoretical_max = math.ceil(math.log2(delta_max - delta_min + 1))
            efficiency_ratio = oracle_calls / theoretical_max if theoretical_max > 0 else 1.0
            
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
                'Epsilon': epsilon,
                'Largest_Delta_Heterogeneous': largest_delta if largest_delta is not None else 'None',
                'Oracle_Calls': oracle_calls,
                'Runtime_Seconds': round(elapsed_time, 3),
                'Runtime_Minutes': round(elapsed_time / 60, 3),
                'Theoretical_Max_Iterations': theoretical_max,
                'Efficiency_Ratio': round(efficiency_ratio, 3),
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
            delta_str = str(largest_delta) if largest_delta is not None else "No violation found"
            print(f"   ✓ Result: δ* = {delta_str}")
            print(f"   ✓ Oracle calls: {oracle_calls} (theoretical max: {theoretical_max})")
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
            'Avg Efficiency Ratio',
            'Total Runtime (minutes)'
        ],
        'Value': [
            len(results_df),
            results_df['Runtime_Seconds'].mean(),
            results_df['Oracle_Calls'].mean(),
            results_df['Oracle_Calls'].min(),
            results_df['Oracle_Calls'].max(),
            results_df['Efficiency_Ratio'].mean(),
            results_df['Runtime_Seconds'].sum() / 60
        ]
    })
    summary['Value'] = summary['Value'].round(3)
    return summary


def create_visualizations(results_df: pd.DataFrame, output_dir: str = "benchmark_results"):
    """Create visualization plots for benchmark results."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    
    # Set style
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Oracle Calls vs Epsilon (grouped by rule)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Oracle Calls vs Epsilon
    ax1 = axes[0, 0]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        ax1.plot(rule_data['Epsilon'], rule_data['Oracle_Calls'], 
                marker='o', label=f'Rule {rule_id}', linewidth=2)
    ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Oracle Calls (Iterations)', fontsize=12, fontweight='bold')
    ax1.set_title('Binary Search Efficiency: Oracle Calls vs Epsilon', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs Epsilon
    ax2 = axes[0, 1]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        ax2.plot(rule_data['Epsilon'], rule_data['Runtime_Seconds'], 
                marker='s', label=f'Rule {rule_id}', linewidth=2)
    ax2.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Runtime vs Epsilon', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Largest Delta Found vs Epsilon
    ax3 = axes[1, 0]
    for rule_id in results_df['Rule_ID'].unique():
        rule_data = results_df[results_df['Rule_ID'] == rule_id]
        # Filter out None values
        valid_data = rule_data[rule_data['Largest_Delta_Heterogeneous'] != 'None'].copy()
        if not valid_data.empty:
            valid_data['Largest_Delta_Heterogeneous'] = pd.to_numeric(valid_data['Largest_Delta_Heterogeneous'])
            ax3.plot(valid_data['Epsilon'], valid_data['Largest_Delta_Heterogeneous'], 
                    marker='^', label=f'Rule {rule_id}', linewidth=2)
    ax3.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Largest δ with Violation', fontsize=12, fontweight='bold')
    ax3.set_title('Found δ* (Heterogeneous) vs Epsilon', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Ratio Distribution
    ax4 = axes[1, 1]
    ax4.hist(results_df['Efficiency_Ratio'], bins=15, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax4.axvline(results_df['Efficiency_Ratio'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {results_df["Efficiency_Ratio"].mean():.2f}')
    ax4.set_xlabel('Efficiency Ratio (Actual / Theoretical)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Binary Search Efficiency Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_path / 'benchmark_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Visualization saved: {plot_path}")
    plt.close()
    
    # 2. Heatmap: Oracle Calls by Rule and Epsilon
    pivot_oracle = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Oracle_Calls')
    
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
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{int(val)}", ha='center', va='center', fontsize=8, color='black')
    plt.title('Oracle Calls Heatmap: Rule vs Epsilon', fontsize=14, fontweight='bold')
    plt.xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    plt.ylabel('Rule ID', fontsize=12, fontweight='bold')
    plt.tight_layout()
    heatmap_path = output_path / 'oracle_calls_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Heatmap saved: {heatmap_path}")
    plt.close()


def generate_html_report(results_df: pd.DataFrame, summary_df: pd.DataFrame,
                         output_dir: str = "benchmark_results"):
    """Generate a beautiful HTML report with interactive tables."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    html_path = output_path / 'benchmark_report.html'
    
    # HTML template with modern styling
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Find Largest Delta - Benchmark Results</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
            .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
            .content {{ padding: 40px; }}
            h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin: 30px 0 20px 0;
            }}
            .table-scroll {{
                width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                margin: 20px 0;
            }}
            .table-scroll table {{
                width: max-content;
                min-width: 100%;
                margin: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: visible;
            }}
            thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            th {{
                padding: 15px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 0.5px;
            }}
            tbody tr {{ transition: background 0.3s; }}
            tbody tr:nth-child(even) {{ background: #f8f9fa; }}
            tbody tr:hover {{
                background: #e3f2fd;
                /* Avoid clipping wide tables on hover */
                transform: none;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #dee2e6;
            }}
            .metric-value {{
                font-weight: bold;
                color: #667eea;
                font-size: 1.1em;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
            }}
            .badge-success {{
                background: #d4edda;
                color: #155724;
            }}
            .badge-info {{
                background: #d1ecf1;
                color: #0c5460;
            }}
            .badge-warning {{
                background: #fff3cd;
                color: #856404;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .summary-card h3 {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .summary-card .value {{
                font-size: 2.5em;
                font-weight: bold;
            }}
            .note {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .note strong {{ color: #856404; }}
            footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #6c757d;
                font-size: 0.9em;
            }}
            .highlight {{ background: #ffe066; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔍 Find Largest Delta - Benchmark Results</h1>
                <p class="subtitle">Binary Search Algorithm Performance Analysis</p>
            </header>
            
            <div class="content">
                <div class="note">
                    <strong>📌 Key Insight:</strong> If δ* = <span class="highlight">largest heterogeneous</span>, 
                    then δ* + 1 = <span class="highlight">smallest homogeneous</span>. 
                    Below δ*, violations exist. At δ* + 1 and above, the rule becomes homogeneous.
                </div>
                
                <h2>📊 Summary Statistics</h2>
                <div class="summary-grid">
                    {summary_cards}
                </div>
                
                <h2>📋 Detailed Results</h2>
                {detailed_table}
                
                <h2>📈 Analysis by Rule</h2>
                {pivot_tables}
            </div>
            
            <footer>
                <p>Generated on {timestamp}</p>
                <p>Algorithm: Binary Search with FPGrowth Oracle</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Generate summary cards HTML
    summary_cards_html = ""
    for _, row in summary_df.iterrows():
        summary_cards_html += f"""
        <div class="summary-card">
            <h3>{row['Metric']}</h3>
            <div class="value">{row['Value']}</div>
        </div>
        """
    
    # Generate detailed table HTML (wrapped for horizontal scrolling)
    detailed_html = "<div class='table-scroll'><table><thead><tr>"
    for col in results_df.columns:
        detailed_html += f"<th>{col.replace('_', ' ')}</th>"
    detailed_html += "</tr></thead><tbody>"
    
    for _, row in results_df.iterrows():
        detailed_html += "<tr>"
        for col in results_df.columns:
            value = row[col]
            if col == 'Largest_Delta_Heterogeneous':
                badge_class = 'badge-warning' if value != 'None' else 'badge-success'
                detailed_html += f'<td><span class="badge {badge_class}">{value}</span></td>'
            elif col == 'Oracle_Calls':
                detailed_html += f'<td class="metric-value">{value}</td>'
            elif col in ['Runtime_Seconds', 'Runtime_Minutes']:
                detailed_html += f'<td><span class="badge badge-info">{value}</span></td>'
            else:
                detailed_html += f"<td>{value}</td>"
        detailed_html += "</tr>"
    detailed_html += "</tbody></table></div>"
    
    # Generate pivot tables
    pivot_html = "<h3>Oracle Calls by Rule × Epsilon</h3>"
    pivot_oracle = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Oracle_Calls')
    pivot_html += "<div class='table-scroll'>" + pivot_oracle.to_html(classes='', border=0) + "</div>"
    
    pivot_html += "<h3 style='margin-top: 30px;'>Runtime (seconds) by Rule × Epsilon</h3>"
    pivot_runtime = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Runtime_Seconds')
    pivot_html += "<div class='table-scroll'>" + pivot_runtime.to_html(classes='', border=0) + "</div>"
    
    # Fill template
    import datetime
    html_content = html_template.format(
        summary_cards=summary_cards_html,
        detailed_table=detailed_html,
        pivot_tables=pivot_html,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✓ HTML report saved: {html_path}")


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                output_dir: str = "../benchmark_results"):
    """Save results to Excel file with multiple sheets."""
    output_path = Path(output_dir)
    excel_path = output_path / 'find_delta_benchmark_results.xlsx'
    
    # Excel is optional (requires openpyxl)
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Summary statistics
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Pivot tables
            pivot_oracle = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Oracle_Calls')
            pivot_oracle.to_excel(writer, sheet_name='Oracle_Calls_by_Rule')
            
            pivot_runtime = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Runtime_Seconds')
            pivot_runtime.to_excel(writer, sheet_name='Runtime_by_Rule')
            
            pivot_delta = results_df.pivot(index='Rule_ID', columns='Epsilon', values='Largest_Delta_Heterogeneous')
            pivot_delta.to_excel(writer, sheet_name='Found_Delta_by_Rule')
        
        print(f"   ✓ Results saved: {excel_path}")
    except ModuleNotFoundError:
        print("   ℹ️  openpyxl not installed; skipping Excel output (find_delta_benchmark_results.xlsx)")
    
    # Also save as CSV for easy viewing
    csv_path = output_path / 'find_delta_benchmark_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV saved: {csv_path}")
    
    # Generate HTML report
    generate_html_report(results_df, summary_df, output_dir)


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Find Largest Delta algorithm')
    parser.add_argument('--rules', type=int, default=5, 
                       help='Number of rules to test (default: 5)')
    parser.add_argument('--epsilons', type=str, default='10000,20000,30000,40000,50000,60000',
                       help='Comma-separated epsilon values (default: 10k,20k,30k,40k,50k,60k)')
    parser.add_argument('--delta_min', type=int, default=100, help='Minimum delta (default: 100)')
    parser.add_argument('--delta_max', type=int, default=10000, help='Maximum delta (default: 10000)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory (default: benchmark_results)')
    parser.add_argument('--dataset', type=str, default='so', choices=['so', 'acs'],
                       help='Dataset to benchmark (default: so)')
    
    args = parser.parse_args()
    
    # Parse epsilon values
    epsilon_values = [float(x.strip()) for x in args.epsilons.split(',')]
    
    # ── Dataset-specific configuration ──
    ds_kwargs = {}
    if args.dataset == 'acs':
        proj_root = Path(__file__).resolve().parent.parent
        ds_kwargs = dict(
            treatment_file=str(proj_root / "algorithms" / "ACSChosen10Treatments.json"),
            outcome_col="Wages or salary income past 12 months",
            base_dataset_path=str(proj_root / "acs" / "acs_encoded.csv"),
            delta_min_pct=5.0,
            delta_max_pct=20.0,
        )
    
    # Run benchmark
    print("\n🚀 Starting benchmark...\n")
    start_total = time.time()
    
    results_df = run_benchmark(
        num_rules=args.rules,
        epsilon_values=epsilon_values,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        output_dir=args.output,
        **ds_kwargs
    )
    
    total_time = time.time() - start_total
    
    # Generate summary
    print("\n📊 Generating summary statistics...")
    summary_df = generate_summary_statistics(results_df)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(results_df, args.output)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(results_df, summary_df, args.output)
    
    # Print summary to console
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print(f"\nTotal benchmark time: {total_time/60:.2f} minutes")
    print("="*80)
    print(f"\n✅ Benchmark complete! Results saved to: {args.output}/")
    print(f"   - 🌐 HTML Report: benchmark_report.html (open in browser!)")
    print(f"   - 📊 Excel: find_delta_benchmark_results.xlsx")
    print(f"   - 📄 CSV: find_delta_benchmark_results.csv")
    print(f"   - 📈 Plots: benchmark_visualization.png, oracle_calls_heatmap.png")
    print("="*80)


if __name__ == "__main__":
    main()

