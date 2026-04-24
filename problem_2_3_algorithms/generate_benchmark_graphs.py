import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Load data
problem2_csv = Path("benchmark_results_full/problem2_largest_delta/find_delta_benchmark_results.csv")
problem3_csv = Path("benchmark_results_full/problem3_smallest_epsilon/find_epsilon_benchmark_results.csv")

df_delta = pd.read_csv(problem2_csv)
df_epsilon = pd.read_csv(problem3_csv)

print(f"📊 Loaded {len(df_delta)} rows for Problem 2 (Finding Delta)")
print(f"📊 Loaded {len(df_epsilon)} rows for Problem 3 (Finding Epsilon)")

# Prepare rule labels
df_delta['Rule_Label'] = df_delta.apply(
    lambda x: f"Rule {x['Rule_ID']}: {x['Condition']} → {x['Treatment']}", axis=1
)
df_epsilon['Rule_Label'] = df_epsilon.apply(
    lambda x: f"Rule {x['Rule_ID']}: {x['Condition']} → {x['Treatment']}", axis=1
)

# Define color palette for rules
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
          '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']

# ================== GRAPH 1: Runtime vs Epsilon (Fixed Delta) ==================
print("\n📈 Creating Graph 1: Runtime vs Epsilon for each Delta...")

fig1 = go.Figure()

# Plot each rule with a different color
for rule_id, rule_group in df_delta.groupby('Rule_ID'):
    color_idx = (rule_id - 1) % len(colors)
    rule_label = rule_group['Rule_Label'].iloc[0]
    
    # Add delta information to hover text
    fig1.add_trace(go.Scatter(
        x=rule_group['Epsilon'],
        y=rule_group['Runtime_Seconds'],
        mode='lines+markers',
        name=rule_label,
        line=dict(width=3, color=colors[color_idx]),
        marker=dict(size=10),
        customdata=rule_group['Largest_Delta_Heterogeneous'],
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Epsilon: %{x:,.0f}<br>' +
                      'Found Delta: %{customdata}<br>' +
                      'Runtime: %{y:.2f}s<br>' +
                      '<extra></extra>'
    ))

# Get the search range info
delta_min = df_delta['Largest_Delta_Heterogeneous'].min()
delta_max = df_delta['Largest_Delta_Heterogeneous'].max()

fig1.update_layout(
    title=f"Problem 2: Runtime vs Epsilon (Finding Largest Delta)<br><sub>Searching for delta in range [{delta_min:,.0f}, {delta_max:,.0f}] for each epsilon</sub>",
    xaxis_title="Epsilon (Homogeneity Threshold) - Fixed for each test",
    yaxis_title="Runtime (seconds)",
    template="plotly_white",
    font=dict(size=14),
    height=600,
    hovermode='closest',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

# ================== GRAPH 2: Runtime vs Delta (Fixed Epsilon) ==================
print("📈 Creating Graph 2: Runtime vs Delta for each Epsilon...")

fig2 = go.Figure()

# Plot each rule with a different color
for rule_id, rule_group in df_epsilon.groupby('Rule_ID'):
    color_idx = (rule_id - 1) % len(colors)
    rule_label = rule_group['Rule_Label'].iloc[0]
    
    # Add epsilon information to hover text
    fig2.add_trace(go.Scatter(
        x=rule_group['Delta'],
        y=rule_group['Runtime_Seconds'],
        mode='lines+markers',
        name=rule_label,
        line=dict(width=3, color=colors[color_idx]),
        marker=dict(size=10),
        customdata=rule_group['Smallest_Epsilon_Homogeneous'],
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Delta (Fixed): %{x}<br>' +
                      'Found Epsilon: %{customdata:,.2f}<br>' +
                      'Runtime: %{y:.2f}s<br>' +
                      '<extra></extra>'
    ))

# Get the epsilon search range info
epsilon_min = df_epsilon['Smallest_Epsilon_Homogeneous'].min()
epsilon_max = df_epsilon['Smallest_Epsilon_Homogeneous'].max()

fig2.update_layout(
    title=f"Problem 3: Runtime vs Delta (Finding Smallest Epsilon)<br><sub>Searching for epsilon (found range: [{epsilon_min:,.0f}, {epsilon_max:,.0f}]) for each fixed delta</sub>",
    xaxis_title="Delta (Minimum Subgroup Size) - Fixed for each test",
    yaxis_title="Runtime (seconds)",
    template="plotly_white",
    font=dict(size=14),
    height=600,
    hovermode='closest',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

# ================== GRAPH 3: Combined View - Log Scale ==================
print("📈 Creating Graph 3: Combined view with log scale...")

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Runtime vs Epsilon (Problem 2)', 'Runtime vs Delta (Problem 3)'),
    horizontal_spacing=0.12
)

# Problem 2: Runtime vs Epsilon
for rule_id, rule_group in df_delta.groupby('Rule_ID'):
    color_idx = (rule_id - 1) % len(colors)
    rule_label = f"Rule {rule_id}"
    
    fig3.add_trace(go.Scatter(
        x=rule_group['Epsilon'],
        y=rule_group['Runtime_Seconds'],
        mode='lines+markers',
        name=rule_label,
        line=dict(width=2, color=colors[color_idx]),
        marker=dict(size=8),
        legendgroup=f"rule{rule_id}",
        showlegend=True
    ), row=1, col=1)

# Problem 3: Runtime vs Delta
for rule_id, rule_group in df_epsilon.groupby('Rule_ID'):
    color_idx = (rule_id - 1) % len(colors)
    
    fig3.add_trace(go.Scatter(
        x=rule_group['Delta'],
        y=rule_group['Runtime_Seconds'],
        mode='lines+markers',
        name=f"Rule {rule_id}",
        line=dict(width=2, color=colors[color_idx]),
        marker=dict(size=8),
        legendgroup=f"rule{rule_id}",
        showlegend=False
    ), row=1, col=2)

fig3.update_xaxes(title_text="Epsilon", row=1, col=1)
fig3.update_xaxes(title_text="Delta", row=1, col=2)
fig3.update_yaxes(title_text="Runtime (seconds)", type="log", row=1, col=1)
fig3.update_yaxes(title_text="Runtime (seconds)", type="log", row=1, col=2)

fig3.update_layout(
    title_text="Combined Runtime Analysis (Log Scale)<br><sub>Comparing both problems side by side</sub>",
    template="plotly_white",
    font=dict(size=13),
    height=550
)

# ================== Save Graphs ==================
output_dir = Path("benchmark_results_full")
output_dir.mkdir(exist_ok=True)

# Save as HTML
fig1.write_html(str(output_dir / "graph_runtime_vs_epsilon.html"))
fig2.write_html(str(output_dir / "graph_runtime_vs_delta.html"))
fig3.write_html(str(output_dir / "graph_combined_analysis.html"))

print(f"\n✅ Graphs saved to {output_dir}/")
print("   - graph_runtime_vs_epsilon.html")
print("   - graph_runtime_vs_delta.html")
print("   - graph_combined_analysis.html")

# Return figures for embedding
print("\n✨ Graphs generated successfully!")

