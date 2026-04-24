# Ablation Study Results

This folder contains results from the ablation study comparing FPGrowth and RW_Direct algorithms.

## How to Run

```bash
cd algorithms
python ablation_study.py
```

This single script will automatically:
- Test FPGrowth and RW_Direct algorithms
- Run Experiment 1: Varying Epsilon (250k, 300k, 350k, 400k, 450k) with fixed delta=10%
- Run Experiment 2: Varying Delta (5%, 10%, 15%, 20%) with fixed epsilon=350k
- Save raw results to Excel file
- Create summary CSV with aggregated statistics
- Generate interactive HTML report with graphs

## Output Files

1. **`{dataset}_ablation_raw_results.xlsx`** - All individual runs with detailed results
2. **`{dataset}_ablation_summary.csv`** - Aggregated statistics by algorithm/parameter
3. **`{dataset}_ablation_report.html`** - Interactive HTML report with graphs

## Experiments

### Experiment 1: Epsilon Sensitivity
- **Fixed:** Delta = 10% of dataset
- **Varying:** Epsilon = [250k, 300k, 350k, 400k, 450k]
- **Question:** How does homogeneity threshold affect results?

### Experiment 2: Delta Sensitivity  
- **Fixed:** Epsilon = 350k
- **Varying:** Delta = [5%, 10%, 15%, 20%] of dataset
- **Question:** How does minimum subgroup size affect results?

## Metrics Tracked

- Homogeneity rate (% of rules satisfying homogeneity)
- Number of subgroups checked
- Runtime (seconds)
- Dataset size and delta percentage

