# Benchmark Scripts

## Quick Start

**Run both Problem 2 & 3:**
```bash
cd problem_2_3_algorithms
# Activate your venv (example):
# source ../../.venv-py311/bin/activate
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000" --deltas "500,1000,1500"
```

**Compare epsilon-finding methods:**
```bash
python benchmark_epsilon_comparison.py --rules 5 --deltas "500,1000,1500"
```

## Available Scripts

### 1. `run_all_benchmarks.py`
Runs Problem 2 (largest δ) and Problem 3 (smallest ε) together.
- **View results**: `benchmark_results/summary_report.html`

### 2. `benchmark_epsilon_comparison.py` 
Compares two methods for finding smallest ε:
- **Method 1**: Two-Phase Search (exponential + binary)
- **Method 2**: Brute Force (FPGrowth all subgroups)
- **View results**: `benchmark_results_epsilon_comparison/epsilon_comparison_report.html`

## What They Do

- **Problem 2**: Finds largest δ breaking homogeneity (fixed ε)
- **Problem 3**: Finds smallest ε achieving homogeneity (fixed δ)
  - Method 1: Efficient search with oracle calls
  - Method 2: Exhaustive enumeration (always finds answer)
- Generates HTML reports with violation details

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--rules N` | Number of rules to test (default: 5) |
| `--epsilons "a,b,c"` | Epsilon values for Problem 2 |
| `--deltas "x,y,z"` | Delta values for Problem 3 |
| `--delta_max N` | Max delta search range (default: 10000) |
| `--epsilon_max N` | Max epsilon search range (default: 500000) |

## Examples

**Quick test (~3 min):**
```bash
python run_all_benchmarks.py --rules 2 --epsilons "25000,50000" --deltas "1000,2000"
```

**Standard (~15 min):**
```bash
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000,40000,50000,60000" --deltas "500,1000,1500,2000,2500,3000"
```

## Output

```
benchmark_results/
├── summary_report.html           ← Open this first
├── problem2_largest_delta/
│   └── benchmark_report.html
└── problem3_smallest_epsilon/
    └── benchmark_report.html
```

## Troubleshooting

- **ModuleNotFoundError**: Activate venv first
- **Problem 3 "Not found"**: Increase `--epsilon_max 1000000`
- **Too slow**: Reduce `--rules` or parameter count
