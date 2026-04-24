# Homogeneity project (organized repo)

Self-contained repo: clone and run from the repo root. All paths are relative to the repo so nothing outside the clone is required for the default (ACS) dataset.

## Quick start (clone and run)

```bash
git clone <this-repo>
cd homogeneity_repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS (required for XGBoost):** install the OpenMP runtime once:

```bash
brew install libomp
```

Then run the main script from the repo root:

```bash
python algorithms/all_subgroups_loop.py
```

## Layout

- **configs/** – dataset and run configuration (paths are inside the repo)
- **yarden_files/** – ATE and linear model utilities
- **algorithms/** – main loop, FPGrowth/RW, scalability and ablation scripts
- **problem_2_3_algorithms/** – benchmarks for largest δ and smallest ε
- **article_figures/** – notebook to generate figures from result CSVs
- **ablation_results/** – ablation result CSVs and reports
- **acs/** – sample ACS dataset (included); other datasets go under **data/** (see config)
