# Stack Overflow homogeneity pipeline

End-to-end flow: **Phase 1 mining** → **RW prune** → **brute force / greedy / ε-greedy**.

Requires: Python 3.10+, `pip install -r requirements.txt`, encoded dataset at  
`algorithms/code/code/stackoverflow_data_encoded.csv` (included in repo).

Outputs go to `algorithms_results/` (gitignored).

## Step 0 — Config

`configs/config.json` — dataset key `stackoverflow`, treatment column, attribute weights for RW.

## Step 1 — Phase 1 (utility + score for all mined rules)

```bash
python experiments/phase1_fixed_intervention_scored.py \
  --out-csv algorithms_results/dataset_rule_mining.csv \
  --out-meta algorithms_results/dataset_rule_mining.meta.json
```

~10k rules with `utility`, `score`, `psi_grp_json`, `coverage` (~1 min).

## Step 2 — Preprocessing (random RW prune)

```bash
python experiments/random_rw_prune_candidates.py \
  --phase1-csv algorithms_results/dataset_rule_mining.csv \
  --iterations 10000 \
  --out-candidates algorithms_results/random_rw_pruned_candidates.csv \
  --out-meta algorithms_results/random_rw_prune.meta.json
```

Reduces pool (~10k → ~3.7k surviving candidates).

## Step 3 — Experiments (core scripts)

From `experiments/core/`:

### Brute force (exhaustive homogeneity, ground truth top-10)

```bash
cd experiments/core
python brute_force.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --rank-by score \
  --out-csv ../../algorithms_results/brute_force_top10.csv \
  --out-meta ../../algorithms_results/brute_force_top10.meta.json
```

### Greedy (sliding window + RW; needs brute-force top-10 as target)

```bash
python greedy.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --target-csv ../../algorithms_results/brute_force_top10.csv \
  --rank-by score \
  --on-eviction continue_next_slot
```

Defaults: 1 RW/sweep, 100 RW cap per rule.

### ε-greedy

```bash
python eps_greedy.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --iterations 5000 \
  --epsilon-arm 0.8
```

## Key modules

| Path | Role |
|------|------|
| `experiments/phase1_fixed_intervention_scored.py` | Phase 1 lattice mining |
| `experiments/random_rw_prune_candidates.py` | RW preprocessing prune |
| `experiments/core/*.py` | Three comparison experiments |
| `experiments/run_fixed_intervention_pipeline.py` | Data load, masks, `run_rw_once` |
| `experiments/phase_rw_config.py` | RW attribute weights |
| `algorithms/bruteForce_algorithm.py` | Exhaustive homogeneity check |
| `algorithms/rw_unlearning.py` | Random-walk homogeneity |
| `yarden_files/ATE_update.py` | CATE / utility |
