# Core experiment scripts

Minimal entry points for the three homogeneity-selection approaches.  
Shared helpers live in `common.py`. Algorithm code stays in `algorithms/` and `run_fixed_intervention_pipeline.py`.

**Input:** `algorithms_results/random_rw_pruned_candidates.csv` (3,708 rules with precomputed `utility` / `score` from Phase 1).

## 1. Brute force (exhaustive homogeneity)

```bash
cd experiments/core
python brute_force.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --rank-by score \
  --out-csv ../../algorithms_results/brute_force_top10.csv \
  --out-meta ../../algorithms_results/brute_force_top10.meta.json
```

## 2. Greedy (sliding window + RW)

Run brute force first (or point `--target-csv` at an existing top-10 CSV).

```bash
python greedy.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --target-csv ../../algorithms_results/brute_force_top10.csv \
  --rank-by score \
  --rw-per-rule 1 \
  --max-rw-walks-per-rule 100 \
  --on-eviction continue_next_slot \
  --out-meta ../../algorithms_results/greedy.meta.json
```

## 3. ε-greedy

```bash
python eps_greedy.py \
  --candidates-csv ../../algorithms_results/random_rw_pruned_candidates.csv \
  --iterations 5000 \
  --epsilon-arm 0.8 \
  --rw-batch 100 \
  --out-topk ../../algorithms_results/eps_greedy_top10.csv \
  --out-meta ../../algorithms_results/eps_greedy.meta.json
```

## Defaults

| Flag | Default |
|------|---------|
| `--delta` | 5000 |
| `--epsilon-hom` | 2000 |
| `--rank-by` | score (brute force & greedy) |
| Greedy `--on-eviction` | `continue_next_slot` |
| Greedy RW cap | 100 walks per rule |

Each script writes **`runtime_sec_total`** and top-k **utility/coverage** stats to `.meta.json`.

## Legacy scripts

Older, fuller-featured versions remain under `experiments/` (`brute_force_updated.py`, `greedy_rule_mining.py`, `eps_greedy_pruned_original_rw.py`, etc.) for traces and one-off experiments. Use **`experiments/core/`** for clean comparison runs.
