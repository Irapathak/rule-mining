#!/usr/bin/env python3
"""
Phase 1: Rule space analysis and precomputation feasibility benchmark.

This script estimates whether exhaustive rule precomputation is feasible for
non-binary categorical data where each attribute has:
  - wildcard (attribute not used), or
  - exactly one selected value.

Total patterns = product_j (nunique(col_j) + 1)
"""

from __future__ import annotations

import argparse
import json
import time
from math import prod
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    return df


def full_rule_space_count(df: pd.DataFrame, outcome_col: str) -> int:
    features = [c for c in df.columns if c != outcome_col]
    return int(prod((df[c].nunique() + 1) for c in features))


def benchmark_dfs_pruning(
    df: pd.DataFrame,
    outcome_col: str,
    delta: int,
    benchmark_seconds: int,
) -> Dict[str, float]:
    # Search columns (exclude outcome). Order high-cardinality first for stronger early pruning.
    features = sorted([c for c in df.columns if c != outcome_col], key=lambda c: df[c].nunique(), reverse=True)
    values: Dict[str, List[int]] = {
        c: sorted(pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist()) for c in features
    }
    arrays = {c: pd.to_numeric(df[c], errors="coerce").to_numpy() for c in features}
    n = len(df)
    full_space = int(prod(len(values[c]) + 1 for c in features))

    # DFS state = (column_index, current_mask)
    stack = [(0, np.ones(n, dtype=bool))]
    visited_states = 0
    pruned_branches = 0

    t0 = time.time()
    while stack and (time.time() - t0) < benchmark_seconds:
        i, mask = stack.pop()
        visited_states += 1

        if i == len(features):
            continue

        # Wildcard branch.
        stack.append((i + 1, mask))

        # Value branches.
        col = features[i]
        arr = arrays[col]
        for v in values[col]:
            new_mask = mask & (arr == v)
            if int(new_mask.sum()) < delta:
                pruned_branches += 1
                continue
            stack.append((i + 1, new_mask))

    elapsed = max(time.time() - t0, 1e-9)
    throughput = visited_states / elapsed
    progress = visited_states / full_space if full_space else 0.0
    eta_sec = (full_space - visited_states) / throughput if throughput > 0 else float("inf")

    return {
        "rows": int(n),
        "num_features": int(len(features)),
        "full_space": int(full_space),
        "elapsed_sec": float(elapsed),
        "visited_states": int(visited_states),
        "pruned_branches": int(pruned_branches),
        "states_per_sec": float(throughput),
        "progress_fraction": float(progress),
        "eta_days_linear": float(eta_sec / 86400.0) if np.isfinite(eta_sec) else float("inf"),
        "eta_years_linear": float(eta_sec / (86400.0 * 365.25)) if np.isfinite(eta_sec) else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 feasibility benchmark for exhaustive rule precomputation.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV path.")
    parser.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    parser.add_argument("--delta", type=int, default=100)
    parser.add_argument("--benchmark-seconds", type=int, default=75)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    df = load_df(args.dataset)
    full_space = full_rule_space_count(df, args.outcome_col)
    stats = benchmark_dfs_pruning(df, args.outcome_col, args.delta, args.benchmark_seconds)
    stats["full_space_recomputed"] = int(full_space)

    print("Phase 1 Feasibility Summary")
    print(f"- Rows: {stats['rows']:,}")
    print(f"- Features: {stats['num_features']}")
    print(f"- Full rule space: {stats['full_space']:,}")
    print(f"- Visited states: {stats['visited_states']:,}")
    print(f"- Pruned branches: {stats['pruned_branches']:,}")
    print(f"- Throughput: {stats['states_per_sec']:,.1f} states/sec")
    print(f"- Progress: {stats['progress_fraction']:.3e}")
    print(f"- Linear ETA: {stats['eta_days_linear']:,.1f} days (~{stats['eta_years_linear']:,.1f} years)")

    out = args.output_json
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)
        print(f"- Saved JSON: {out}")


if __name__ == "__main__":
    main()
