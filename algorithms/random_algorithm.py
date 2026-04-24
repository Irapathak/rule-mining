"""
* **ALGORITHM: Random Baseline (Two-Phase)**
* **Goal:** A baseline to compare against Apriori-RW.
* **Phase 1 (Generation):** Randomly selects N unique, valid subgroups (size > delta) using Validity-Aware Forward Sampling.
* **Phase 2 (Evaluation):** Iteratively calculates CATE for the selected set. Stops immediately on violation.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    """
    Creates One-Hot encoding and a lookup dictionary.
    """
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫").astype(str), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup


def _random_baseline_algo(
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        delta: int,
        epsilon: float,
        ate_all: float,
        n_subgroups: int = 1000,
        rng: Optional[random.Random] = None,
) -> Tuple[bool, int]:
    rng = rng or random.Random()

    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    # 1. One-Hot Encoding & Pre-processing
    onehot, lookup = _onehot_lookup(mining_df)

    # --- OPTIMIZATION SETUP ---
    matrix = onehot.values  # shape: (n_samples, n_features)
    col_names = np.array(onehot.columns)

    # Pre-filter valid single columns
    col_supports = matrix.sum(axis=0)
    valid_indices = np.where(col_supports >= delta)[0]

    if len(valid_indices) == 0:
        print(f"No single column satisfies delta={delta}. Exiting.")
        return True, 0

    # --- PHASE 1: GENERATION (Validity-Aware Forward Sampling) ---
    print(f"--- Phase 1: Generating {n_subgroups} random valid subgroups ---")

    generated_subgroups: List[Tuple[frozenset, np.ndarray]] = []
    visited_fingerprints: Set[frozenset] = set()

    attempts = 0
    # Higher safety limit because deep sampling is more computationally intensive
    max_attempts = n_subgroups * 500

    while len(generated_subgroups) < n_subgroups and attempts < max_attempts:
        attempts += 1

        # A. Pick a random valid root
        start_idx = rng.choice(valid_indices)
        current_mask = matrix[:, start_idx]
        current_indices = {start_idx}

        # B. Randomly Deepen (Walk the Lattice)
        # We don't just guess one column. We sample a BATCH of candidates
        # and see which ones are valid steps. This allows us to walk deeper.
        target_depth = rng.randint(1, 4)

        while len(current_indices) < target_depth:
            # 1. Sample a batch of potential neighbors (e.g., 10 random cols)
            # Sampling a small batch is O(1). Checking all is O(N).
            # This balances speed and exploration.
            candidates = []
            for _ in range(10):
                c = rng.choice(valid_indices)
                if c not in current_indices:
                    candidates.append(c)

            if not candidates:
                break  # No valid candidates found (rare)

            # 2. Check validity of candidates (Lookahead)
            valid_next_steps = []
            for cand in candidates:
                # Fast NumPy Intersection
                temp_mask = current_mask & matrix[:, cand]
                if temp_mask.sum() >= delta:
                    valid_next_steps.append((cand, temp_mask))

            # 3. Transition
            if not valid_next_steps:
                # Dead end in this direction
                break

            # Pick one valid step randomly
            chosen_idx, chosen_mask = rng.choice(valid_next_steps)
            current_indices.add(chosen_idx)
            current_mask = chosen_mask

        # C. Store if Unique
        current_itemset = frozenset(col_names[i] for i in current_indices)

        if current_itemset in visited_fingerprints:
            continue

        visited_fingerprints.add(current_itemset)
        generated_subgroups.append((current_itemset, current_mask))

    print(f"Generation Complete. Collected {len(generated_subgroups)} subgroups.")

    # --- PHASE 2: EVALUATION ---
    # Now we iterate the list and check for violations.
    print(f"--- Phase 2: Checking CATE for violations ---")

    checked_count = 0

    for itemset, mask in generated_subgroups:
        checked_count += 1

        # Calculate CATE
        sub_df = df[mask]
        try:
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta)
        except LinAlgError:
            continue

        # Check Violation
        if abs(cate - ate_all) > epsilon:
            pretty_dict = {lookup[c][0]: lookup[c][1] for c in itemset}
            print(f"Breaking Subgroup Found (Random): {pretty_dict}")
            print(f"Total unique subgroups checked: {checked_count}")
            return False, checked_count

    print(f"Finished Random Baseline. No violations found in {checked_count} checks.")
    return True, checked_count


# Public API
def calc_utility_for_subgroups(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
        *,
        outcome_col: Optional[str] = None,
        tgtO: Optional[str] = None,
        n_subgroups: int = 1_000,
        rng: Optional[random.Random] = None,
        **kwargs: object,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col / tgtO")

    # Compatibility: Check if caller passed 'k_walks' (legacy) or 'max_checks'
    n = kwargs.get('max_checks', kwargs.get('k_walks', n_subgroups))


    if mode == 0:
        return _random_baseline_algo(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            ate_all=utility_all,
            n_subgroups=n,
            rng=rng,
        )

    return [], 0