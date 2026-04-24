"""
Core subgroup analysis algorithms using Apriori.
This module contains functions for finding subgroups and calculating their utility.
"""
import sys
import json
import pandas as pd
import random
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
from ATE_update import calculate_ate_safe
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from numpy.linalg import LinAlgError

# Load config safely (repo-root relative so clone-and-run works)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
try:
    with open(_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    BINARY_TREATMENT = config.get('TREATMENT_COL', 'TempTreatment')
except (FileNotFoundError, KeyError):
    BINARY_TREATMENT = 'TempTreatment'


def mine_subgroups(
    algorithm: Callable,
    df: pd.DataFrame,
    delta: int,
    exclude_cols: List[str] = None
) -> List[Tuple[Dict[str, object], int]]:
    """
    Return [(filter-dict, size), …] for every subgroup size ≥ delta.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Filter out columns that should not be used for subgroup mining
    mining_df = df.drop(columns=exclude_cols, errors='ignore')

    # one‑hot encode attribute=value pairs
    onehot_parts = []
    lookup: Dict[str, Tuple[str, object]] = {}

    # FIX: Use a separator ('|') that is unlikely to be in column names.
    # The default '_' causes issues with columns like "employment_duration".
    sep = "|"

    for col in mining_df.columns:
        # Use the safe separator
        d = pd.get_dummies(mining_df[col].fillna('⧫NA⧫'), prefix=col, prefix_sep=sep, dtype=bool)
        onehot_parts.append(d)

        # Build lookup table using the safe split
        for c in d.columns:
            # Split exactly once on the separator
            parts = c.split(sep, 1)
            if len(parts) == 2:
                original_col, original_val = parts[0], parts[1]
                lookup[c] = (original_col, original_val)
            else:
                # Fallback safety (should not happen)
                lookup[c] = (col, c)

    onehot = pd.concat(onehot_parts, axis=1)

    # Apriori algorithm for frequent itemsets
    min_sup = delta / len(df)
    freq = algorithm(onehot, min_support=min_sup, use_colnames=True)

    if freq.empty:
        return []

    # discard item‑sets that mention the same attribute twice
    def valid(itemset):
        attrs = [lookup[col][0] for col in itemset]
        return len(attrs) == len(set(attrs))
    freq = freq[freq['itemsets'].apply(valid)]

    # convert back to {attr:value} + absolute size
    results: List[Tuple[Dict[str, object], int]] = []
    n_rows = len(df)
    for items, sup in zip(freq['itemsets'], freq['support']):
        fdict = {lookup[c][0]: lookup[c][1] for c in items}
        results.append((fdict, int(round(sup * n_rows))))
    return results


def filter_by_attribute(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Vectorised AND-filter without early-exit.
    """
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)

    for a, v in filters.items():
        if a not in df.columns:
            continue

        # FIX: Robust check to avoid 'invalid literal for int()'
        if pd.api.types.is_numeric_dtype(df[a]):
            try:
                # Try converting the string value (from lookup) to an int
                val = int(float(v))
            except (ValueError, TypeError):
                # If conversion fails, use the raw value
                val = v
            mask &= (df[a] == val)
        else:
            # For non-numeric columns, compare as-is
            mask &= (df[a] == v)

    return df[mask]


def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: float,
    utility_all: float
) -> Union[Tuple[bool, int, float, float], Tuple[List[dict], int, float, float]]:
    """
    Calculate utility for subgroups using Apriori.

    Returns:
        If mode != 1: (is_homogeneous, count_checked, enum_time, iter_time)
        If mode == 1: (subgroup_records, count_checked, enum_time, iter_time)
    """
    exclude_cols = [treatment_col, BINARY_TREATMENT, tgtO]
    
    # Exclude index columns (like "Unnamed: 0") that have one unique value per row
    # These create massive one-hot matrices and shouldn't be used for subgroup mining.
    # Note: The ablation study also excludes these (see ablation_study.py line 220).
    # Subgroups from such columns would be size 1, which never meets delta threshold.
    for col in df.columns:
        if col.startswith('Unnamed:') or df[col].nunique() == len(df):
            exclude_cols.append(col)

    # --- PHASE 1: ENUMERATION (Mining) ---
    start_enum = time.time()  # <--- Start Timer
    all_subgroups = mine_subgroups(algorithm, df, delta, exclude_cols=exclude_cols)
    end_enum = time.time()    # <--- End Timer

    enumeration_time = end_enum - start_enum

    # --- PHASE 2: ITERATION (Search) ---
    # Shuffle to ensure random search order
    random.shuffle(all_subgroups)

    subgroup_records = []
    cate_calc_count = 0
    is_homogeneous = True

    start_iter = time.time()

    for filt, sz in all_subgroups:
        sub_df = filter_by_attribute(df, filt)

        # Skip if empty
        if sub_df.empty:
            continue

        try:
            cate = calculate_ate_safe(sub_df, treatment_col, tgtO, delta)

            # Additional safety: check if ATE is NaN (sample size too small, etc.)
            if pd.isna(cate):
                continue

            cate_calc_count += 1
        except (LinAlgError, ValueError):
            continue

        # Check for violation (Mode 0/2/3)
        if mode != 1:
            if abs(utility_all - cate) > epsilon:
                print(f"breaking subgroup = {filt} size {sz} cate {cate}")
                print(f"Total unique subgroups checked: {cate_calc_count}")

                # Stop timer immediately upon finding violation
                iteration_time = time.time() - start_iter
                
                # Return violation info as dict
                violation_info = {
                    "subgroup": str(filt),
                    "size": sz,
                    "utility": cate,
                    "utility_diff": cate - utility_all,
                    "abs_diff": abs(utility_all - cate)
                }
                return False, cate_calc_count, enumeration_time, iteration_time, violation_info
        else:
            # Mode != 0: Collect all records
            abs_diff = abs(utility_all - cate)
            subgroup_records.append({
                "AttributeValues": str(filt),
                "Size": sz,
                "Utility": cate,
                "UtilityDiff": cate - utility_all,
                "AbsDiff": abs_diff,
            })

    # End timer if loop finishes without returning
    iteration_time = time.time() - start_iter

    if mode != 1:
        # Returns: (Passed?, Count, Enum Time, Iter Time, Violation Info)
        print(f"Total unique subgroups checked: {cate_calc_count}")
        return True, cate_calc_count, enumeration_time, iteration_time, None

    # Find max utility difference for mode != 0
    max_abs_diff = 0
    max_violation_subgroup = None
    if subgroup_records:
        for record in subgroup_records:
            if record["AbsDiff"] > max_abs_diff:
                max_abs_diff = record["AbsDiff"]
                max_violation_subgroup = {
                    "subgroup": record["AttributeValues"],
                    "size": record["Size"],
                    "utility": record["Utility"],
                    "utility_diff": record["UtilityDiff"],
                    "abs_diff": record["AbsDiff"]
                }
    
    # Returns: (Records, Count, Enum Time, Iter Time, Max Abs Diff, Max Violation Info)
    return subgroup_records, cate_calc_count, enumeration_time, iteration_time, max_abs_diff, max_violation_subgroup