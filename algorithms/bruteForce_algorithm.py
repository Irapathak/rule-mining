"""
Core subgroup utility calculation algorithm.
This module contains functions for finding subgroups and calculating their utility.
"""
import sys
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple, Callable, Any
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
from ATE_update import calculate_ate_safe
from numpy.linalg import LinAlgError

EPSILON = 5000


def filter_by_attribute(df: pd.DataFrame, filters: dict, delta: int) -> pd.DataFrame:
    """
    Filters the DataFrame based on the given attribute-value pairs.
    Stops filtering if the group size falls below delta.
    """
    for attribute, value in filters.items():
        df = df[df[attribute] == value]
        if len(df) < delta:  # Stop filtering if group size is below delta
            return pd.DataFrame()  # Return an empty DataFrame
    return df


def _dfs_serial(df: pd.DataFrame,
                attrs: List[str],
                attr_values: Dict[str, List],
                delta: int,
                parent_filters: Dict[str, object],
                start_idx: int
                ) -> List[Tuple[Dict[str, object], int]]:
    """
    Depth-first expansion of one branch — NO multiprocessing inside here.
    Returns a flat list of (filters, size); caller takes care of levels.
    """
    out: List[Tuple[Dict[str, object], int]] = []
    parent_df = df
    for a, v in parent_filters.items():
        parent_df = parent_df[parent_df[a] == v]

    # parent itself is already ≥ delta (checked by caller)
    out.append((parent_filters.copy(), len(parent_df)))

    for idx in range(start_idx, len(attrs)):
        attr = attrs[idx]
        for val in attr_values[attr]:
            child_filters = {**parent_filters, attr: val}
            child_df = parent_df[parent_df[attr] == val]
            size = len(child_df)
            if size >= delta:
                out.extend(
                    _dfs_serial(child_df, attrs, attr_values,
                                delta, child_filters, idx + 1)
                )
    return out


def generate_pruned_levels_mp(df: pd.DataFrame,
                              attr_values: Dict[str, List],
                              delta: int = 10_000,
                              n_jobs=None
                              ) -> List[List[Tuple[Dict[str, object], int]]]:
    """
    Parallel version: fan out *only* the first level that survives ≥ delta,
    then recurse serially inside each process.

    Returns the structure:
        List[ level0, level1, level2, … ]
    where each level element is (filters‑dict, size).
    """
    attrs = list(attr_values.keys())
    root_size = len(df)

    # ----- build first level tasks ---------------------------------
    first_level_tasks = []
    for idx, attr in enumerate(attrs):
        for val in attr_values[attr]:
            filt = {attr: val}
            slice_ = df[df[attr] == val]
            size = len(slice_)
            if size >= delta:
                first_level_tasks.append((slice_, attrs, attr_values,
                                          delta, filt, idx + 1))

    # nothing qualifies
    if not first_level_tasks:
        return [[({}, root_size)]]

    # ----- run first level in parallel -----------------------------
    with mp.Pool(processes=n_jobs) as pool:
        branch_results = pool.starmap(_dfs_serial, first_level_tasks)

    # ----- re‑assemble into explicit levels ------------------------
    flat = [({}, root_size)]  # level 0 (root)
    for branch in branch_results:
        flat.extend(branch)  # each branch already includes its own parent

    # bucket into levels by "number of keys in filters"
    max_depth = max(len(f) for f, _ in flat)
    levels: List[List[Tuple[Dict[str, object], int]]] = [[] for _ in range(max_depth + 1)]
    for filt, sz in flat:
        levels[len(filt)].append((filt, sz))

    return levels


def calc_utility_for_subgroups(
        mode: int,
        attr_vals: Dict[str, List],
        df: pd.DataFrame,
        treatment_col: str,
        tgtO: str,
        delta: int,
        epsilon: int,
        utility_all: float
) -> Any:
    """
    Calculate utility for each subgroup in the DataFrame.
    Returns Boolean for mode 0, or (data, count) for mode 1.
    """
    # Initialize a list to store subgroup data
    subgroup_data = []
    kept = generate_pruned_levels_mp(df, attr_vals, delta, n_jobs=mp.cpu_count())
    num_subgroups = 0
    for lvl, groups in enumerate(kept):
        if lvl == 0:
            continue
        for filt, sz in groups:
            filtered_df = filter_by_attribute(df, filt, delta)

            if not filtered_df.empty:
                try:
                    cate_value = calculate_ate_safe(filtered_df, treatment_col, tgtO, delta)
                except LinAlgError:  # XᵀX still singular
                    continue

                if mode == 0 and abs(utility_all - cate_value) > epsilon:
                    # Violation found - Silent return False
                    return False

                else:
                    utility_diff = cate_value - utility_all
                    num_subgroups += 1

                    subgroup_data.append({
                        "AttributeValues": str(filt),
                        "Size": sz,
                        "Utility": cate_value,
                        "UtilityDiff": utility_diff,
                    })

    # Return the data needed for saving to Excel
    if mode == 0:
        return True  # Silent return True (Homogeneous)

    return subgroup_data, num_subgroups