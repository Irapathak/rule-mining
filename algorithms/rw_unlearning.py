from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)
BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫").astype(str), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup


def _homog_rw_direct(
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        delta: int,
        epsilon: float,
        ate_all: float,
        k_walks: int = 400,
        max_depth: int = 5,
        size_stop: float = 0.8,          # ← accepted, intentionally unused
        rng: Optional[random.Random] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        **_: object,                     # ← swallow future params safely
) -> Tuple[bool, int, int]:
    """
    Returns (is_homogeneous, checked_subgroups, walks_completed).
    ``walks_completed`` counts started outer-loop random walks (root draws), including
    the walk that triggers an early ``False`` exit.
    """

    rng = rng or random.Random()

    # ------------------ SETUP ------------------
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    onehot, lookup = _onehot_lookup(mining_df)
    X = onehot.values
    Y = df[outcome_col].values
    col_names = list(onehot.columns)
    n_features = len(col_names)

    # ------------------ WEIGHTS (NO COMPRESSION) ------------------
    attr_w = attribute_weights or {}
    col_weights = np.array([
        attr_w.get(lookup[c][0], 1.0) for c in col_names
    ], dtype=float)

    # ------------------ ROOT SCORING (RISK-BASED) ------------------
    root_scores = []
    root_indices = []

    for i in range(n_features):
        mask = X[:, i]
        if mask.sum() < delta:
            continue

        try:
            sub_df = df[mask]
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta)
            if not np.isfinite(cate):
                continue  # skip NaN/Inf
            diff = abs(cate - ate_all)
        except LinAlgError:
            continue

        score = col_weights[i] * (1.0 + diff * 10.0)
        if not np.isfinite(score):
            continue  # skip NaN/Inf
        if score <= 0.0:
            continue  # skip zero-weight

        root_scores.append(score)
        root_indices.append(i)

    if not root_indices:
        return True, 0, 0  # nothing valid to explore

    root_scores = np.array(root_scores, dtype=float)
    root_probs = root_scores / np.sum(root_scores)  # guaranteed finite now

    # ------------------ SEARCH ------------------
    visited: Set[frozenset] = set()
    cate_cache: Dict[frozenset, float] = {}
    checked = 0

    def eval_subset(mask, idxs):
        nonlocal checked
        key = frozenset(idxs)
        if key in visited:
            return None
        visited.add(key)
        checked += 1

        if key in cate_cache:
            return cate_cache[key]

        try:
            val = calculate_ate_safe(df[mask], treatment_col, outcome_col, delta)
            cate_cache[key] = val
            return val
        except LinAlgError:
            return None

    # ------------------ WALKS ------------------
    chosen_roots = rng.choices(range(len(root_indices)), weights=root_probs, k=k_walks)

    walks_completed = 0
    for r in chosen_roots:
        walks_completed += 1
        start_idx = root_indices[r]
        mask = X[:, start_idx]
        idxs = {start_idx}
        attrs_used = {lookup[col_names[start_idx]][0]}

        cate = eval_subset(mask, idxs)
        if cate is not None and abs(cate - ate_all) > epsilon:
            print(f"Breaking Subgroup: "
                  f"{ {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs} }")
            return False, checked, walks_completed

        # ---------- GREEDY EXPANSION ----------
        for _ in range(max_depth):
            candidates = []

            for i in range(n_features):
                if i in idxs:
                    continue
                attr = lookup[col_names[i]][0]
                if attr in attrs_used:
                    continue

                new_mask = mask & X[:, i]
                if new_mask.sum() < delta:
                    continue

                candidates.append(i)

            if not candidates:
                break

            # evaluate top-K only (huge speedup)
            candidates = sorted(
                candidates,
                key=lambda i: col_weights[i],
                reverse=True
            )[:8]

            best_gain = 0.0
            best_i = None
            best_mask = None

            for i in candidates:
                new_mask = mask & X[:, i]
                cate_new = eval_subset(new_mask, idxs | {i})
                if cate_new is None:
                    continue

                gain = abs(cate_new - ate_all)
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_mask = new_mask

            if best_i is None:
                break

            # commit step
            mask = best_mask
            idxs.add(best_i)
            attrs_used.add(lookup[col_names[best_i]][0])

            if best_gain > epsilon:
                print(f"Breaking Subgroup: "
                      f"{ {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs} }")
                return False, checked, walks_completed

    print(f"Total checked: {checked}")
    return True, checked, walks_completed



def calc_utility_for_subgroups(
        mode: int,
        algorithm: Callable,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        *,
        outcome_col: Optional[str] = None,
        tgtO: Optional[str] = None,
        utility_all: Optional[float] = None,
        k_walks: int = 1500,  # CHANGED: Propagate increased k_walks
        size_stop: float = 0.8,
        rng: Optional[random.Random] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        **kwargs: object,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None: raise ValueError("Need outcome_col")
    if utility_all is None: raise ValueError("Need utility_all for RW")
    if mode != 1:
        is_h, checked, _walks = _homog_rw_direct(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            ate_all=utility_all,
            k_walks=k_walks,
            size_stop=size_stop,
            rng=rng,
            attribute_weights=attribute_weights
        )
        return is_h, checked
    return [], 0