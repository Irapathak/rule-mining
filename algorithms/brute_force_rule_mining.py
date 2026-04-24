from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

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


def _compute_linear_score(utility: float, coverage: int, full_n: int, alpha: float, max_outcome: float) -> float:
    cov_term = (coverage / full_n) if full_n else 0.0
    denom = 2.0 * max_outcome if max_outcome > 0 else 1.0
    util_term = utility / denom
    return float(alpha * cov_term + (1.0 - alpha) * util_term)


def _single_walk_best_gain(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    ate_all: float,
    full_n: int,
    alpha: float,
    max_outcome: float,
    max_depth: int,
    X: np.ndarray,
    lookup: Dict[str, Tuple[str, str]],
    col_names: List[str],
    global_cate_cache: Dict[Tuple[int, ...], float],
) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
    """
    One RW-style walk with deterministic greedy candidate choice:
    pick candidate with max |CATE(new)-ATE_all| each step.
    """
    n_features = len(col_names)
    checked = 0
    visited: Set[Tuple[int, ...]] = set()

    def eval_subset(mask: np.ndarray, idxs: Set[int]) -> Optional[float]:
        nonlocal checked
        key = tuple(sorted(idxs))
        if key in visited:
            return None
        visited.add(key)
        checked += 1
        if key in global_cate_cache:
            return global_cate_cache[key]
        try:
            val = calculate_ate_safe(df[mask], treatment_col, outcome_col, delta)
            global_cate_cache[key] = val
            return val
        except LinAlgError:
            return None

    mask = np.ones(len(df), dtype=bool)
    idxs: Set[int] = set()
    attrs_used: Set[str] = set()

    cate_root = eval_subset(mask, idxs)
    if cate_root is not None and abs(cate_root - ate_all) > epsilon:
        return False, checked, None

    for _ in range(max_depth):
        candidates: List[int] = []
        for i in range(n_features):
            if i in idxs:
                continue
            attr = lookup[col_names[i]][0]
            if attr in attrs_used:
                continue
            new_mask = mask & X[:, i]
            if int(new_mask.sum()) < delta:
                continue
            candidates.append(i)
        if not candidates:
            break

        best_i = None
        best_mask = None
        best_cate = None
        best_gain = -1.0
        for i in candidates:
            new_mask = mask & X[:, i]
            cate_new = eval_subset(new_mask, idxs | {i})
            if cate_new is None or (not np.isfinite(cate_new)):
                continue
            gain = abs(cate_new - ate_all)
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_mask = new_mask
                best_cate = cate_new

        if best_i is None:
            break

        mask = best_mask
        idxs.add(best_i)
        attrs_used.add(lookup[col_names[best_i]][0])

        if abs(best_cate - ate_all) > epsilon:
            return False, checked, None

    final_cate = calculate_ate_safe(df[mask], treatment_col, outcome_col, delta)
    coverage = int(mask.sum())
    score = _compute_linear_score(final_cate, coverage, full_n, alpha, max_outcome)
    rule = {
        "psi_grp": {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs},
        "coverage": coverage,
        "utility": float(final_cate),
        "score": float(score),
        "num_predicates": len(idxs),
        "is_homogeneous": True,
    }
    return True, checked, rule


def mine_top_k_homogeneous_rules_bruteforce(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    intervention_attr: str,
    intervention_value: str,
    total_walks: int = 20000,
    top_k: int = 10,
    alpha: float = 0.5,
    max_depth: int = 5,
) -> Tuple[bool, int, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Baseline: run one RW-style walk per iteration with pure greedy best-gain
    candidate selection (no epsilon-greedy exploration).
    """
    ate_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    max_outcome = pd.to_numeric(df[outcome_col], errors="coerce").max()
    if not np.isfinite(max_outcome):
        max_outcome = 1.0

    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")
    onehot, lookup = _onehot_lookup(mining_df)
    X = onehot.values
    col_names = list(onehot.columns)
    n_features = len(col_names)

    checked = 0
    top_rules: List[Dict[str, Any]] = []
    seen_rules: Set[str] = set()
    homogeneous_walks = 0
    non_homogeneous_walks = 0
    global_cate_cache: Dict[Tuple[int, ...], float] = {}

    def update_topk(rule_obj: Dict[str, Any]):
        sig = json.dumps(rule_obj["psi_grp"], sort_keys=True)
        if sig in seen_rules:
            return
        seen_rules.add(sig)
        row = {
            "psi_int_attr": intervention_attr,
            "psi_int_value": intervention_value,
            "psi_grp": sig,
            "coverage": rule_obj["coverage"],
            "utility": rule_obj["utility"],
            "score": rule_obj["score"],
            "num_predicates": rule_obj["num_predicates"],
            "is_homogeneous": True,
        }
        top_rules.append(row)
        top_rules.sort(key=lambda r: r["score"], reverse=True)
        if len(top_rules) > top_k:
            top_rules.pop()
    for _ in range(total_walks):
        is_hom, c, rule_obj = _single_walk_best_gain(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            ate_all=ate_all,
            full_n=len(df),
            alpha=alpha,
            max_outcome=float(max_outcome),
            max_depth=max_depth,
            X=X,
            lookup=lookup,
            col_names=col_names,
            global_cate_cache=global_cate_cache,
        )
        checked += c
        if is_hom and rule_obj is not None:
            homogeneous_walks += 1
            update_topk(rule_obj)
        else:
            non_homogeneous_walks += 1

    details = {
        "homogeneous_walks": homogeneous_walks,
        "non_homogeneous_walks": non_homogeneous_walks,
        "total_states_checked": checked,
        "total_walks": total_walks,
        "max_depth": max_depth,
    }
    return len(top_rules) > 0, checked, details, top_rules


def calc_utility_for_subgroups(
    mode: int,
    algorithm: Any,
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    *,
    outcome_col: Optional[str] = None,
    tgtO: Optional[str] = None,
    intervention_attr: str,
    intervention_value: str,
    total_walks: int = 20000,
    top_k: int = 10,
    alpha: float = 0.5,
    max_depth: int = 5,
    output_topk_path: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col/tgtO")
    if mode == 1:
        return [], 0

    is_hom, checked, details, top_rules = mine_top_k_homogeneous_rules_bruteforce(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        delta=delta,
        epsilon=epsilon,
        intervention_attr=intervention_attr,
        intervention_value=intervention_value,
        total_walks=total_walks,
        top_k=top_k,
        alpha=alpha,
        max_depth=max_depth,
    )

    print(
        f"BruteForceRuleMining summary: total_walks={details['total_walks']} checked={checked} "
        f"homogeneous_walks={details['homogeneous_walks']} non_homogeneous_walks={details['non_homogeneous_walks']}"
    )

    if output_topk_path:
        out_path = Path(output_topk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(top_rules).to_csv(out_path, index=False)
        print(f"Saved top-k homogeneous rules to {out_path}")

    return is_hom, checked
