from __future__ import annotations

import json
import random
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
    """
    score(r) = alpha * coverage(r)/|D| + (1-alpha) * utility(r)/(2*max(O))
    """
    cov_term = (coverage / full_n) if full_n else 0.0
    # Guard against degenerate outcome scale; keep denominator positive.
    denom = 2.0 * max_outcome if max_outcome > 0 else 1.0
    util_term = utility / denom
    return float(alpha * cov_term + (1.0 - alpha) * util_term)


def _subset_signature(indices: Set[int]) -> Tuple[int, ...]:
    return tuple(sorted(indices))


def _single_walk_with_eps_greedy(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    ate_all: float,
    full_dataset_size: int,
    alpha: float,
    max_outcome: float,
    max_depth: int,
    predicate_explore_prob: float,
    attribute_weights: Optional[Dict[str, float]],
    rng: random.Random,
    X: np.ndarray,
    lookup: Dict[str, Tuple[str, str]],
    col_names: List[str],
    col_weights: np.ndarray,
    global_cate_cache: Dict[Tuple[int, ...], float],
    top_k_candidates: int = 8,
) -> Tuple[bool, int, Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    One randomized walk with RW stop logic and epsilon-greedy predicate selection.
    Returns:
      (is_homogeneous, checked_count, breaking_subgroup, homogeneous_rule)
    """
    n_features = len(col_names)

    checked = 0
    visited: Set[Tuple[int, ...]] = set()

    def eval_subset(mask: np.ndarray, idxs: Set[int]) -> Optional[float]:
        nonlocal checked
        key = _subset_signature(idxs)
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

    # Start walk from full intervention-conditioned dataset
    mask = np.ones(len(df), dtype=bool)
    idxs: Set[int] = set()
    attrs_used: Set[str] = set()

    cate_root = eval_subset(mask, idxs)
    if cate_root is not None and abs(cate_root - ate_all) > epsilon:
        return False, checked, {"(full rule sub-dataset)": ""}, None

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

        candidates = sorted(candidates, key=lambda i: col_weights[i], reverse=True)[:top_k_candidates]

        # Evaluate candidates for exploitation objective (linear score)
        evaluated: List[Tuple[int, np.ndarray, float, float, int]] = []
        for i in candidates:
            new_mask = mask & X[:, i]
            cate_new = eval_subset(new_mask, idxs | {i})
            if cate_new is None or (not np.isfinite(cate_new)):
                continue
            coverage = int(new_mask.sum())
            rule_score = _compute_linear_score(cate_new, coverage, full_dataset_size, alpha, max_outcome)
            evaluated.append((i, new_mask, cate_new, rule_score, coverage))

        if not evaluated:
            break

        if rng.random() < predicate_explore_prob:
            chosen = rng.choice(evaluated)
        else:
            chosen = max(evaluated, key=lambda t: t[3])

        best_i, best_mask, best_cate, best_score, best_cov = chosen
        mask = best_mask
        idxs.add(best_i)
        attrs_used.add(lookup[col_names[best_i]][0])

        if abs(best_cate - ate_all) > epsilon:
            breaking = {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs}
            return False, checked, breaking, None

    # End-of-walk rule is homogeneous
    final_cate = calculate_ate_safe(df[mask], treatment_col, outcome_col, delta)
    final_cov = int(mask.sum())
    final_score = _compute_linear_score(final_cate, final_cov, full_dataset_size, alpha, max_outcome)
    homogeneous_rule = {
        "psi_grp": {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs},
        "coverage": final_cov,
        "utility": float(final_cate),
        "score": float(final_score),
        "num_predicates": len(idxs),
    }
    return True, checked, None, homogeneous_rule


def mine_top_k_homogeneous_rules(
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
    predicate_explore_prob: float = 0.1,
    max_depth: int = 5,
    attribute_weights: Optional[Dict[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[bool, int, Dict[str, Any], List[Dict[str, Any]]]:
    rng = rng or random.Random()
    ate_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    max_outcome = pd.to_numeric(df[outcome_col], errors="coerce").max()
    if not np.isfinite(max_outcome):
        max_outcome = 1.0
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")
    onehot, lookup = _onehot_lookup(mining_df)
    X = onehot.values
    col_names = list(onehot.columns)
    attr_w = attribute_weights or {}
    col_weights = np.array([attr_w.get(lookup[c][0], 1.0) for c in col_names], dtype=float)

    total_checked = 0
    homogeneous_walks = 0
    non_homogeneous_walks = 0
    top_rules: List[Dict[str, Any]] = []
    seen_signatures: Set[str] = set()
    global_cate_cache: Dict[Tuple[int, ...], float] = {}

    for _ in range(total_walks):
        is_hom, checked, _breaking, hom_rule = _single_walk_with_eps_greedy(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            ate_all=ate_all,
            full_dataset_size=len(df),
            alpha=alpha,
            max_outcome=float(max_outcome),
            max_depth=max_depth,
            predicate_explore_prob=predicate_explore_prob,
            attribute_weights=attribute_weights,
            rng=rng,
            X=X,
            lookup=lookup,
            col_names=col_names,
            col_weights=col_weights,
            global_cate_cache=global_cate_cache,
        )
        total_checked += checked

        if is_hom and hom_rule is not None:
            homogeneous_walks += 1
            signature = json.dumps(hom_rule["psi_grp"], sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            row = {
                "psi_int_attr": intervention_attr,
                "psi_int_value": intervention_value,
                "psi_grp": json.dumps(hom_rule["psi_grp"], sort_keys=True),
                "coverage": hom_rule["coverage"],
                "utility": hom_rule["utility"],
                "score": hom_rule["score"],
                "num_predicates": hom_rule["num_predicates"],
                "is_homogeneous": True,
            }
            top_rules.append(row)
            top_rules = sorted(top_rules, key=lambda r: r["score"], reverse=True)[:top_k]
        else:
            non_homogeneous_walks += 1

    details = {
        "homogeneous_walks": homogeneous_walks,
        "non_homogeneous_walks": non_homogeneous_walks,
        "unique_homogeneous_rules_seen": len(seen_signatures),
        "total_walks": total_walks,
        "ate_all": float(ate_all) if np.isfinite(ate_all) else np.nan,
    }
    overall_homogeneous = len(top_rules) > 0
    return overall_homogeneous, total_checked, details, top_rules


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
    predicate_explore_prob: float = 0.1,
    max_depth: int = 5,
    attribute_weights: Optional[Dict[str, float]] = None,
    output_topk_path: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col/tgtO")
    if mode == 1:
        return [], 0

    is_hom, checked, details, top_rules = mine_top_k_homogeneous_rules(
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
        predicate_explore_prob=predicate_explore_prob,
        max_depth=max_depth,
        attribute_weights=attribute_weights,
    )

    print(
        f"RuleMining summary: total_walks={details['total_walks']} homogeneous_walks={details['homogeneous_walks']} "
        f"non_homogeneous_walks={details['non_homogeneous_walks']} checked={checked}"
    )

    if output_topk_path:
        out_path = Path(output_topk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(top_rules).to_csv(out_path, index=False)
        print(f"Saved top-k homogeneous rules to {out_path}")

    return is_hom, checked
