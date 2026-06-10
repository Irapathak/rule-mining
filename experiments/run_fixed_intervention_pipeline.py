#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "yarden_files"))

from ATE_update import calculate_ate_safe  # noqa: E402
from algorithms.epsilon_greedy_rule_mining import _compute_linear_score  # noqa: E402
from algorithms.rw_unlearning import _homog_rw_direct  # noqa: E402


def encode_dataframe_local(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        vals = out[c].unique()
        out[c] = out[c].map({v: i + 1 for i, v in enumerate(vals)})
    for c in out.select_dtypes(include=["bool"]).columns:
        out[c] = out[c].astype(int)
    return out


def load_fixed_intervention_frame(
    raw_path: Path,
    outcome_col: str,
    intervention_attr: str,
    intervention_value_text: str,
    treatment_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(raw_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    counts = {"rows": len(df)}
    counts["treated_count"] = int((df[intervention_attr] == intervention_value_text).sum())

    df[treatment_col] = (df[intervention_attr] == intervention_value_text).astype(int)
    df = df.drop(columns=[intervention_attr])
    df = encode_dataframe_local(df)
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    return df, counts


def load_pre_encoded_intervention_frame(
    path: Path,
    outcome_col: str,
    treatment_col: str,
    intervention_col: str,
    intervention_equals: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    ic = pd.to_numeric(df[intervention_col], errors="coerce")
    df[treatment_col] = (ic == intervention_equals).astype(int)
    counts = {"rows": len(df), "treated_count": int(df[treatment_col].sum())}
    df = df.drop(columns=[intervention_col])
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    return df, counts


def build_subgroup_mask(df: pd.DataFrame, psi_grp_json: object) -> pd.Series:
    """Boolean mask of rows satisfying conjunctive ψ_grp (Phase 1 JSON); empty {} = all rows."""
    if psi_grp_json is None or (isinstance(psi_grp_json, float) and np.isnan(psi_grp_json)):
        return pd.Series(True, index=df.index)
    s = str(psi_grp_json).strip()
    if not s or s == "{}":
        return pd.Series(True, index=df.index)
    pat = json.loads(s)
    m = pd.Series(True, index=df.index)
    for col, val in pat.items():
        if col not in df.columns:
            raise ValueError(f"psi_grp references unknown column {col!r}")
        series = pd.to_numeric(df[col], errors="coerce")
        if isinstance(val, bool):
            target: float | int = int(val)
        elif isinstance(val, float):
            target = val
        else:
            target = int(val)
        m &= series == target
    return m


def enumerate_phase1_candidates(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    intervention_attr: str,
    intervention_value_text: str,
    coverage_min: int,
    utility_min: float,
    alpha: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    features = sorted(
        [c for c in df.columns if c not in {outcome_col, treatment_col}],
        key=lambda c: df[c].nunique(),
        reverse=True,
    )
    arr = {c: pd.to_numeric(df[c], errors="coerce").to_numpy() for c in features}
    vals = {c: sorted(pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist()) for c in features}
    n = len(df)
    max_outcome = float(np.nanmax(df[outcome_col].to_numpy()))
    if not np.isfinite(max_outcome) or max_outcome <= 0:
        max_outcome = 1.0
    ate_all = calculate_ate_safe(df, treatment_col, outcome_col, coverage_min)

    stack: List[Tuple[int, np.ndarray, Dict[str, int]]] = [(0, np.ones(n, dtype=bool), {})]
    rows: List[Dict[str, object]] = []
    visited_states = 0
    coverage_pruned = 0
    t0 = time.time()

    while stack:
        i, mask, pat = stack.pop()
        visited_states += 1
        cov = int(mask.sum())
        util = calculate_ate_safe(df.loc[mask], treatment_col, outcome_col, coverage_min) if cov > 0 else np.nan
        score = (
            _compute_linear_score(float(util), cov, n, alpha, max_outcome)
            if util is not None and np.isfinite(util)
            else np.nan
        )
        pruned = bool((util is not None) and np.isfinite(util) and (float(util) < utility_min) and (cov < coverage_min))
        rows.append(
            {
                "psi_int_attr": intervention_attr,
                "psi_int_value": intervention_value_text,
                "psi_grp_json": json.dumps(pat, sort_keys=True),
                "num_predicates": len(pat),
                "coverage": cov,
                "utility": float(util) if util is not None and np.isfinite(util) else np.nan,
                "score": float(score) if np.isfinite(score) else np.nan,
                "is_valid_rule": (not pruned),
            }
        )
        if i == len(features):
            continue
        if cov < coverage_min:
            coverage_pruned += 1
            continue
        stack.append((i + 1, mask, dict(pat)))  # wildcard
        c = features[i]
        a = arr[c]
        for v in vals[c]:
            m2 = mask & (a == v)
            p2 = dict(pat)
            p2[c] = int(v) if float(v).is_integer() else float(v)
            stack.append((i + 1, m2, p2))

    all_df = pd.DataFrame(rows)
    valid_df = all_df[all_df["is_valid_rule"] == True].copy()  # noqa: E712
    meta = {
        "visited_states": int(visited_states),
        "coverage_pruned_subtrees": int(coverage_pruned),
        "all_candidates": int(len(all_df)),
        "valid_candidates": int(len(valid_df)),
        "elapsed_sec": float(time.time() - t0),
        "ate_all": float(ate_all) if ate_all is not None and np.isfinite(ate_all) else np.nan,
    }
    return valid_df, meta


def run_rw_once(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    delta_rw: int,
    epsilon_hom: float,
    ate_all: float,
    seed: int,
    quiet: bool = False,
) -> Tuple[bool, int]:
    rng = random.Random(seed)
    ctx: contextlib.AbstractContextManager = (
        contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    )
    with ctx:
        is_h, checked, _walks = _homog_rw_direct(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta_rw,
            epsilon=epsilon_hom,
            ate_all=ate_all,
            k_walks=1,
            max_depth=5,
            rng=rng,
            attribute_weights=None,
        )
    return bool(is_h), int(checked)


def run_phase2(
    candidates: pd.DataFrame,
    df_rw: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    delta_rw: int,
    epsilon_hom: float,
    top_k: int,
    max_candidates: int,
    seed: int,
    quiet_rw: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    ate_all = calculate_ate_safe(df_rw, treatment_col, outcome_col, delta_rw)
    ate_all_f = float(ate_all) if ate_all is not None and np.isfinite(ate_all) else float("nan")
    cand = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    if max_candidates > 0:
        cand = cand.head(max_candidates).copy()
    trace: List[Dict[str, object]] = []
    t0 = time.time()
    for i, r in cand.iterrows():
        rs = time.time()
        try:
            sm = build_subgroup_mask(df_rw, r["psi_grp_json"])
        except (json.JSONDecodeError, ValueError) as e:
            trace.append(
                {
                    "method": "phase2_bruteforce",
                    "candidate_idx": int(i),
                    "psi_int_attr": r["psi_int_attr"],
                    "psi_int_value": r["psi_int_value"],
                    "psi_grp_json": r["psi_grp_json"],
                    "coverage": r["coverage"],
                    "utility": r["utility"],
                    "score": r["score"],
                    "subgroup_rows": 0,
                    "is_homogeneous": False,
                    "checked_subgroups": 0,
                    "runtime_sec_rule": float(time.time() - rs),
                    "error": str(e),
                }
            )
            continue
        df_sub = df_rw.loc[sm].copy()
        cov_sub = int(len(df_sub))
        if cov_sub < delta_rw:
            trace.append(
                {
                    "method": "phase2_bruteforce",
                    "candidate_idx": int(i),
                    "psi_int_attr": r["psi_int_attr"],
                    "psi_int_value": r["psi_int_value"],
                    "psi_grp_json": r["psi_grp_json"],
                    "coverage": r["coverage"],
                    "utility": r["utility"],
                    "score": r["score"],
                    "subgroup_rows": cov_sub,
                    "is_homogeneous": False,
                    "checked_subgroups": 0,
                    "runtime_sec_rule": float(time.time() - rs),
                    "skip_reason": f"subgroup_rows<{delta_rw}",
                }
            )
            continue
        is_h, checked = run_rw_once(
            df_sub,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            delta_rw=delta_rw,
            epsilon_hom=epsilon_hom,
            ate_all=ate_all_f,
            seed=seed + int(i),
            quiet=quiet_rw,
        )
        trace.append(
            {
                "method": "phase2_bruteforce",
                "candidate_idx": int(i),
                "psi_int_attr": r["psi_int_attr"],
                "psi_int_value": r["psi_int_value"],
                "psi_grp_json": r["psi_grp_json"],
                "coverage": r["coverage"],
                "utility": r["utility"],
                "score": r["score"],
                "subgroup_rows": cov_sub,
                "is_homogeneous": bool(is_h),
                "checked_subgroups": int(checked),
                "runtime_sec_rule": float(time.time() - rs),
            }
        )
    total = float(time.time() - t0)
    tr = pd.DataFrame(trace)
    if tr.empty:
        top = tr.copy()
    else:
        # "Overall homogeneous" means a rule (psi_grp_json) never returned False
        # across all of its visits in this run.
        tr_scored = tr.copy()
        tr_scored["psi_key"] = tr_scored["psi_grp_json"].astype(str)
        per_rule = tr_scored.groupby("psi_key", dropna=False)["is_homogeneous"].agg(
            n_visits="size",
            n_true="sum",
        )
        per_rule["n_false"] = per_rule["n_visits"] - per_rule["n_true"]
        overall_hom_keys = set(per_rule[per_rule["n_false"] == 0].index.tolist())

        top = tr_scored[
            (tr_scored["is_homogeneous"] == True) & (tr_scored["psi_key"].isin(overall_hom_keys))  # noqa: E712
        ].sort_values("score", ascending=False).drop_duplicates(
            subset=["psi_key"]
        ).head(top_k).copy()
        top = top.drop(columns=["psi_key"])
    top.insert(1, "rank", range(1, len(top) + 1))
    top["runtime_sec_total"] = total
    return tr, top, total


def run_phase3(
    candidates: pd.DataFrame,
    df_rw: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str,
    delta_rw: int,
    epsilon_hom: float,
    eps_arm: float,
    iterations: int,
    top_k: int,
    seed: int,
    quiet_rw: bool = True,
    no_replacement: bool = False,
    exploit_always_top: bool = False,
    max_picks_per_rule: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    ε-greedy scheduling over Phase 1 rules (sampling with replacement).

    - With probability ``eps_arm`` (explore): pick a uniformly random rule.
    - Otherwise (exploit): pick uniformly among rule(s) with maximum Phase 1 ``score``, **or**
      if ``exploit_always_top`` is True, always pick row ``0`` (best Phase 1 ``score`` after sort).

    If ``no_replacement`` is True, each chosen rule index is removed from the pool. Exploration and
    exploitation only consider remaining indices; at most ``min(iterations, len(candidates))`` steps.
    ``exploit_always_top`` cannot be combined with ``no_replacement``.

    ``candidates`` are sorted by ``score`` descending before the run; ``iloc`` row 0 is best Phase 1 score.

    Each draw runs one RW on ``df_rw[ψ_grp]`` with population ``ate_all`` on full ``df_rw`` (same as Phase 2).

    If ``max_picks_per_rule`` is set (>0), each candidate row index may be selected at most that many
    times; after that it is removed from the selection pool (explore and exploit). Stops early if the
    pool becomes empty.
    """
    if exploit_always_top and no_replacement:
        raise ValueError("exploit_always_top requires sampling with replacement (no_replacement=False)")
    ate_all = calculate_ate_safe(df_rw, treatment_col, outcome_col, delta_rw)
    ate_all_f = float(ate_all) if ate_all is not None and np.isfinite(ate_all) else float("nan")
    cand = candidates.sort_values("score", ascending=False).reset_index(drop=True).copy()
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0
    rng = random.Random(seed)
    n = len(cand)
    available = list(range(n))
    use_cap = max_picks_per_rule is not None and max_picks_per_rule > 0
    eligible: Optional[Set[int]] = set(range(n)) if use_cap else None
    pick_counts: Dict[int, int] = {i: 0 for i in range(n)} if use_cap else {}
    trace: List[Dict[str, object]] = []
    t0 = time.time()
    max_iters = min(iterations, n) if no_replacement else iterations

    def _selection_pool() -> List[int]:
        if no_replacement:
            base = list(available)
        else:
            base = list(range(n))
        if use_cap and eligible is not None:
            return [i for i in base if i in eligible]
        return base

    for t in range(max_iters):
        if no_replacement and not available:
            break
        pool = _selection_pool()
        if not pool:
            break
        if rng.random() < eps_arm:
            pick = int(rng.choice(pool))
            mode = "explore"
        else:
            if exploit_always_top:
                if 0 in pool:
                    pick = 0
                else:
                    finite_pool = [ix for ix in pool if np.isfinite(cand.at[ix, "score"])]
                    if finite_pool:
                        best_score = max(float(cand.at[ix, "score"]) for ix in finite_pool)
                        best_idxs = [ix for ix in finite_pool if float(cand.at[ix, "score"]) == best_score]
                        pick = int(rng.choice(best_idxs))
                    else:
                        pick = int(rng.choice(pool))
            else:
                finite_pool = [ix for ix in pool if np.isfinite(cand.at[ix, "score"])]
                if finite_pool:
                    best_score = max(float(cand.at[ix, "score"]) for ix in finite_pool)
                    best_idxs = [ix for ix in finite_pool if float(cand.at[ix, "score"]) == best_score]
                    pick = int(rng.choice(best_idxs))
                else:
                    pick = int(rng.choice(pool))
            mode = "exploit"
        if no_replacement:
            available.remove(pick)
        r = cand.iloc[pick]
        rs = time.time()

        def _bump_pick_cap() -> None:
            if not use_cap or eligible is None:
                return
            pick_counts[pick] += 1
            if pick_counts[pick] >= max_picks_per_rule:
                eligible.discard(pick)

        try:
            sm = build_subgroup_mask(df_rw, r["psi_grp_json"])
        except (json.JSONDecodeError, ValueError):
            trace.append(
                {
                    "method": "phase3_eps_greedy",
                    "iter": int(t + 1),
                    "selection_mode": mode,
                    "candidate_row_in_sorted": int(pick),
                    "psi_int_attr": r["psi_int_attr"],
                    "psi_int_value": r["psi_int_value"],
                    "psi_grp_json": r["psi_grp_json"],
                    "coverage": r["coverage"],
                    "utility": r["utility"],
                    "score": r["score"],
                    "subgroup_rows": 0,
                    "is_homogeneous": False,
                    "checked_subgroups": 0,
                    "runtime_sec_rule": float(time.time() - rs),
                }
            )
            _bump_pick_cap()
            continue
        df_sub = df_rw.loc[sm].copy()
        cov_sub = int(len(df_sub))
        if cov_sub < delta_rw:
            trace.append(
                {
                    "method": "phase3_eps_greedy",
                    "iter": int(t + 1),
                    "selection_mode": mode,
                    "candidate_row_in_sorted": int(pick),
                    "psi_int_attr": r["psi_int_attr"],
                    "psi_int_value": r["psi_int_value"],
                    "psi_grp_json": r["psi_grp_json"],
                    "coverage": r["coverage"],
                    "utility": r["utility"],
                    "score": r["score"],
                    "subgroup_rows": cov_sub,
                    "is_homogeneous": False,
                    "checked_subgroups": 0,
                    "runtime_sec_rule": float(time.time() - rs),
                    "skip_reason": f"subgroup_rows<{delta_rw}",
                }
            )
            _bump_pick_cap()
            continue
        is_h, checked = run_rw_once(
            df_sub,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            delta_rw=delta_rw,
            epsilon_hom=epsilon_hom,
            ate_all=ate_all_f,
            seed=seed + 100000 + t,
            quiet=quiet_rw,
        )
        trace.append(
            {
                "method": "phase3_eps_greedy",
                "iter": int(t + 1),
                "selection_mode": mode,
                "candidate_row_in_sorted": int(pick),
                "psi_int_attr": r["psi_int_attr"],
                "psi_int_value": r["psi_int_value"],
                "psi_grp_json": r["psi_grp_json"],
                "coverage": r["coverage"],
                "utility": r["utility"],
                "score": r["score"],
                "subgroup_rows": cov_sub,
                "is_homogeneous": bool(is_h),
                "checked_subgroups": int(checked),
                "runtime_sec_rule": float(time.time() - rs),
            }
        )
        _bump_pick_cap()
    total = float(time.time() - t0)
    tr = pd.DataFrame(trace)
    top = tr[tr["is_homogeneous"] == True].sort_values("score", ascending=False).drop_duplicates(  # noqa: E712
        subset=["psi_grp_json"]
    ).head(top_k).copy()
    top.insert(1, "rank", range(1, len(top) + 1))
    top["runtime_sec_total"] = total
    return tr, top, total


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-intervention Phase1/2/3 pipeline with standardized outputs.")
    p.add_argument("--raw-dataset", type=Path, required=True)
    p.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    p.add_argument("--treatment-col", type=str, default="TempTreatment")
    p.add_argument("--intervention-attr", type=str, default="FormalEducation")
    p.add_argument("--intervention-value-text", type=str, required=True)
    p.add_argument("--coverage-min", type=int, default=5000)
    p.add_argument("--utility-min", type=float, default=520000.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--delta-rw", type=int, default=100)
    p.add_argument(
        "--epsilon-hom",
        type=float,
        default=2000.0,
        help="RW homogeneity: max |CATE - ate_all| before subgroup breaks (default 2000 for SO salary scale).",
    )
    p.add_argument("--epsilon-greedy", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--phase2-max-candidates", type=int, default=1000)
    p.add_argument("--phase3-iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "algorithms_results")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_rw, counts = load_fixed_intervention_frame(
        args.raw_dataset,
        args.outcome_col,
        args.intervention_attr,
        args.intervention_value_text,
        args.treatment_col,
    )

    phase1_valid, m1 = enumerate_phase1_candidates(
        df_rw,
        outcome_col=args.outcome_col,
        treatment_col=args.treatment_col,
        intervention_attr=args.intervention_attr,
        intervention_value_text=args.intervention_value_text,
        coverage_min=args.coverage_min,
        utility_min=args.utility_min,
        alpha=args.alpha,
    )
    f1 = args.out_dir / "phase1_fixed_intervention_valid_rules.csv"
    phase1_valid.to_csv(f1, index=False)

    tr2, top2, rt2 = run_phase2(
        phase1_valid,
        df_rw,
        outcome_col=args.outcome_col,
        treatment_col=args.treatment_col,
        delta_rw=args.delta_rw,
        epsilon_hom=args.epsilon_hom,
        top_k=args.top_k,
        max_candidates=args.phase2_max_candidates,
        seed=args.seed,
        quiet_rw=True,
    )
    f2t = args.out_dir / "phase2_bruteforce_trace.csv"
    f2k = args.out_dir / "phase2_bruteforce_top10.csv"
    tr2.to_csv(f2t, index=False)
    top2.to_csv(f2k, index=False)

    tr3, top3, rt3 = run_phase3(
        phase1_valid,
        df_rw,
        outcome_col=args.outcome_col,
        treatment_col=args.treatment_col,
        delta_rw=args.delta_rw,
        epsilon_hom=args.epsilon_hom,
        eps_arm=args.epsilon_greedy,
        iterations=args.phase3_iterations,
        top_k=args.top_k,
        seed=args.seed,
        quiet_rw=True,
        no_replacement=False,
        exploit_always_top=False,
    )
    f3t = args.out_dir / "phase3_epsgreedy_trace.csv"
    f3k = args.out_dir / "phase3_epsgreedy_top10.csv"
    tr3.to_csv(f3t, index=False)
    top3.to_csv(f3k, index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "phase2_bruteforce",
                "runtime_sec_total": rt2,
                "num_candidates_evaluated": len(tr2),
                "num_homogeneous": int(tr2["is_homogeneous"].sum()) if not tr2.empty else 0,
                "top10_count": len(top2),
            },
            {
                "method": "phase3_eps_greedy",
                "runtime_sec_total": rt3,
                "num_candidates_evaluated": len(tr3),
                "num_homogeneous": int(tr3["is_homogeneous"].sum()) if not tr3.empty else 0,
                "top10_count": len(top3),
            },
        ]
    )
    fs = args.out_dir / "phase23_summary.csv"
    summary.to_csv(fs, index=False)

    meta = {
        "counts": counts,
        "phase1_meta": m1,
        "files": {
            "phase1_valid_rules": str(f1),
            "phase2_trace": str(f2t),
            "phase2_top10": str(f2k),
            "phase3_trace": str(f3t),
            "phase3_top10": str(f3k),
            "phase23_summary": str(fs),
        },
        "params": vars(args),
    }
    fm = args.out_dir / "phase_pipeline_meta.json"
    with open(fm, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
