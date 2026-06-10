#!/usr/bin/env python3
"""
Phase 1: Fixed intervention + grouping enumeration with coverage-only pruning.

For each lattice node: if coverage < delta, prune immediately (no utility/score, no expansion).
Only for coverage >= delta: if (``pat``, row mask) was never seen before, compute utility
(CATE via ``calculate_ate_safe``) and score; duplicate (``pat``, mask) states skip ATE
and CSV append but still expand children so the DFS is unchanged.

Output: CSV with one row per **distinct** valid (``pat``, mask) (coverage >= delta only) + meta.json.

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "yarden_files"))

from ATE_update import calculate_ate_safe
from algorithms.epsilon_greedy_rule_mining import _compute_linear_score 


def encode_dataframe_local(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        vals = out[c].unique()
        out[c] = out[c].map({v: i + 1 for i, v in enumerate(vals)})
    for c in out.select_dtypes(include=["bool"]).columns:
        out[c] = out[c].astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--raw-dataset",
        type=Path,
        help="Non-encoded CSV: treatment from --intervention-value-text, then label-encode objects.",
    )

    src.add_argument(
        "--encoded-dataset",
        type=Path,
        help="Pre-encoded numeric CSV: treatment = (intervention_col == intervention_equals); no label-encode.",
    )

    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, default=None)
    p.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    p.add_argument("--treatment-col", type=str, default="TempTreatment")
    p.add_argument("--intervention-col", type=str, default="FormalEducation")
    p.add_argument(
        "--intervention-value-text",
        type=str,
        default=None,
        help="Required with --raw-dataset: exact string for treated units on intervention column.",
    )

    p.add_argument(
        "--intervention-equals",
        type=int,
        default=None,
        help="Required with --encoded-dataset: treated if intervention_col equals this integer.",
    )
    
    p.add_argument("--delta", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--min-utility",
        type=float,
        default=None,
        help="If set, keep only rules with utility >= this threshold in output CSV.",
    )
    args = p.parse_args()

    if args.raw_dataset is not None:
        if not args.intervention_value_text:
            p.error("--raw-dataset requires --intervention-value-text")
        df = pd.read_csv(args.raw_dataset)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
        intervention_label = args.intervention_value_text
        df[args.treatment_col] = (df[args.intervention_col] == args.intervention_value_text).astype(int)
        treated_count = int(df[args.treatment_col].sum())
        df = df.drop(columns=[args.intervention_col])
        enc = encode_dataframe_local(df)
        pre_encoded = False
    else:
        if args.intervention_equals is None:
            p.error("--encoded-dataset requires --intervention-equals")
        df = pd.read_csv(args.encoded_dataset)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
        ic = pd.to_numeric(df[args.intervention_col], errors="coerce")
        df[args.treatment_col] = (ic == args.intervention_equals).astype(int)
        treated_count = int(df[args.treatment_col].sum())
        df = df.drop(columns=[args.intervention_col])
        enc = df
        intervention_label = f"{args.intervention_col}=={args.intervention_equals}"
        pre_encoded = True

    enc[args.outcome_col] = pd.to_numeric(enc[args.outcome_col], errors="coerce")

    features = sorted(
        [c for c in enc.columns if c not in {args.outcome_col, args.treatment_col}],
        key=lambda c: enc[c].nunique(),
        reverse=True,
    )
    arr = {c: enc[c].to_numpy() for c in features}
    vals = {c: sorted(pd.Series(arr[c]).dropna().unique().tolist()) for c in features}
    n = len(enc)
    max_outcome = float(np.nanmax(enc[args.outcome_col].to_numpy()))
    if not np.isfinite(max_outcome) or max_outcome <= 0:
        max_outcome = 1.0
    ate_all = calculate_ate_safe(enc, args.treatment_col, args.outcome_col, args.delta)

    stack = [(0, np.ones(n, dtype=bool), {})]
    rows = []
    visited_total = 0
    pruned_low_coverage = 0
    skipped_duplicate_pat_mask = 0
    seen_pat_mask: set[tuple[str, bytes]] = set()
    t0 = time.time()

    while stack:
        i, mask, pat = stack.pop()
        visited_total += 1
        cov = int(mask.sum())
        if cov < args.delta:
            pruned_low_coverage += 1
            continue

        pat_key = json.dumps(pat, sort_keys=True)
        mask_key = np.asarray(mask, dtype=bool).tobytes()
        key = (pat_key, mask_key)
        if key in seen_pat_mask:
            skipped_duplicate_pat_mask += 1
        else:
            util = calculate_ate_safe(enc.loc[mask], args.treatment_col, args.outcome_col, args.delta)
            score = (
                _compute_linear_score(float(util), cov, n, args.alpha, max_outcome)
                if util is not None and np.isfinite(util)
                else np.nan
            )
            util_f = float(util) if util is not None and np.isfinite(util) else np.nan
            keep = True
            if args.min_utility is not None:
                keep = bool(np.isfinite(util_f) and util_f >= float(args.min_utility))

            if keep:
                rows.append(
                    {
                        "psi_int_attr": args.intervention_col,
                        "psi_int_value": intervention_label,
                        "psi_grp_json": pat_key,
                        "num_predicates": len(pat),
                        "coverage": cov,
                        "utility": util_f,
                        "score": float(score) if np.isfinite(score) else np.nan,
                    }
                )
            seen_pat_mask.add(key)

        if i == len(features):
            continue
        stack.append((i + 1, mask, dict(pat)))
        c = features[i]
        a = arr[c]
        for v in vals[c]:
            m2 = mask & (a == v)
            p2 = dict(pat)
            p2[c] = int(v) if float(v).is_integer() else float(v)
            stack.append((i + 1, m2, p2))

    elapsed_lo = time.time()
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    meta = {
        "dataset_rows": int(n),
        "treated_count": treated_count,
        "pre_encoded_dataset": pre_encoded,
        "intervention_col": args.intervention_col,
        "intervention_spec": intervention_label,
        "delta": args.delta,
        "alpha": args.alpha,
        "ate_all": float(ate_all) if ate_all is not None and np.isfinite(ate_all) else None,
        "visited_nodes_total": int(visited_total),
        "pruned_low_coverage_nodes": int(pruned_low_coverage),
        "skipped_duplicate_pat_mask_nodes": int(skipped_duplicate_pat_mask),
        "distinct_valid_patterns_saved": int(len(df)),
        "valid_patterns_scored_and_saved": int(len(df)),
        "elapsed_sec": float(elapsed_lo - t0),
        "output_csv": str(args.out_csv.resolve()),
    }
    meta_path = args.out_meta or args.out_csv.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
