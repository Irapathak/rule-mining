#!/usr/bin/env python3
"""
Brute-force top-k: scan candidates sorted by score/utility; exhaustive homogeneity check per rule.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import (
    RESULTS,
    add_dataset_args,
    load_ate_all,
    load_candidates_csv,
    load_frame,
    setup_paths,
    topk_stats,
    write_meta,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Brute-force homogeneous top-k scan.")
    add_dataset_args(p)
    p.add_argument("--rank-by", choices=["utility", "score"], default="score")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--out-csv", type=Path, default=RESULTS / "brute_force_top10.csv")
    p.add_argument("--out-meta", type=Path, default=RESULTS / "brute_force_top10.meta.json")
    args = p.parse_args()

    setup_paths()
    from run_fixed_intervention_pipeline import build_subgroup_mask  # noqa: E402
    from algorithms.bruteForce_algorithm import calc_utility_for_subgroups  # noqa: E402

    df_rw, counts = load_frame(args)
    ate_all = load_ate_all(df_rw, args)
    ordered = load_candidates_csv(args.candidates_csv, args.rank_by)

    selected: List[Dict[str, object]] = []
    scanned = evaluated = failed = 0
    t0 = time.time()

    for _, r in ordered.iterrows():
        if len(selected) >= args.k:
            break
        scanned += 1
        psi = str(r["psi_grp_json"]).strip()

        try:
            sm = build_subgroup_mask(df_rw, psi)
        except Exception:
            continue

        df_sub = df_rw.loc[sm]
        if len(df_sub) < args.delta:
            continue

        feature_cols = [c for c in df_sub.columns if c not in {args.treatment_col, args.outcome_col}]
        attr_vals = {
            c: sorted(pd.to_numeric(df_sub[c], errors="coerce").dropna().unique().tolist())
            for c in feature_cols
        }

        evaluated += 1
        is_h = bool(
            calc_utility_for_subgroups(
                mode=0,
                attr_vals=attr_vals,
                df=df_sub,
                treatment_col=args.treatment_col,
                tgtO=args.outcome_col,
                delta=args.delta,
                epsilon=int(args.epsilon_hom),
                utility_all=ate_all,
            )
        )
        if not is_h:
            failed += 1
            if args.progress_every and scanned % args.progress_every == 0:
                print(f"[progress] scanned={scanned} passed={len(selected)} elapsed={time.time()-t0:.1f}s")
            continue

        selected.append(
            {
                "rank": len(selected) + 1,
                "psi_grp_json": psi,
                "coverage": int(r["coverage"]),
                "utility": float(r["utility"]) if pd.notna(r["utility"]) else np.nan,
                "score": float(r["score"]) if pd.notna(r["score"]) else np.nan,
            }
        )
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"[progress] scanned={scanned} passed={len(selected)} elapsed={time.time()-t0:.1f}s")

    out_df = pd.DataFrame(selected)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    write_meta(
        args.out_meta,
        {
            "method": "brute_force",
            "counts": counts,
            "candidates_csv": str(args.candidates_csv.resolve()),
            "rank_by": args.rank_by,
            "k_selected": len(out_df),
            "scanned_rules": scanned,
            "evaluated": evaluated,
            "failed": failed,
            "runtime_sec_total": time.time() - t0,
            **topk_stats(out_df),
            "output_csv": str(args.out_csv.resolve()),
        },
    )


if __name__ == "__main__":
    main()
