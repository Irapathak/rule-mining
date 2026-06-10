#!/usr/bin/env python3
"""
ε-greedy top-k: explore/exploit over candidates; batched RW homogeneity per pick.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from common import (
    REPO_ROOT,
    RESULTS,
    add_dataset_args,
    load_ate_all,
    load_candidates_csv,
    load_frame,
    topk_stats,
    write_meta,
)


def _build_topk(trace: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if trace.empty:
        return trace.copy()
    tr = trace.copy()
    tr["psi_key"] = tr["psi_grp_json"].astype(str)
    per = tr.groupby("psi_key")["is_homogeneous"].agg(n_visits="size", n_true="sum")
    hom = set(per[per.n_visits == per.n_true].index)
    top = (
        tr[(tr["is_homogeneous"]) & (tr["psi_key"].isin(hom))]
        .sort_values("score", ascending=False)
        .drop_duplicates("psi_key")
        .head(top_k)
        .drop(columns=["psi_key"])
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def main() -> None:
    p = argparse.ArgumentParser(description="ε-greedy RW top-k.")
    add_dataset_args(p)
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--epsilon-arm", type=float, default=0.8, help="Explore probability.")
    p.add_argument("--rw-batch", type=int, default=100)
    p.add_argument("--max-rw-walks-per-rule", type=int, default=2000)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--out-topk", type=Path, default=RESULTS / "eps_greedy_top10.csv")
    p.add_argument("--out-meta", type=Path, default=RESULTS / "eps_greedy.meta.json")
    args = p.parse_args()

    from common import setup_paths

    setup_paths()
    from run_fixed_intervention_pipeline import build_subgroup_mask  # noqa: E402
    from algorithms.rw_unlearning import _homog_rw_direct  # noqa: E402
    from phase_rw_config import resolve_attribute_weights  # noqa: E402

    df_rw, counts = load_frame(args)
    ate_all = load_ate_all(df_rw, args)
    cand = load_candidates_csv(args.candidates_csv, "score")
    attr_w = resolve_attribute_weights(REPO_ROOT / "configs" / "config.json")

    n = len(cand)
    rng = random.Random(args.seed)
    eligible: Set[int] = set(range(n))
    rw_used: Dict[int, int] = defaultdict(int)
    trace: List[Dict[str, object]] = []
    t0 = time.time()

    for t in range(args.iterations):
        pool = [i for i in eligible]
        if not pool:
            break

        if rng.random() < args.epsilon_arm:
            pick = int(rng.choice(pool))
            mode = "explore"
        else:
            finite = [i for i in pool if np.isfinite(cand.at[i, "score"])]
            if finite:
                best = max(float(cand.at[i, "score"]) for i in finite)
                pick = int(rng.choice([i for i in finite if float(cand.at[i, "score"]) == best]))
            else:
                pick = int(rng.choice(pool))
            mode = "exploit"

        r = cand.iloc[pick]
        rem = args.max_rw_walks_per_rule - rw_used[pick]
        base = {
            "iter": t + 1,
            "mode": mode,
            "candidate_idx": pick,
            "psi_grp_json": r["psi_grp_json"],
            "coverage": r["coverage"],
            "utility": r["utility"],
            "score": r["score"],
        }

        if rem <= 0:
            trace.append({**base, "is_homogeneous": False, "rw_walks": 0, "skip": "cap"})
            eligible.discard(pick)
            continue

        try:
            sm = build_subgroup_mask(df_rw, r["psi_grp_json"])
            df_sub = df_rw.loc[sm]
        except Exception as e:
            trace.append({**base, "is_homogeneous": False, "rw_walks": 0, "skip": str(e)})
            continue

        if len(df_sub) < args.delta:
            trace.append({**base, "is_homogeneous": False, "rw_walks": 0, "skip": "small"})
            continue

        k_walks = min(args.rw_batch, rem)
        with contextlib.redirect_stdout(io.StringIO()):
            is_h, checked, walks = _homog_rw_direct(
                df_sub,
                treatment_col=args.treatment_col,
                outcome_col=args.outcome_col,
                delta=args.delta,
                epsilon=args.epsilon_hom,
                ate_all=ate_all,
                k_walks=k_walks,
                max_depth=args.max_depth,
                rng=random.Random(args.seed + 1_000_000 + t),
                attribute_weights=attr_w,
            )
        rw_used[pick] += walks
        if rw_used[pick] >= args.max_rw_walks_per_rule:
            eligible.discard(pick)

        trace.append(
            {
                **base,
                "is_homogeneous": bool(is_h),
                "checked_subgroups": int(checked),
                "rw_walks": int(walks),
            }
        )
        if args.progress_every and (t + 1) % args.progress_every == 0:
            print(f"[progress] iter={t+1} eligible={len(eligible)} rw={sum(rw_used.values())}")

    tr = pd.DataFrame(trace)
    top = _build_topk(tr, args.top_k)
    args.out_topk.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_topk, index=False)

    write_meta(
        args.out_meta,
        {
            "method": "eps_greedy",
            "counts": counts,
            "candidates_csv": str(args.candidates_csv.resolve()),
            "n_candidates": n,
            "iterations": len(tr),
            "epsilon_arm": args.epsilon_arm,
            "rw_batch": args.rw_batch,
            "max_rw_walks_per_rule": args.max_rw_walks_per_rule,
            "total_rw_walks": int(tr["rw_walks"].sum()) if not tr.empty else 0,
            "top_k_saved": len(top),
            "runtime_sec_total": time.time() - t0,
            **{f"topk_{k}": v for k, v in topk_stats(top).items()},
            "output_csv": str(args.out_topk.resolve()),
        },
    )


if __name__ == "__main__":
    main()
