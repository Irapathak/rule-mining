#!/usr/bin/env python3
"""
Greedy sliding-window top-k: RW validation with 1 walk/sweep, 100 RW cap, continue on eviction.
Stops when window matches brute-force top-k multiset.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from common import RESULTS, add_dataset_args, load_ate_all, load_candidates_csv, load_frame, write_meta


def _norm_psi(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "{}"
    s = str(x).strip()
    return s if s else "{}"


def _next_rank(ordered: pd.DataFrame, positions: List[int], next_rank: int) -> Tuple[Optional[int], int]:
    have = {_norm_psi(ordered.iloc[i]["psi_grp_json"]) for i in positions}
    j = next_rank
    while j < len(ordered):
        if _norm_psi(ordered.iloc[j]["psi_grp_json"]) not in have:
            return j, j + 1
        j += 1
    return None, j


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy RW sliding window.")
    add_dataset_args(p)
    p.add_argument("--rank-by", choices=["utility", "score"], default="score")
    p.add_argument("--target-csv", type=Path, default=RESULTS / "brute_force_top10.csv")
    p.add_argument("--window-k", type=int, default=10)
    p.add_argument("--rw-per-rule", type=int, default=1)
    p.add_argument("--max-rw-walks-per-rule", type=int, default=100)
    p.add_argument(
        "--on-eviction",
        choices=["restart_slot0", "continue_next_slot"],
        default="continue_next_slot",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--out-meta", type=Path, default=RESULTS / "greedy.meta.json")
    args = p.parse_args()

    from common import setup_paths

    setup_paths()
    from run_fixed_intervention_pipeline import build_subgroup_mask, run_rw_once  # noqa: E402

    restart_on_fail = args.on_eviction == "restart_slot0"
    k = int(args.window_k)
    cap = int(args.max_rw_walks_per_rule)

    df_rw, counts = load_frame(args)
    ate_all = load_ate_all(df_rw, args)
    ordered = load_candidates_csv(args.candidates_csv, args.rank_by)
    if len(ordered) < k:
        raise SystemExit(f"Need at least {k} candidates")

    target = Counter(_norm_psi(v) for v in pd.read_csv(args.target_csv)["psi_grp_json"])

    positions = list(range(k))
    next_rank = k
    sweeps = events = 0
    rw_used: Dict[str, int] = defaultdict(int)
    frozen: Set[str] = set()
    discard_rw: List[int] = []
    stop = "unknown"
    t0 = time.time()

    while True:
        if Counter(_norm_psi(ordered.iloc[i]["psi_grp_json"]) for i in positions) == target:
            stop = "matched_target"
            break

        sweeps += 1
        aborted_sweep = False

        for slot in range(k):
            gi = positions[slot]
            psi = _norm_psi(ordered.iloc[gi]["psi_grp_json"])
            used_before = rw_used[psi]
            is_h = True
            skip = ""

            if psi in frozen or used_before >= cap:
                skip = "frozen"
            else:
                try:
                    sm = build_subgroup_mask(df_rw, psi)
                    df_sub = df_rw.loc[sm]
                    if len(df_sub) < args.delta:
                        is_h = False
                        skip = "small_subgroup"
                    else:
                        for rw_i in range(min(args.rw_per_rule, cap - used_before)):
                            ok, _ = run_rw_once(
                                df_sub,
                                outcome_col=args.outcome_col,
                                treatment_col=args.treatment_col,
                                delta_rw=args.delta,
                                epsilon_hom=float(args.epsilon_hom),
                                ate_all=ate_all,
                                seed=args.seed + sweeps * 10007 + slot * 17 + gi * 97 + rw_i,
                                quiet=True,
                            )
                            rw_used[psi] += 1
                            if not ok:
                                is_h = False
                                discard_rw.append(rw_used[psi])
                                break
                        if is_h and rw_used[psi] >= cap:
                            frozen.add(psi)
                except Exception as e:
                    is_h = False
                    skip = str(e)

            if not is_h:
                new_pos = positions[:slot] + positions[slot + 1:]
                pick, next_rank = _next_rank(ordered, new_pos, next_rank)
                if pick is None:
                    stop = "exhausted"
                    break
                new_pos.append(pick)
                positions = new_pos
                events += 1
                if restart_on_fail:
                    aborted_sweep = True
                    break

        if stop == "exhausted":
            break
        if aborted_sweep:
            if args.progress_every and sweeps % args.progress_every == 0:
                print(f"[progress] sweeps={sweeps} replacements={events} elapsed={time.time()-t0:.1f}s")
            continue
        if args.progress_every and sweeps % args.progress_every == 0:
            print(f"[progress] sweeps={sweeps} replacements={events} elapsed={time.time()-t0:.1f}s")

    final_psi = [_norm_psi(ordered.iloc[i]["psi_grp_json"]) for i in positions]
    write_meta(
        args.out_meta,
        {
            "method": "greedy",
            "counts": counts,
            "candidates_csv": str(args.candidates_csv.resolve()),
            "target_csv": str(args.target_csv.resolve()),
            "rank_by": args.rank_by,
            "rw_per_rule": args.rw_per_rule,
            "max_rw_walks_per_rule": cap,
            "on_eviction": args.on_eviction,
            "stop_reason": stop,
            "sweeps": sweeps,
            "replacement_events": events,
            "total_rw_walks": int(sum(rw_used.values())),
            "n_frozen": len(frozen),
            "n_discarded": len(discard_rw),
            "avg_rw_to_discard": float(np.mean(discard_rw)) if discard_rw else None,
            "matches_target": Counter(final_psi) == target,
            "final_psi_grp_json": final_psi,
            "runtime_sec_total": time.time() - t0,
        },
    )


if __name__ == "__main__":
    main()
