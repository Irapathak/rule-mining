#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "yarden_files"))

from run_fixed_intervention_pipeline import (  # noqa: E402
    build_subgroup_mask,
    load_pre_encoded_intervention_frame,
    run_rw_once,
)
from ATE_update import calculate_ate_safe  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Randomly sample rules (with replacement) and run one RW per draw. "
            "If a sampled rule fails RW, remove it from candidate pool."
        )
    )
    p.add_argument(
        "--phase1-csv",
        type=Path,
        default=REPO_ROOT / "algorithms_results" / "dataset_rule_mining.csv",
    )
    p.add_argument(
        "--encoded-dataset",
        type=Path,
        default=REPO_ROOT / "algorithms" / "code" / "code" / "stackoverflow_data_encoded.csv",
    )
    p.add_argument("--intervention-col", type=str, default="FormalEducation")
    p.add_argument("--intervention-equals", type=int, default=1)
    p.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    p.add_argument("--treatment-col", type=str, default="TempTreatment")
    p.add_argument("--delta", type=int, default=5000)
    p.add_argument("--epsilon-hom", type=float, default=2000.0)
    p.add_argument("--iterations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N iterations (0 disables).",
    )
    p.add_argument(
        "--out-candidates",
        type=Path,
        default=REPO_ROOT / "algorithms_results" / "random_rw_pruned_candidates.csv",
    )
    p.add_argument(
        "--out-trace",
        type=Path,
        default=REPO_ROOT / "algorithms_results" / "random_rw_prune.trace.csv",
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=REPO_ROOT / "algorithms_results" / "random_rw_prune.meta.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rules = pd.read_csv(args.phase1_csv).copy()
    need = {"psi_grp_json", "coverage", "utility", "score"}
    miss = need - set(rules.columns)
    if miss:
        raise SystemExit(f"Phase1 CSV missing required columns: {miss}")
    rules["utility"] = pd.to_numeric(rules["utility"], errors="coerce")
    rules["coverage"] = pd.to_numeric(rules["coverage"], errors="coerce")
    rules["score"] = pd.to_numeric(rules["score"], errors="coerce")
    rules = rules.reset_index(drop=False).rename(columns={"index": "phase1_row_idx"})

    df_rw, counts = load_pre_encoded_intervention_frame(
        args.encoded_dataset,
        args.outcome_col,
        args.treatment_col,
        args.intervention_col,
        args.intervention_equals,
    )
    ate_all = calculate_ate_safe(df_rw, args.treatment_col, args.outcome_col, args.delta)
    ate_all_f = float(ate_all) if ate_all is not None and np.isfinite(ate_all) else float("nan")

    active_indices: Set[int] = set(rules.index.tolist())
    discarded_indices: Set[int] = set()

    trace_rows: List[Dict[str, object]] = []
    stats = {
        "draws": 0,
        "rw_evaluated": 0,
        "rw_pass": 0,
        "rw_fail": 0,
        "mask_errors": 0,
        "skipped_small_subgroup": 0,
        "picked_already_discarded": 0,
    }
    t0 = time.time()

    for t in range(args.iterations):
        if not active_indices:
            break
        pick = rng.choice(tuple(active_indices))
        stats["draws"] += 1
        r = rules.iloc[int(pick)]
        psi = str(r.get("psi_grp_json", "{}")).strip()
        status = "unknown"
        checked_subgroups = 0
        subgroup_rows = 0

        try:
            sm = build_subgroup_mask(df_rw, psi)
        except Exception:
            stats["mask_errors"] += 1
            status = "mask_error"
        else:
            df_sub = df_rw.loc[sm].copy()
            subgroup_rows = int(len(df_sub))
            if subgroup_rows < args.delta:
                stats["skipped_small_subgroup"] += 1
                status = "skip_small_subgroup"
            else:
                stats["rw_evaluated"] += 1
                is_h, checked_subgroups = run_rw_once(
                    df_sub,
                    outcome_col=args.outcome_col,
                    treatment_col=args.treatment_col,
                    delta_rw=args.delta,
                    epsilon_hom=float(args.epsilon_hom),
                    ate_all=ate_all_f,
                    seed=args.seed + t,
                    quiet=True,
                )
                if is_h:
                    stats["rw_pass"] += 1
                    status = "rw_pass"
                else:
                    stats["rw_fail"] += 1
                    status = "rw_fail_discarded"
                    active_indices.discard(int(pick))
                    discarded_indices.add(int(pick))

        trace_rows.append(
            {
                "iter": int(t + 1),
                "phase1_row_idx": int(r["phase1_row_idx"]),
                "picked_rule_idx_internal": int(pick),
                "psi_grp_json": psi,
                "status": status,
                "subgroup_rows": int(subgroup_rows),
                "checked_subgroups": int(checked_subgroups),
                "active_candidates_after_iter": int(len(active_indices)),
            }
        )

        if args.progress_every > 0 and (t + 1) % args.progress_every == 0:
            print(
                f"[progress] iter={t+1} active={len(active_indices)} "
                f"rw_eval={stats['rw_evaluated']} rw_pass={stats['rw_pass']} rw_fail={stats['rw_fail']}"
            )

    elapsed = float(time.time() - t0)

    survivors = rules.iloc[sorted(active_indices)].copy()
    args.out_candidates.parent.mkdir(parents=True, exist_ok=True)
    survivors.to_csv(args.out_candidates, index=False)
    pd.DataFrame(trace_rows).to_csv(args.out_trace, index=False)

    meta = {
        "counts": counts,
        "phase1_csv": str(args.phase1_csv.resolve()),
        "encoded_dataset": str(args.encoded_dataset.resolve()),
        "iterations_requested": int(args.iterations),
        "iterations_executed": int(stats["draws"]),
        "delta": int(args.delta),
        "epsilon_hom": float(args.epsilon_hom),
        "seed": int(args.seed),
        "initial_candidates": int(len(rules)),
        "final_candidates": int(len(active_indices)),
        "discarded_candidates": int(len(discarded_indices)),
        "discard_fraction": float(len(discarded_indices) / len(rules)) if len(rules) else 0.0,
        "rw_evaluated": int(stats["rw_evaluated"]),
        "rw_pass": int(stats["rw_pass"]),
        "rw_fail": int(stats["rw_fail"]),
        "mask_errors": int(stats["mask_errors"]),
        "skipped_small_subgroup": int(stats["skipped_small_subgroup"]),
        "runtime_sec_total": elapsed,
        "output_candidates_csv": str(args.out_candidates.resolve()),
        "output_trace_csv": str(args.out_trace.resolve()),
    }
    with open(args.out_meta, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
