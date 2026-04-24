#!/usr/bin/env python3
"""
Phase 3: ε-greedy scheduling over Phase 1 rules, one subgroup RW per draw.

- Explore (prob --epsilon-arm): uniform random rule among all candidates.
- Exploit (1 - that prob): rule(s) with highest Phase 1 score (tie broken uniformly), or with
  ``--exploit-always-top`` always row 0 after sorting by score (with replacement).

Same RW mechanics and ate_all / delta as Phase 2 (see run_phase3 in run_fixed_intervention_pipeline.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "yarden_files"))

from run_fixed_intervention_pipeline import load_pre_encoded_intervention_frame, run_phase3  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3: ε-greedy over Phase 1 rules + subgroup RW.")
    p.add_argument("--encoded-dataset", type=Path, required=True)
    p.add_argument("--phase1-csv", type=Path, required=True)
    p.add_argument("--out-trace", type=Path, required=True)
    p.add_argument("--out-topk", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, default=None)
    p.add_argument("--intervention-col", type=str, default="FormalEducation")
    p.add_argument("--intervention-equals", type=int, required=True)
    p.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    p.add_argument("--treatment-col", type=str, default="TempTreatment")
    p.add_argument("--delta", type=int, default=5000)
    p.add_argument(
        "--epsilon-hom",
        type=float,
        default=2000.0,
        help="RW homogeneity threshold: break if |CATE_subgroup - ate_all| exceeds this (salary-scale; not mining EPSILON).",
    )
    p.add_argument(
        "--epsilon-arm",
        type=float,
        default=0.1,
        help="P(explore): random rule. P(exploit)=1-this: max Phase 1 score.",
    )
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--no-replacement",
        action="store_true",
        help="Within a run, remove picked rules from selection pool.",
    )
    p.add_argument(
        "--exploit-always-top",
        action="store_true",
        help="On exploit, always evaluate Phase 1 top-score rule (row 0); implies with replacement.",
    )
    p.add_argument("--verbose-rw", action="store_true")
    p.add_argument(
        "--max-picks-per-rule",
        type=int,
        default=2000,
        help="Remove a Phase 1 row from the selection pool after this many picks (0 = unlimited).",
    )
    p.add_argument(
        "--no-meta-json-stdout",
        action="store_true",
        help="Do not print full meta JSON to stdout after the run (summary line is still printed).",
    )
    args = p.parse_args()

    df_rw, counts = load_pre_encoded_intervention_frame(
        args.encoded_dataset,
        args.outcome_col,
        args.treatment_col,
        args.intervention_col,
        args.intervention_equals,
    )
    candidates = pd.read_csv(args.phase1_csv)
    need = {"psi_grp_json", "psi_int_attr", "psi_int_value", "coverage", "utility", "score"}
    miss = need - set(candidates.columns)
    if miss:
        raise SystemExit(f"Phase 1 CSV missing columns: {miss}")
    if args.exploit_always_top and args.no_replacement:
        raise SystemExit("--exploit-always-top cannot be used with --no-replacement")

    max_picks = args.max_picks_per_rule if args.max_picks_per_rule > 0 else None

    tr, top, rt = run_phase3(
        candidates,
        df_rw,
        outcome_col=args.outcome_col,
        treatment_col=args.treatment_col,
        delta_rw=args.delta,
        epsilon_hom=args.epsilon_hom,
        eps_arm=args.epsilon_arm,
        iterations=args.iterations,
        top_k=args.top_k,
        seed=args.seed,
        quiet_rw=not args.verbose_rw,
        no_replacement=args.no_replacement,
        exploit_always_top=args.exploit_always_top,
        max_picks_per_rule=max_picks,
    )

    args.out_trace.parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(args.out_trace, index=False)
    top.to_csv(args.out_topk, index=False)

    n_exploit = int((tr["selection_mode"] == "exploit").sum()) if not tr.empty and "selection_mode" in tr.columns else 0
    n_explore = int((tr["selection_mode"] == "explore").sum()) if not tr.empty and "selection_mode" in tr.columns else 0

    meta = {
        "counts": counts,
        "delta": args.delta,
        "epsilon_hom": args.epsilon_hom,
        "epsilon_arm_explore": args.epsilon_arm,
        "iterations": args.iterations,
        "no_replacement": bool(args.no_replacement),
        "exploit_always_top": bool(args.exploit_always_top),
        "phase1_csv": str(args.phase1_csv.resolve()),
        "encoded_dataset": str(args.encoded_dataset.resolve()),
        "runtime_sec_total": rt,
        "n_trace_rows": int(len(tr)),
        "n_explore_draws": n_explore,
        "n_exploit_draws": n_exploit,
        "n_homogeneous": int(tr["is_homogeneous"].sum()) if not tr.empty else 0,
        "top_k_saved": int(len(top)),
        "max_picks_per_rule": max_picks,
    }
    meta_path = args.out_meta or args.out_trace.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    n_homog = meta["n_homogeneous"]
    print(
        f"Phase3 done: iterations={args.iterations} epsilon_arm={args.epsilon_arm} "
        f"trace_rows={meta['n_trace_rows']} explore={n_explore} exploit={n_exploit} "
        f"homogeneous={n_homog} top_k_saved={meta['top_k_saved']} "
        f"runtime_sec={rt:.1f} meta={meta_path}",
        flush=True,
    )
    if not args.no_meta_json_stdout:
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
