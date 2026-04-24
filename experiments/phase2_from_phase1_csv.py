#!/usr/bin/env python3
"""
Phase 2: one random walk per Phase 1 candidate, on the candidate's subgroup only.

Loads the same intervention frame as Phase 1 (here: pre-encoded SO + FormalEducation==k),
reads Phase 1 CSV, and calls run_phase2 (population ate_all on full df_rw, RW on df_rw[ψ_grp]).
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

from run_fixed_intervention_pipeline import load_pre_encoded_intervention_frame, run_phase2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 from Phase 1 CSV (subgroup RW).")
    p.add_argument("--encoded-dataset", type=Path, required=True)
    p.add_argument("--phase1-csv", type=Path, required=True)
    p.add_argument("--out-trace", type=Path, required=True)
    p.add_argument("--out-topk", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, default=None)
    p.add_argument("--intervention-col", type=str, default="FormalEducation")
    p.add_argument("--intervention-equals", type=int, required=True)
    p.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    p.add_argument("--treatment-col", type=str, default="TempTreatment")
    p.add_argument("--delta", type=int, default=5000, help="Same as Phase 1 / RW minimum subgroup size.")
    p.add_argument("--epsilon-hom", type=float, default=2000.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-candidates", type=int, default=0, help="0 = all rows in Phase 1 CSV.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--verbose-rw", action="store_true", help="Print RW break messages (very noisy).")
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

    tr, top, rt = run_phase2(
        candidates,
        df_rw,
        outcome_col=args.outcome_col,
        treatment_col=args.treatment_col,
        delta_rw=args.delta,
        epsilon_hom=args.epsilon_hom,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        seed=args.seed,
        quiet_rw=not args.verbose_rw,
    )

    args.out_trace.parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(args.out_trace, index=False)
    top.to_csv(args.out_topk, index=False)

    meta = {
        "counts": counts,
        "delta": args.delta,
        "epsilon_hom": args.epsilon_hom,
        "phase1_csv": str(args.phase1_csv.resolve()),
        "encoded_dataset": str(args.encoded_dataset.resolve()),
        "runtime_sec_total": rt,
        "n_trace_rows": int(len(tr)),
        "n_homogeneous": int(tr["is_homogeneous"].sum()) if not tr.empty else 0,
        "top_k_saved": int(len(top)),
    }
    meta_path = args.out_meta or args.out_trace.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
