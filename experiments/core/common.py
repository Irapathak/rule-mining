"""Shared setup for brute-force, greedy, and ε-greedy experiment scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "algorithms_results"
ENCODED_DATASET = REPO_ROOT / "algorithms" / "code" / "code" / "stackoverflow_data_encoded.csv"
CANDIDATES_CSV = RESULTS / "random_rw_pruned_candidates.csv"


def setup_paths() -> None:
    for p in (REPO_ROOT, REPO_ROOT / "experiments", REPO_ROOT / "yarden_files"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def add_dataset_args(parser) -> None:
    parser.add_argument("--encoded-dataset", type=Path, default=ENCODED_DATASET)
    parser.add_argument("--candidates-csv", type=Path, default=CANDIDATES_CSV)
    parser.add_argument("--intervention-col", type=str, default="FormalEducation")
    parser.add_argument("--intervention-equals", type=int, default=1)
    parser.add_argument("--outcome-col", type=str, default="ConvertedSalary")
    parser.add_argument("--treatment-col", type=str, default="TempTreatment")
    parser.add_argument("--delta", type=int, default=5000)
    parser.add_argument("--epsilon-hom", type=float, default=2000.0)


def load_frame(args):
    setup_paths()
    from run_fixed_intervention_pipeline import load_pre_encoded_intervention_frame  # noqa: E402

    return load_pre_encoded_intervention_frame(
        args.encoded_dataset,
        args.outcome_col,
        args.treatment_col,
        args.intervention_col,
        args.intervention_equals,
    )


def load_ate_all(df, args) -> float:
    setup_paths()
    from ATE_update import calculate_ate_safe  # noqa: E402
    import numpy as np

    v = calculate_ate_safe(df, args.treatment_col, args.outcome_col, args.delta)
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def load_candidates_csv(path: Path, rank_by: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"psi_grp_json", "coverage", "utility", "score", rank_by}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV missing columns: {miss}")
    out = df.copy()
    out["utility"] = pd.to_numeric(out["utility"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)


def topk_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    u = pd.to_numeric(df["utility"], errors="coerce")
    c = pd.to_numeric(df["coverage"], errors="coerce")
    return {
        "sum_utility": float(u.sum()),
        "sum_coverage": float(c.sum()),
        "avg_utility": float(u.mean()),
        "avg_coverage": float(c.mean()),
    }


def write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(json.dumps(meta, indent=2))
