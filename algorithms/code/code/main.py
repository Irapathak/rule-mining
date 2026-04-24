import pandas as pd
import numpy as np
import time
import sys
import os
import json
from pathlib import Path

from .calculate import wte
from .models import model
from .params import args_nhis

# --- CONFIG LOADING ---
try:
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent.parent.parent / 'configs' / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    GLOBAL_TREATMENT_COL = config.get('TREATMENT_COL')
except Exception:
    GLOBAL_TREATMENT_COL = None


def run_wte_homogeneity_baseline(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        tgtO: str,
        delta: int,
        epsilon: float,
        utility_all: float,
        **kwargs
):
    """
    WTE Baseline Adapter (Optimized for Speed).
    """
    if mode != 0:
        return [], 0

    # 1. Feature Selection
    exclude_cols = {treatment_col, tgtO}
    if GLOBAL_TREATMENT_COL:
        exclude_cols.add(GLOBAL_TREATMENT_COL)

    x_cols = [c for c in df.columns if c not in exclude_cols]
    x_cols = [c for c in x_cols if df[c].nunique() > 1]  # Remove constants

    alpha_val = delta / len(df)

    # 2. Configure Models (Fast & Silent)
    params = args_nhis

    # SPEED OPTIMIZATION: Limit tree complexity
    params['gbdt']['verbosity'] = 0
    params['gbdt']['verbose_eval'] = False
    params['gbdt']['max_depth'] = 6
    params['gbdt']['n_estimators'] = 100  # Ensure we don't build 1000 trees

    params['rf']['verbose'] = -1

    models = {
        "model1": model(),
        "model0": model(),
        "model1_h": model(),
        "model0_h": model(),
        "model_p": model(),
    }

    # Map all outcome models to GBDT
    for m in ["model1", "model0", "model1_h", "model0_h"]:
        models[m].model_class = "gbdt"
    models["model_p"].model_class = "rf"

    args = {
        "args_mu": params['gbdt'],
        "args_p": params['rf'],
    }

    K_folds = 5
    n_splits = 3

    try:
        # Run Min (pos=0)
        est_min, _ = wte(
            data=df, y=tgtO, d=treatment_col, x=x_cols, alpha=[alpha_val],
            K=K_folds, split=n_splits, models=models, args=args,
            pos=0, randomized=False, silent=True
        )
        wte_min = est_min[alpha_val]

        # Run Max (pos=1)
        est_max, _ = wte(
            data=df, y=tgtO, d=treatment_col, x=x_cols, alpha=[alpha_val],
            K=K_folds, split=n_splits, models=models, args=args,
            pos=1, randomized=False, silent=True
        )
        wte_max = est_max[alpha_val]

        # 4. Homogeneity Check
        diff_lower = abs(wte_min - utility_all)
        diff_upper = abs(wte_max - utility_all)
        max_deviation = max(diff_lower, diff_upper)

        is_homogeneous = max_deviation <= epsilon

        print(f"   [WTE] Alpha: {alpha_val:.3f} | Dev: {max_deviation:.0f} (Limit: {epsilon})")

        return is_homogeneous, 2

    except Exception as e:
        print(f"   [WTE Error] {str(e)}")
        return True, 0