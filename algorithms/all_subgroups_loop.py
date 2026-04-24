import re
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
from contextlib import contextmanager
import os
import queue
import numpy as np

# Add project root to sys.path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from ATE_update import calculate_ate_safe
from mlxtend.frequent_patterns import fpgrowth, apriori
from apriori_algorithm import calc_utility_for_subgroups as apriori_calc_utility_for_subgroups
from rw_unlearning import calc_utility_for_subgroups as rw_unlearning_calc_utility_for_subgroups
from epsilon_greedy_rule_mining import calc_utility_for_subgroups as epsilon_greedy_rule_mining_calc_utility_for_subgroups
from brute_force_rule_mining import calc_utility_for_subgroups as brute_force_rule_mining_calc_utility_for_subgroups
from random_algorithm import calc_utility_for_subgroups as random_calc_utility_for_subgroups
from causalForest_algorithm import calc_utility_for_subgroups as causalForest_calc_utility_for_subgroups
from algorithms.code.code.main import run_wte_homogeneity_baseline

# --- Configuration ---
TIMEOUT_SECONDS = 3600

_REPO_ROOT = Path(__file__).resolve().parent.parent
_config_path = _REPO_ROOT / 'configs' / 'config.json'

if not _config_path.exists():
    raise FileNotFoundError(f"Config not found: {_config_path}")

with open(_config_path, 'r') as f:
    config = json.load(f)

CHOSEN_DS = config["CHOSEN_DATASET"]
if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]
# Resolve paths securely to the repo root
FULL_DATASET_PATH = str((_REPO_ROOT / ds_config['FULL_DATASET_PATH']).resolve())
RULES_FILE = str((Path(__file__).resolve().parent / ds_config['RULES_FILE']).resolve())
DELTAS = ds_config['DELTAS']
EPSILONS = ds_config['EPSILONS']
TARGET_COLUMN_NAME = ds_config['TARGET_COLUMN']
ATTRIBUTE_WEIGHTS = ds_config.get('ATTRIBUTE_WEIGHTS', {})

# ALGORITHM_NAMES = ["RW", "FPGrowth"]
# ALGORITHM_NAMES = ["FPGrowth"]
ALGORITHM_NAMES = ["BruteForceRuleMining"]
# ALGORITHM_NAMES = ["FPGrowth", "RW"]
RUN_RANDOM_BASELINE = True
NUM_RW_RUNS = 3
TREATMENT_COL = config['TREATMENT_COL']
OPTIMIZATION_MODES = config.get('OPTIMIZATION_MODES', ['direct'])
MODES = config['MODES']


@contextmanager
def timer() -> callable:
    t0 = perf_counter()
    yield lambda: perf_counter() - t0


def worker_wrapper(func, kwargs, result_queue):
    try:
        res = func(**kwargs)
        result_queue.put(("success", res))
    except Exception as e:
        result_queue.put(("error", str(e)))


def save_results_to_csv(algorithm_name, subgroup_data, num_subgroups, condition, treatment, delta, index=0):
    if isinstance(subgroup_data, bool): return ""
    subgroup_df = pd.DataFrame(subgroup_data)
    results_dir = _REPO_ROOT / "algorithms_results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"{CHOSEN_DS}_{algorithm_name}_subgroups_results_delta_{delta}_{index}.csv"
    subgroup_df.to_csv(output_file, index=False)
    print(f"✔  {len(subgroup_data):,} subgroups saved to {output_file}")
    return str(output_file)


def _append_dict_to_csv(csv_path: Path, new_row_dict: dict):
    df_new = pd.DataFrame([new_row_dict])
    if not csv_path.exists():
        df_new.to_csv(csv_path, index=False, mode='w')
    else:
        df_new.to_csv(csv_path, index=False, mode='a', header=False)


def _append_dict_to_excel(excel_path: Path, new_row_dict: dict):
    df_new = pd.DataFrame([new_row_dict])
    if not excel_path.exists():
        df_new.to_excel(excel_path, index=False)
    else:
        try:
            df_existing = pd.read_excel(excel_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"❌ Error appending to Excel: {e}")


def append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta, runtime_seconds):
    results_dir = _REPO_ROOT / "graphs"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"{CHOSEN_DS}_algorithms_time.csv"
    _append_dict_to_csv(csv_path, {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": algorithm_name,
        "chosen_treatment": str(treatment),
        "chosen_condition": str(condition),
        "num_subgroups": str(num_subgroups),
        "delta": str(delta),
        "run_time_seconds": runtime_seconds,
        "run_time_minutes": runtime_seconds / 60
    })
    print(f"✅ Timing results appended to {csv_path}")


def append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, status, runtime,
                               num_subgroups=None, enumeration_time=None, iteration_time=None):
    results_dir = _REPO_ROOT / "graphs"
    results_dir.mkdir(exist_ok=True)
    xlsx_path = results_dir / f"{CHOSEN_DS}_homogeneity_results.xlsx"
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {
        "date": current_date,
        "algorithm": algorithm_name,
        "treatment": str(treatment),
        "condition": str(condition),
        "delta": delta,
        "epsilon": epsilon,
        "homogeneity_status": status,
        "num_subgroups": num_subgroups,
        "run_time_seconds": runtime,
        "run_time_minutes": runtime / 60,
        "enumeration_time_sec": enumeration_time,
        "iteration_time_sec": iteration_time
    }
    _append_dict_to_excel(xlsx_path, new_row)
    print(f"🧬 Homogeneity results appended to {xlsx_path}")


def save_scalability_results(algorithm_name, condition, treatment, num_subgroups, delta, epsilon, status,
                             runtime_seconds, mode, metric_value):
    results_dir = _REPO_ROOT / "graphs"
    results_dir.mkdir(exist_ok=True)
    if mode == 2:
        csv_path = results_dir / f"{CHOSEN_DS}_scalability_rows.csv"
        metric_col = "dataset_percentage"
    elif mode == 3:
        csv_path = results_dir / f"{CHOSEN_DS}_scalability_attributes.csv"
        metric_col = "num_attributes"
    else:
        return
    data = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": algorithm_name,
        "chosen_treatment": str(treatment),
        "chosen_condition": str(condition),
        "num_subgroups": num_subgroups,
        "delta": delta,
        "epsilon": epsilon,
        "status": status,
        metric_col: metric_value,
        "run_time_seconds": runtime_seconds
    }
    header = not csv_path.exists()
    pd.DataFrame([data]).to_csv(csv_path, mode='a', header=header, index=False)
    print(f"📊 Scalability results appended to {csv_path}")


def run_single_execution(target_func, target_kwargs, algorithm_name, chosen_mode, condition, treatment, delta, epsilon,
                         utility_time, attr_vals_time, index=0, metric_value=None):
    result_queue = mp.Queue()
    p = mp.Process(target=worker_wrapper, args=(target_func, target_kwargs, result_queue))
    print(f"▶️  [{algorithm_name}] Starting execution with {TIMEOUT_SECONDS}s timeout...")
    with timer() as elapsed:
        p.start()
        p.join(timeout=TIMEOUT_SECONDS)
        if p.is_alive():
            print(f"⏳ [{algorithm_name}] Timed out! Killing process...")
            p.terminate()
            p.join()
            res = "TIMEOUT"
        else:
            try:
                status, res_obj = result_queue.get_nowait()
                res = res_obj if status == "success" else None
            except:
                res = None

    total_time = elapsed() + utility_time + attr_vals_time
    print(
        f"⏱️  [{algorithm_name}] Finished. Total Time: {total_time:.4f}s (Algo: {elapsed():.4f}s + Overhead: {utility_time + attr_vals_time:.4f}s)")

    if res == "TIMEOUT":
        print(f"\033[93mResult: TIMED OUT\033[0m")
        if chosen_mode == 0:
            append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, "TIMEOUT", total_time)
        elif chosen_mode in [2, 3]:
            save_scalability_results(algorithm_name, condition, treatment, None, delta, epsilon, "TIMEOUT", total_time,
                                     chosen_mode, metric_value)
        return "TIMEOUT", 0

    # --- RESTORED TUPLE UNPACKING BLOCK ---
    raw_result = res
    num_checked = None
    enum_time = None
    iter_time = None

    if isinstance(res, tuple):
        if len(res) == 2:
            raw_result = res[0]
            num_checked = res[1]
        elif len(res) == 3:
            raw_result, enum_time, iter_time = res
        elif len(res) == 4:
            raw_result, num_checked, enum_time, iter_time = res
        elif len(res) == 5:
            raw_result, num_checked, enum_time, iter_time, _ = res

    # --- LOGIC FIX: Handle bool vs list correctly ---
    if isinstance(raw_result, bool):
        is_homogeneous = raw_result
    elif isinstance(raw_result, list):
        is_homogeneous = (len(raw_result) == 0)
    else:
        is_homogeneous = bool(raw_result)

    status_str = "Homogeneous" if is_homogeneous else "NOT Homogeneous"
    color = "\033[92m" if is_homogeneous else "\033[91m"
    print(f"{color}Result: {status_str}\033[0m")

    # Only show subgroup if NOT RW (RW prints its own) and it's a list result
    if not is_homogeneous and algorithm_name != "RW" and isinstance(raw_result, list) and len(raw_result) > 0:
        print(f"Breaking Subgroup: {raw_result[0]}")

    if num_checked is not None:
        print(f"Subgroups checked: {num_checked}")
    # --------------------------------------

    if chosen_mode == 0:
        append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, status_str, total_time,
                                   num_checked, enum_time, iter_time)
        return is_homogeneous, num_checked
    else:
        if chosen_mode == 1:
            save_results_to_csv(algorithm_name, raw_result, num_checked, condition, treatment, delta, index)
            append_timing_results(algorithm_name, condition, treatment, num_checked, delta, total_time)
        elif chosen_mode in [2, 3]:
            save_scalability_results(algorithm_name, condition, treatment, num_checked, delta, epsilon, status_str,
                                     total_time, chosen_mode, metric_value)
        return res, num_checked


def run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                    attr_vals_time=0, force_n_subgroups=None, override_epsilons=None, metric_value=None):
    print(f"Using algorithm: {chosen_algorithm_name}")
    epsilons = override_epsilons if override_epsilons is not None else (EPSILONS if chosen_mode == 0 else [EPSILONS[0]])
    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    with timer() as utility_timer:
        utility_all = calculate_ate_safe(df, TREATMENT_COL, tgtO, delta)
    utility_time = utility_timer()

    if (chosen_mode in [2, 3]) and (not np.isfinite(utility_all)):
        print(f"   ⚠️ [Scalability] Global ATE is NaN/Inf. Forcing to 0.0 to allow runtime test.")
        utility_all = 0.0

    execution_stats = []
    for epsilon in epsilons:
        if chosen_mode == 0: print(f"Running with epsilon: {epsilon}")
        common = dict(df=df, treatment_col=TREATMENT_COL, tgtO=tgtO, delta=delta, epsilon=epsilon, mode=chosen_mode,
                      utility_all=utility_all)
        rw_common = common.copy()
        rw_common['mode'] = 0
        algo_dispatch = {
            "Apriori": (apriori_calc_utility_for_subgroups, dict(common, algorithm=apriori)),
            "FPGrowth": (apriori_calc_utility_for_subgroups, dict(common, algorithm=fpgrowth)),
            "RW": (rw_unlearning_calc_utility_for_subgroups,
                   dict(rw_common, algorithm=apriori, size_stop=0.8, attribute_weights=ATTRIBUTE_WEIGHTS)),
            "Random": (random_calc_utility_for_subgroups,
                       dict(common, n_subgroups=force_n_subgroups if force_n_subgroups else 1000)),
            "CausalForest": (causalForest_calc_utility_for_subgroups, common),
            "WTE": (run_wte_homogeneity_baseline, common)
        }
        dispatch_key = "RW" if chosen_algorithm_name == "RW" else chosen_algorithm_name

        target_func, kwargs = algo_dispatch[dispatch_key]
        _, count = run_single_execution(target_func, kwargs, chosen_algorithm_name, chosen_mode, condition, treatment,
                                        delta, epsilon, utility_time, attr_vals_time, i, metric_value=metric_value)
        execution_stats.append((epsilon, count))
    return execution_stats


def encode_dataframe_local(df):
    df_encoded = df.copy()
    for column in df_encoded.select_dtypes(include=['object']).columns:
        unique_values = df_encoded[column].unique()
        df_encoded[column] = df_encoded[column].map({val: idx + 1 for idx, val in enumerate(unique_values)})
    for col in df_encoded.select_dtypes(include=['bool']).columns:
        df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


def _build_mining_dataset_for_fixed_intervention(full_df, tgtO, intervention_attr, intervention_value):
    """
    Build one mining dataset for fixed intervention psi_int.
    Rule-mining will search psi_grp patterns over this frame.
    """
    if intervention_attr not in full_df.columns:
        return None
    sub_df = full_df.copy()
    sub_df[TREATMENT_COL] = (sub_df[intervention_attr] == intervention_value).astype(int)
    sub_df = sub_df.drop(columns=[intervention_attr])
    if sub_df[TREATMENT_COL].sum() == 0:
        return None
    if CHOSEN_DS != "acs":
        sub_df_encoded = encode_dataframe_local(sub_df)
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
        sub_df_encoded[tgtO] = pd.to_numeric(sub_df_encoded[tgtO], errors='coerce')
    else:
        sub_df_encoded = sub_df.copy()
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
    return sub_df_encoded


def run_eps_greedy_rule_mining_experiment(full_df, chosen_mode, tgtO):
    """
    Mine top-k homogeneous rules r=(psi_grp, psi_int) with fixed intervention.
    """
    intervention_attr = "FormalEducation"
    intervention_value = "Other doctoral degree (Ph.D, Ed.D., etc.)"
    mining_df = _build_mining_dataset_for_fixed_intervention(full_df, tgtO, intervention_attr, intervention_value)
    if mining_df is None:
        print("EpsGreedyRuleMining: fixed intervention could not produce a valid mining dataset.")
        return
    print(
        f"EpsGreedyRuleMining: rows={len(mining_df)} cols={len(mining_df.columns)} "
        f"psi_int={intervention_attr}={intervention_value}"
    )

    for delta in DELTAS:
        if len(mining_df) < delta:
            continue
        print(f"Running for delta: {delta}")
        epsilons = EPSILONS if chosen_mode == 0 else [EPSILONS[0]]
        for epsilon in epsilons:
            if chosen_mode == 0:
                print(f"Running with epsilon: {epsilon}")
            topk_path = _REPO_ROOT / "algorithms_results" / (
                f"{CHOSEN_DS}_EpsGreedyRuleMining_top10_delta_{delta}_eps_{epsilon}.csv"
            )
            kwargs = dict(
                mode=chosen_mode,
                algorithm=None,
                df=mining_df,
                treatment_col=TREATMENT_COL,
                tgtO=tgtO,
                delta=delta,
                epsilon=epsilon,
                intervention_attr=intervention_attr,
                intervention_value=intervention_value,
                total_walks=20000,
                top_k=10,
                alpha=0.5,
                eta=0.01,
                predicate_explore_prob=0.1,
                max_depth=5,
                attribute_weights=ATTRIBUTE_WEIGHTS,
                output_topk_path=str(topk_path),
            )
            run_single_execution(
                epsilon_greedy_rule_mining_calc_utility_for_subgroups,
                kwargs,
                "EpsGreedyRuleMining",
                chosen_mode,
                intervention_attr,
                intervention_value,
                delta,
                epsilon,
                0,
                0,
            )


def run_bruteforce_rule_mining_experiment(full_df, chosen_mode, tgtO):
    """
    Brute-force baseline for fixed-intervention rule mining (top-k homogeneous rules).
    """
    intervention_attr = "FormalEducation"
    intervention_value = "Other doctoral degree (Ph.D, Ed.D., etc.)"
    mining_df = _build_mining_dataset_for_fixed_intervention(full_df, tgtO, intervention_attr, intervention_value)
    if mining_df is None:
        print("BruteForceRuleMining: fixed intervention could not produce a valid mining dataset.")
        return
    print(
        f"BruteForceRuleMining: rows={len(mining_df)} cols={len(mining_df.columns)} "
        f"psi_int={intervention_attr}={intervention_value}"
    )

    for delta in DELTAS:
        if len(mining_df) < delta:
            continue
        print(f"Running for delta: {delta}")
        epsilons = EPSILONS if chosen_mode == 0 else [EPSILONS[0]]
        for epsilon in epsilons:
            if chosen_mode == 0:
                print(f"Running with epsilon: {epsilon}")
            topk_path = _REPO_ROOT / "algorithms_results" / (
                f"{CHOSEN_DS}_BruteForceRuleMining_top10_delta_{delta}_eps_{epsilon}.csv"
            )
            kwargs = dict(
                mode=chosen_mode,
                algorithm=None,
                df=mining_df,
                treatment_col=TREATMENT_COL,
                tgtO=tgtO,
                delta=delta,
                epsilon=epsilon,
                intervention_attr=intervention_attr,
                intervention_value=intervention_value,
                total_walks=20000,
                top_k=10,
                alpha=0.5,
                max_depth=5,
                output_topk_path=str(topk_path),
            )
            run_single_execution(
                brute_force_rule_mining_calc_utility_for_subgroups,
                kwargs,
                "BruteForceRuleMining",
                chosen_mode,
                intervention_attr,
                intervention_value,
                delta,
                epsilon,
                0,
                0,
            )


def process_dataset_dynamic(i, rule, full_df, chosen_mode, chosen_algorithm_name, tgtO, override_df=None,
                            metric_value=None):
    if override_df is not None: full_df = override_df
    cond_dict = rule["condition"]
    c_attr, c_val = list(cond_dict.items())[0]
    treat_dict = rule["treatment"]
    t_attr, t_val = list(treat_dict.items())[0]

    if c_attr not in full_df.columns or t_attr not in full_df.columns: return
    print(f"--- Processing Rule #{i + 1}: {c_attr}={c_val} -> {t_attr}={t_val} ---")

    # Type mismatch protection
    if c_val not in full_df[c_attr].values:
        if str(c_val) in full_df[c_attr].values:
            c_val = str(c_val)
        elif str(c_val).isdigit() and int(c_val) in full_df[c_attr].values:
            c_val = int(c_val)

    try:
        sub_df = full_df[full_df[c_attr] == c_val].copy()
    except KeyError:
        return
    if sub_df.empty: return
    sub_df = sub_df.drop(columns=[c_attr])
    sub_df[TREATMENT_COL] = (sub_df[t_attr] == t_val).astype(int)
    if t_attr in sub_df.columns: sub_df = sub_df.drop(columns=[t_attr])
    if sub_df[TREATMENT_COL].sum() == 0: return

    if CHOSEN_DS != "acs":
        sub_df_encoded = encode_dataframe_local(sub_df)
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
        sub_df_encoded[tgtO] = pd.to_numeric(sub_df_encoded[tgtO], errors='coerce')
    else:
        sub_df_encoded = sub_df.copy()
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))

    print(f"   ℹ️  Working Dataset Size: {len(sub_df_encoded)} rows x {len(sub_df_encoded.columns)} cols")
    with timer() as attr_timer:
        attr_vals = {col: sorted(sub_df_encoded[col].dropna().unique()) for col in sub_df_encoded.columns if
                     col not in [TREATMENT_COL, tgtO]}
    attr_time = attr_timer()

    for delta in DELTAS:
        curr_delta = delta
        if chosen_mode == 2 and metric_value is not None:
            curr_delta = int(delta * metric_value)
            if curr_delta < 2: curr_delta = 2
        if chosen_mode in [2, 3]: print(f"   [PARAMS] Delta: {curr_delta} (Base {delta}) | Data Metric: {metric_value}")

        if len(sub_df_encoded) < curr_delta: continue
        print(f"Running for delta: {curr_delta}")

        attr_pass = 0

        if chosen_algorithm_name == "RW":
            rw_all_runs_data = []
            for run_num in range(NUM_RW_RUNS):
                print(f"--- Run number: {run_num} ---")
                run_stats = run_experiments(chosen_mode, "RW", curr_delta, sub_df_encoded, tgtO, attr_vals, cond_dict,
                                            treat_dict, i, attr_pass, metric_value=metric_value)
                rw_all_runs_data.append(run_stats)

            if RUN_RANDOM_BASELINE and "Random" in ALGORITHM_NAMES:
                print(f"\n\033[95m>>> Starting Random Baseline Sequence <<<\033[0m")
                for run_idx, run_results in enumerate(rw_all_runs_data):
                    print(f"--- Matching Random Run #{run_idx} ---")
                    for (eps, count) in run_results:
                        if count > 0:
                            print(f"Triggering Random for epsilon {eps} with n={count}")
                            run_experiments(chosen_mode, "Random", curr_delta, sub_df_encoded, tgtO, attr_vals,
                                            cond_dict, treat_dict, i, attr_pass, force_n_subgroups=count,
                                            override_epsilons=[eps], metric_value=metric_value)
                        else:
                            print(f"Skipping Random for epsilon {eps} (RW checked 0 subgroups).")
        else:
            run_experiments(chosen_mode, chosen_algorithm_name, curr_delta, sub_df_encoded, tgtO, attr_vals, cond_dict,
                            treat_dict, i, attr_pass, metric_value=metric_value)


def clean_results_files(mode):
    results_dir_graphs = _REPO_ROOT / "graphs"
    results_dir_graphs.mkdir(exist_ok=True)

    # Define file mapping for all modes
    mode_files = {
        0: results_dir_graphs / f"{CHOSEN_DS}_homogeneity_results.xlsx",
        1: results_dir_graphs / f"{CHOSEN_DS}_algorithms_time.csv",
        2: results_dir_graphs / f"{CHOSEN_DS}_scalability_rows.csv",
        3: results_dir_graphs / f"{CHOSEN_DS}_scalability_attributes.csv"
    }

    target_file = mode_files.get(mode)

    if '-d' not in sys.argv:
        if target_file and target_file.exists():
            target_file.unlink()
            print(f"🧹 Cleaned previous results: {target_file.name}")
    else:
        print(f"⚠️  Append mode active (-d flag): {target_file.name if target_file else 'N/A'} preserved.")


def main():
    tgtO = TARGET_COLUMN_NAME
    with open(RULES_FILE, "r") as f:
        rules_list = [json.loads(line) for line in f]
    full_df = pd.read_csv(FULL_DATASET_PATH)
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]
    full_df = full_df[~full_df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    try:
        chosen_mode = int(input(f"Choose mode {list(enumerate(MODES))}: \n"))
    except:
        chosen_mode = 0
    clean_results_files(chosen_mode)
    main_loop_algos = [algo for algo in ALGORITHM_NAMES if algo != "Random"]
    for algo_name in main_loop_algos:
        if algo_name == "EpsGreedyRuleMining":
            run_eps_greedy_rule_mining_experiment(full_df, chosen_mode, tgtO)
        elif algo_name == "BruteForceRuleMining":
            run_bruteforce_rule_mining_experiment(full_df, chosen_mode, tgtO)
        else:
            for i, rule in enumerate(rules_list):
                process_dataset_dynamic(i, rule, full_df, chosen_mode, algo_name, tgtO)


if __name__ == "__main__":
    main()