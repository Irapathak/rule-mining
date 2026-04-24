"""
Binary Search Algorithm for Problem 2: Finding Largest Delta Breaking Homogeneity.

Given a rule r and fixed epsilon, finds the largest minimum subgroup size delta
for which the rule remains heterogeneous (has violations).

Monotonicity Property: If a rule is heterogeneous at delta, it remains 
heterogeneous for all delta' < delta (smaller thresholds permit more subgroups).
"""
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

# Add project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'algorithms'))

from ATE_update import calculate_ate_safe
from brute_force_algorithm import calc_utility_for_subgroups as brute_force_oracle

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']


def oracle_is_heterogeneous(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    utility_all: float
) -> Tuple[bool, int, Optional[Dict]]:
    """
    Oracle function that checks if rule is heterogeneous at given delta.
    Uses FPGrowth algorithm for subgroup enumeration.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        delta: Minimum subgroup size threshold
        epsilon: Homogeneity threshold
        utility_all: Overall ATE
        
    Returns:
        Tuple of (is_heterogeneous, num_subgroups_checked, violation_info)
        - is_heterogeneous: True if violation found (heterogeneous)
        - num_subgroups_checked: Number of subgroups evaluated
        - violation_info: Dict with violating subgroup info (None if homogeneous)
    """
    # Call FPGrowth with mode=0 (homogeneity check)
    result = brute_force_oracle(
        mode=0,
        algorithm=fpgrowth,
        df=df,
        treatment_col=treatment_col,
        tgtO=outcome_col,
        delta=delta,
        epsilon=epsilon,
        utility_all=utility_all
    )
    
    # Parse result based on return signature
    if isinstance(result, tuple):
        if len(result) >= 5:
            # (homogeneity_status, count, enum_time, iter_time, violation_info) format
            is_homogeneous = result[0]
            num_checked = result[1]
            violation_info = result[4]  # violation_info is 5th element
            # Return True if heterogeneous (NOT homogeneous)
            return (not is_homogeneous, num_checked, violation_info)
        elif len(result) >= 2:
            # Old format without violation_info
            is_homogeneous = result[0]
            num_checked = result[1]
            return (not is_homogeneous, num_checked, None)
        elif len(result) == 1:
            is_homogeneous = result[0]
            return (not is_homogeneous, 0, None)
    
    # Fallback: assume boolean result
    is_homogeneous = bool(result)
    return (not is_homogeneous, 0, None)


def find_largest_delta_breaking_homogeneity(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    epsilon: float,
    delta_min: int = 100,
    delta_max: int = 10000,
    verbose: bool = True
) -> Tuple[Optional[int], int, Optional[Dict], float]:
    """
    Binary search to find the largest delta where the rule is heterogeneous.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name  
        epsilon: Fixed homogeneity threshold
        delta_min: Minimum delta to search
        delta_max: Maximum delta to search
        verbose: Print progress messages
        
    Returns:
        Tuple of (largest_delta, total_oracle_calls, violation_info, utility_all)
        - largest_delta: Largest delta with violation, None if always homogeneous
        - total_oracle_calls: Total number of oracle invocations
        - violation_info: Dict with info about the violating subgroup (None if homogeneous)
        - utility_all: Overall population ATE
    """
    if verbose:
        print("="*70)
        print(f"FINDING LARGEST DELTA BREAKING HOMOGENEITY")
        print(f"Fixed epsilon: {epsilon}")
        print(f"Search range: [{delta_min}, {delta_max}]")
        print("="*70)
    
    # Calculate overall ATE once
    utility_all = calculate_ate_safe(df, treatment_col, outcome_col, delta_min)
    
    low = delta_min
    high = delta_max
    answer = None  # Largest delta with violation
    answer_violation_info = None  # Info about the violating subgroup
    total_oracle_calls = 0
    
    while low <= high:
        mid = (low + high) // 2
        total_oracle_calls += 1
        
        if verbose:
            print(f"\nIteration {total_oracle_calls}:")
            print(f"  Testing delta = {mid} (range: [{low}, {high}])")
        
        # Check if heterogeneous at this delta
        is_heterogeneous, num_checked, violation_info = oracle_is_heterogeneous(
            df, treatment_col, outcome_col, mid, epsilon, utility_all
        )
        
        if verbose:
            status = "HETEROGENEOUS ✗" if is_heterogeneous else "HOMOGENEOUS ✓"
            print(f"  Result: {status} (checked {num_checked} subgroups)")
        
        if is_heterogeneous:
            # Violation found! Save as candidate and search for larger delta
            answer = mid
            answer_violation_info = violation_info  # Save the violation info
            if verbose:
                print(f"  → Violation found! Saving delta={mid}, searching larger...")
            low = mid + 1
        else:
            # No violation, need smaller delta for violations
            if verbose:
                print(f"  → No violation, searching smaller...")
            high = mid - 1
    
    if verbose:
        print("\n" + "="*70)
        if answer is not None:
            print(f"RESULT: Largest delta with violation = {answer}")
            if answer_violation_info:
                print(f"  Violating subgroup: {answer_violation_info['subgroup']}")
                print(f"  Subgroup size: {answer_violation_info['size']}")
                print(f"  Subgroup utility: {answer_violation_info['utility']:.2f}")
                print(f"  Population utility: {utility_all:.2f}")
                print(f"  Difference: {answer_violation_info['abs_diff']:.2f} (threshold: {epsilon})")
        else:
            print(f"RESULT: No violations found in range [{delta_min}, {delta_max}]")
        print(f"Total oracle calls: {total_oracle_calls}")
        print("="*70)
    
    return answer, total_oracle_calls, answer_violation_info, utility_all


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find largest delta breaking homogeneity')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--outcome', type=str, default='ConvertedSalary', help='Outcome column')
    parser.add_argument('--epsilon', type=float, required=True, help='Fixed epsilon threshold')
    parser.add_argument('--delta_min', type=int, default=100, help='Minimum delta')
    parser.add_argument('--delta_max', type=int, default=10000, help='Maximum delta')
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Run algorithm
    largest_delta, oracle_calls, violation_info, utility_all = find_largest_delta_breaking_homogeneity(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=args.outcome,
        epsilon=args.epsilon,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        verbose=True
    )
    
    print(f"\nFinal Answer: {largest_delta}")
    print(f"Efficiency: Found answer in {oracle_calls} oracle calls")
    if violation_info:
        print(f"\nViolating Subgroup Details:")
        print(f"  Subgroup: {violation_info['subgroup']}")
        print(f"  Size: {violation_info['size']}")
        print(f"  Utility: {violation_info['utility']:.2f}")
        print(f"  Population Utility: {utility_all:.2f}")
        print(f"  |Difference|: {violation_info['abs_diff']:.2f}")

