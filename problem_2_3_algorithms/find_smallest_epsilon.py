"""
Binary Search Algorithm for Problem 3: Finding Smallest Epsilon Achieving Homogeneity.

Given a rule r and fixed delta, finds the smallest epsilon threshold
for which the rule becomes homogeneous (no violations).

Monotonicity Property: If a rule is homogeneous at epsilon, it remains 
homogeneous for all epsilon' > epsilon (upward-closure).
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


def oracle_is_homogeneous(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    utility_all: float
) -> Tuple[bool, int, Optional[Dict]]:
    """
    Oracle function that checks if rule is homogeneous at given epsilon.
    Uses FPGrowth algorithm for subgroup enumeration.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        delta: Minimum subgroup size threshold
        epsilon: Homogeneity threshold
        utility_all: Overall ATE
        
    Returns:
        Tuple of (is_homogeneous, num_subgroups_checked, violation_info)
        - is_homogeneous: True if no violations (homogeneous)
        - num_subgroups_checked: Number of subgroups evaluated
        - violation_info: Dict with violating subgroup info (None if homogeneous)
    """
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
    
    # Parse result
    if isinstance(result, tuple):
        if len(result) >= 5:
            # (homogeneity_status, count, enum_time, iter_time, violation_info)
            is_homogeneous = result[0]
            num_checked = result[1]
            violation_info = result[4]
            return (is_homogeneous, num_checked, violation_info)
        elif len(result) >= 2:
            # Old format
            is_homogeneous = result[0]
            num_checked = result[1]
            return (is_homogeneous, num_checked, None)
    
    # Fallback
    is_homogeneous = bool(result)
    return (is_homogeneous, 0, None)


def find_smallest_epsilon_achieving_homogeneity(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon_0: Optional[float] = None,
    epsilon_max_cap: float = 1_000_000_000.0,
    verbose: bool = True
) -> Tuple[Optional[float], int, Optional[Dict], float]:
    """
    Two-Phase algorithm to find the smallest epsilon where rule is homogeneous.

    Phase 1 (Bracketing): Exponential search to find upper bound ε_max
        - Initialize ε_low = 0 (always)
        - Start with ε₀ (default: 1000)
        - Grow exponentially: ε₀, 2ε₀, 4ε₀, ... until ORACLE returns homogeneous
        - This finds ε_high where rule is homogeneous
    
    Phase 2 (Binary Search): Standard binary search on [0, ε_high]
        - Refine to find exact smallest epsilon achieving homogeneity
        - Uses O(log₂(ε_high)) oracle calls
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name  
        delta: Fixed minimum subgroup size
        epsilon_0: Initial ε₀ for Phase 1 exponential growth (default: 1000)
                   If None, automatically set to 1000
        epsilon_max_cap: Upper bound cap to prevent infinite search
        verbose: Print progress messages
        
    Returns:
        Tuple of (smallest_epsilon, total_oracle_calls, violation_info, utility_all, phase1_calls, phase2_calls)
        - smallest_epsilon: Smallest epsilon achieving homogeneity (None if not found within cap)
        - total_oracle_calls: Total oracle invocations (Phase 1 + Phase 2)
        - violation_info_at_max_or_last_violation: Witness subgroup info for last violation
        - utility_all: Overall population ATE
        - phase1_calls: Oracle calls in Phase 1 (exponential bracketing)
        - phase2_calls: Oracle calls in Phase 2 (binary search)
    """
    if verbose:
        print("="*70)
        print("TWO-PHASE ALGORITHM: Find Smallest Epsilon Achieving Homogeneity")
        print(f"Fixed delta: {delta}")
        print(f"Maximum epsilon cap: {epsilon_max_cap:,.0f}")
        print("="*70)
    
    # Calculate overall ATE once
    utility_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    
    total_oracle_calls = 0
    last_violation_info = None  # Track the most recent violation

    # ===== PHASE 1: EXPONENTIAL BRACKETING =====
    # Goal: Find ε_max (upper bound where rule is homogeneous)
    # Method: Exponential search starting from ε₀
    #
    # Initialize:
    #   - ε_low = 0 (always, by definition)
    #   - Test ε_high ∈ {ε₀, 2ε₀, 4ε₀, ...} until ORACLE returns homogeneous
    #
    # This adapts to the actual scale of the solution, much more efficient
    # than binary search over [0, very_large_number]
    
    # Set initial epsilon (ε₀) for Phase 1
    if epsilon_0 is None:
        eps_0 = 1000  # Default starting point
    else:
        eps_0 = max(1, int(epsilon_0))
    
    epsilon_low = 0  # Always 0
    epsilon_high = eps_0  # Start testing from ε₀
    epsilon_cap = int(epsilon_max_cap)

    if epsilon_high > epsilon_cap:
        epsilon_high = epsilon_cap

    if verbose:
        print(f"\n📍 PHASE 1: EXPONENTIAL BRACKETING (Finding ε_max)")
        print(f"   ε_low = 0 (fixed)")
        print(f"   Starting ε₀ = {eps_0:,}")
        print(f"   Testing: ε₀, 2ε₀, 4ε₀, ... until homogeneous")
        print(f"   Search cap: {epsilon_cap:,}")
        print("-" * 70)

    # Exponential growth until homogeneity (or we hit cap)
    phase1_iteration = 0
    while True:
        phase1_iteration += 1
        total_oracle_calls += 1
        
        if verbose:
            print(f"   Iteration {phase1_iteration}: Testing ε = {epsilon_high:,} ...", end=" ")
        
        is_homogeneous, _, violation_info = oracle_is_homogeneous(
            df, treatment_col, outcome_col, delta, float(epsilon_high), utility_all
        )

        if is_homogeneous:
            if verbose:
                print("✅ HOMOGENEOUS (found ε_max)")
            break

        if verbose:
            print("✗ Violation found")

        last_violation_info = violation_info

        if epsilon_high >= epsilon_cap:
            if verbose:
                print(f"\n⚠️  WARNING: No homogeneity found even at cap = {epsilon_cap:,.0f}")
                print(f"   Consider increasing --epsilon_max parameter")
            # If we didn't find anything, all calls were in Phase 1
            return None, total_oracle_calls, last_violation_info, utility_all, total_oracle_calls, 0

        # Double the epsilon value (exponential growth)
        next_epsilon = epsilon_high * 2
        epsilon_high = min(epsilon_cap, next_epsilon)
        
        if verbose:
            print(f"   → Growing to ε = {epsilon_high:,}")

    # epsilon_high is now the ε_max we found (homogeneous)
    # epsilon_low stays at 0
    if verbose:
        print(f"\n✅ Phase 1 Complete:")
        print(f"   Found ε_max = {epsilon_high:,} (homogeneous)")
        print(f"   Searching interval: [0, {epsilon_high:,}]")
        print(f"   Oracle calls in Phase 1: {total_oracle_calls}")
        print(f"\n📍 PHASE 2: BINARY SEARCH")
        print(f"   Finding smallest ε* in [0, {epsilon_high:,}]")
        print("-" * 70)
    
    phase2_iteration = 0
    phase2_start_calls = total_oracle_calls
    
    while epsilon_low < epsilon_high:
        epsilon_mid = (epsilon_low + epsilon_high) // 2
        phase2_iteration += 1
        total_oracle_calls += 1
        
        if verbose:
            print(f"\n   Iteration {phase2_iteration}:")
            print(f"      Range: [{epsilon_low:,.0f}, {epsilon_high:,.0f}]")
            print(f"      Testing ε = {epsilon_mid:,.0f} ...", end=" ")
        
        is_homogeneous, num_checked, violation_info = oracle_is_homogeneous(
            df, treatment_col, outcome_col, delta, epsilon_mid, utility_all
        )
        
        if verbose:
            status = "✅ HOMOGENEOUS" if is_homogeneous else "✗ Violation"
            print(f"{status}")
        
        if is_homogeneous:
            # Can potentially go lower
            epsilon_high = epsilon_mid
            if verbose:
                print(f"      → Update: ε_high = {epsilon_high:,.0f}")
        else:
            # Need higher epsilon - track the violation
            last_violation_info = violation_info
            epsilon_low = epsilon_mid + 1
            if verbose:
                print(f"      → Update: ε_low = {epsilon_low:,.0f}")
    
    smallest_epsilon = epsilon_high
    phase2_calls = total_oracle_calls - phase2_start_calls
    phase1_calls = phase2_start_calls
    
    if verbose:
        print(f"\n✅ Phase 2 Complete: Found ε* = {smallest_epsilon:,.0f}")
        print(f"   Oracle calls in Phase 2: {phase2_calls}")
        print("\n" + "="*70)
        print(f"🎯 FINAL RESULT: ε* = {smallest_epsilon:,.0f}")
        print("="*70)
        if last_violation_info:
            print(f"  ✓ Smallest ε (homogeneous):  {smallest_epsilon:,.0f}")
            print(f"  ✗ Largest ε (heterogeneous): {smallest_epsilon - 1:,.0f}")
            print(f"\n  📊 Violating subgroup at ε = {smallest_epsilon-1:,.0f}:")
            print(f"     Subgroup: {last_violation_info['subgroup']}")
            print(f"     Size: {last_violation_info['size']}")
            print(f"     Subgroup ATE: {last_violation_info['utility']:.2f}")
            print(f"     Population ATE: {utility_all:.2f}")
            print(f"     |Difference|: {last_violation_info['abs_diff']:.2f}")
        print(f"\n  ⚡ Efficiency:")
        print(f"     Phase 1 (Bracketing): {phase1_calls} oracle calls")
        print(f"     Phase 2 (Binary Search): {phase2_calls} oracle calls")
        print(f"     Total: {total_oracle_calls} oracle calls")
        print("="*70)
    
    return smallest_epsilon, total_oracle_calls, last_violation_info, utility_all, phase1_calls, phase2_calls


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find smallest epsilon achieving homogeneity (Two-Phase)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--outcome', type=str, default='ConvertedSalary', help='Outcome column')
    parser.add_argument('--delta', type=int, required=True, help='Fixed delta threshold')
    parser.add_argument('--epsilon_0', type=float, default=1000.0, help='Initial ε₀ for Phase 1 (default: 1000)')
    parser.add_argument('--epsilon_max', type=float, default=1_000_000_000.0, help='Maximum epsilon cap')
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Run algorithm
    smallest_epsilon, oracle_calls, violation_info, utility_all, phase1_calls, phase2_calls = find_smallest_epsilon_achieving_homogeneity(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=args.outcome,
        delta=args.delta,
        epsilon_0=args.epsilon_0,
        epsilon_max_cap=args.epsilon_max,
        verbose=True
    )
    
    if smallest_epsilon is not None:
        print(f"\n✅ Final Answer: ε* = {smallest_epsilon:,.0f}")
        print(f"   Efficiency: Found in {oracle_calls} oracle calls")
        if violation_info:
            print(f"\n   Last Violation (ε={smallest_epsilon-1}):")
            print(f"     Subgroup: {violation_info['subgroup']}")
            print(f"     |Difference|: {violation_info['abs_diff']:.2f}")
    else:
        print(f"\n❌ No homogeneous epsilon found up to {args.epsilon_max:,.0f}")

