"""Compute and analyze LLC additivity defects.

The additivity defect is:
    delta = lambda_hat(T1) + lambda_hat(T2) - lambda_hat(T1 ∪ T2)

where lambda_hat is the estimated LLC and T1, T2 are subtasks.
"""

import math
from typing import Any, Dict, List, Tuple

import pandas as pd


def compute_additivity_defect(
    llc_results: Dict[str, Dict[str, Any]],
    subtask_triples: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """Compute additivity defects for given subtask triples.

    Args:
        llc_results: Dict mapping subtask name to LLC estimation result dict
            (must contain 'llc_mean' and 'llc_std' keys).
        subtask_triples: List of (T1_name, T2_name, T_joint_name) tuples.
            T_joint is the union T1 ∪ T2 as a data restriction.

    Returns:
        DataFrame with columns: T1, T2, T_joint, llc_T1, llc_T2, llc_joint,
        delta, delta_std, delta_zscore.
    """
    rows = []
    for t1_name, t2_name, t_joint_name in subtask_triples:
        r1 = llc_results[t1_name]
        r2 = llc_results[t2_name]
        r_joint = llc_results[t_joint_name]

        llc_t1 = r1["llc_mean"]
        llc_t2 = r2["llc_mean"]
        llc_joint = r_joint["llc_mean"]

        delta = llc_t1 + llc_t2 - llc_joint

        # Error propagation (assuming independence)
        std_t1 = r1["llc_std"]
        std_t2 = r2["llc_std"]
        std_joint = r_joint["llc_std"]
        delta_std = math.sqrt(std_t1**2 + std_t2**2 + std_joint**2)

        delta_zscore = delta / delta_std if delta_std > 0 else float("inf")

        rows.append({
            "T1": t1_name,
            "T2": t2_name,
            "T_joint": t_joint_name,
            "llc_T1": llc_t1,
            "llc_T2": llc_t2,
            "llc_joint": llc_joint,
            "delta": delta,
            "delta_std": delta_std,
            "delta_zscore": delta_zscore,
        })

    return pd.DataFrame(rows)


def compute_full_additivity_defect(
    llc_results: Dict[str, Dict[str, Any]],
    atomic_names: List[str],
    composite_name: str,
) -> Dict[str, Any]:
    """Compute the full additivity defect: sum of atomics vs composite.

    delta = sum_i lambda_hat(T_i) - lambda_hat(T_composite)

    Args:
        llc_results: LLC results dict.
        atomic_names: List of atomic subtask names (e.g. ['{0}', '{1}', '{2}', '{3}']).
        composite_name: Name of the composite/joint data restriction.

    Returns:
        Dict with sum_atomic_llc, composite_llc, delta, delta_std, delta_zscore.
    """
    sum_llc = 0.0
    sum_var = 0.0
    for name in atomic_names:
        r = llc_results[name]
        sum_llc += r["llc_mean"]
        sum_var += r["llc_std"] ** 2

    r_comp = llc_results[composite_name]
    composite_llc = r_comp["llc_mean"]

    delta = sum_llc - composite_llc
    delta_std = math.sqrt(sum_var + r_comp["llc_std"] ** 2)
    delta_zscore = delta / delta_std if delta_std > 0 else float("inf")

    return {
        "sum_atomic_llc": sum_llc,
        "composite_llc": composite_llc,
        "delta": delta,
        "delta_std": delta_std,
        "delta_zscore": delta_zscore,
        "atomic_names": atomic_names,
        "composite_name": composite_name,
    }


def summarize_results(
    llc_results: Dict[str, Dict[str, Any]],
    defect_df: pd.DataFrame,
) -> str:
    """Format a human-readable summary of LLC and additivity results."""
    lines = ["=" * 60, "LLC Estimation Results", "=" * 60]

    for name, r in sorted(llc_results.items()):
        lines.append(f"  {name:>20s}: LLC = {r['llc_mean']:.4f} ± {r['llc_std']:.4f}")

    lines.extend(["", "=" * 60, "Additivity Defects", "=" * 60])
    for _, row in defect_df.iterrows():
        lines.append(
            f"  {row['T1']} + {row['T2']} - {row['T_joint']}: "
            f"δ = {row['delta']:.4f} ± {row['delta_std']:.4f} "
            f"(z = {row['delta_zscore']:.2f})"
        )

    return "\n".join(lines)
