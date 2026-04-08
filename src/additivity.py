"""Compute and analyze LLC additivity defects and ratios.

The additivity defect for a pair is:
    delta = lambda(T1) + lambda(T2) - lambda(T1 ∪ T2)

The additivity ratio for a pair is:
    rho = lambda(T1 ∪ T2) / (lambda(T1) + lambda(T2))

For independent subtasks, delta ≈ 0 and rho ≈ 1.

These generalize to triplets:
    delta = lambda(T1) + lambda(T2) + lambda(T3) - lambda(T1 ∪ T2 ∪ T3)
    rho   = lambda(T1 ∪ T2 ∪ T3) / (lambda(T1) + lambda(T2) + lambda(T3))
"""

import math
from itertools import combinations
from typing import Any, Dict, List, Tuple

import pandas as pd


def compute_additivity_defect(
    llc_results: Dict[str, Dict[str, Any]],
    subtask_triples: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """Compute additivity defects and ratios for given subtask pairs.

    Args:
        llc_results: Dict mapping subtask name to LLC estimation result dict
            (must contain 'llc_mean' and 'llc_std' keys).
        subtask_triples: List of (T1_name, T2_name, T_joint_name) tuples.
            T_joint is the data union T1 ∪ T2.

    Returns:
        DataFrame with columns: T1, T2, T_joint, llc_T1, llc_T2, llc_joint,
        delta, delta_std, delta_zscore, ratio, ratio_std.
    """
    rows = []
    for t1_name, t2_name, t_joint_name in subtask_triples:
        r1 = llc_results[t1_name]
        r2 = llc_results[t2_name]
        r_joint = llc_results[t_joint_name]

        llc_t1 = r1["llc_mean"]
        llc_t2 = r2["llc_mean"]
        llc_joint = r_joint["llc_mean"]

        std_t1 = r1["llc_std"]
        std_t2 = r2["llc_std"]
        std_joint = r_joint["llc_std"]

        sum_parts = llc_t1 + llc_t2
        delta = sum_parts - llc_joint
        delta_std = math.sqrt(std_t1**2 + std_t2**2 + std_joint**2)
        delta_zscore = delta / delta_std if delta_std > 0 else float("inf")

        # Ratio: joint / sum_of_parts. For independent tasks, ≈ 1.
        ratio = llc_joint / sum_parts if sum_parts > 0 else float("nan")
        # Error propagation for ratio = f/g: sigma_r = |r| * sqrt((sf/f)^2 + (sg/g)^2)
        sum_parts_std = math.sqrt(std_t1**2 + std_t2**2)
        if sum_parts > 0 and llc_joint != 0:
            ratio_std = abs(ratio) * math.sqrt(
                (std_joint / llc_joint) ** 2 + (sum_parts_std / sum_parts) ** 2
            )
        else:
            ratio_std = float("nan")

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
            "ratio": ratio,
            "ratio_std": ratio_std,
        })

    return pd.DataFrame(rows)


def compute_triplet_additivity(
    llc_results: Dict[str, Dict[str, Any]],
    triplets: List[Tuple[str, str, str, str]],
) -> pd.DataFrame:
    """Compute additivity defects and ratios for triplets of tasks.

    Args:
        llc_results: Dict mapping subtask name to LLC result.
        triplets: List of (T1, T2, T3, T_joint) tuples, where T_joint is
            the data union of all three.

    Returns:
        DataFrame with columns: T1, T2, T3, T_joint, llc_T1, llc_T2, llc_T3,
        llc_joint, sum_parts, delta, delta_std, delta_zscore, ratio, ratio_std.
    """
    rows = []
    for t1_name, t2_name, t3_name, t_joint_name in triplets:
        r1 = llc_results[t1_name]
        r2 = llc_results[t2_name]
        r3 = llc_results[t3_name]
        r_joint = llc_results[t_joint_name]

        llcs = [r1["llc_mean"], r2["llc_mean"], r3["llc_mean"]]
        stds = [r1["llc_std"], r2["llc_std"], r3["llc_std"]]
        llc_joint = r_joint["llc_mean"]
        std_joint = r_joint["llc_std"]

        sum_parts = sum(llcs)
        sum_parts_std = math.sqrt(sum(s**2 for s in stds))

        delta = sum_parts - llc_joint
        delta_std = math.sqrt(sum_parts_std**2 + std_joint**2)
        delta_zscore = delta / delta_std if delta_std > 0 else float("inf")

        ratio = llc_joint / sum_parts if sum_parts > 0 else float("nan")
        if sum_parts > 0 and llc_joint != 0:
            ratio_std = abs(ratio) * math.sqrt(
                (std_joint / llc_joint) ** 2 + (sum_parts_std / sum_parts) ** 2
            )
        else:
            ratio_std = float("nan")

        rows.append({
            "T1": t1_name,
            "T2": t2_name,
            "T3": t3_name,
            "T_joint": t_joint_name,
            "llc_T1": llcs[0],
            "llc_T2": llcs[1],
            "llc_T3": llcs[2],
            "llc_joint": llc_joint,
            "sum_parts": sum_parts,
            "delta": delta,
            "delta_std": delta_std,
            "delta_zscore": delta_zscore,
            "ratio": ratio,
            "ratio_std": ratio_std,
        })

    return pd.DataFrame(rows)


def compute_full_additivity_defect(
    llc_results: Dict[str, Dict[str, Any]],
    atomic_names: List[str],
    composite_name: str,
) -> Dict[str, Any]:
    """Compute the full additivity defect: sum of atomics vs composite.

    delta = sum_i lambda(T_i) - lambda(T_composite)
    ratio = lambda(T_composite) / sum_i lambda(T_i)

    Args:
        llc_results: LLC results dict.
        atomic_names: List of atomic subtask names.
        composite_name: Name of the composite/joint data restriction.

    Returns:
        Dict with sum_atomic_llc, composite_llc, delta, delta_std,
        delta_zscore, ratio, ratio_std.
    """
    sum_llc = 0.0
    sum_var = 0.0
    for name in atomic_names:
        r = llc_results[name]
        sum_llc += r["llc_mean"]
        sum_var += r["llc_std"] ** 2

    r_comp = llc_results[composite_name]
    composite_llc = r_comp["llc_mean"]
    sum_std = math.sqrt(sum_var)

    delta = sum_llc - composite_llc
    delta_std = math.sqrt(sum_var + r_comp["llc_std"] ** 2)
    delta_zscore = delta / delta_std if delta_std > 0 else float("inf")

    ratio = composite_llc / sum_llc if sum_llc > 0 else float("nan")
    if sum_llc > 0 and composite_llc != 0:
        ratio_std = abs(ratio) * math.sqrt(
            (r_comp["llc_std"] / composite_llc) ** 2 + (sum_std / sum_llc) ** 2
        )
    else:
        ratio_std = float("nan")

    return {
        "sum_atomic_llc": sum_llc,
        "composite_llc": composite_llc,
        "delta": delta,
        "delta_std": delta_std,
        "delta_zscore": delta_zscore,
        "ratio": ratio,
        "ratio_std": ratio_std,
        "atomic_names": atomic_names,
        "composite_name": composite_name,
    }


def enumerate_pair_triples(
    code_names: List[str],
    llc_results: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Auto-generate (T1, T2, T1∪T2) triples from available LLC results.

    For each pair of code names, checks if the union key exists in llc_results.
    """
    available = set(llc_results.keys())
    triples = []
    for a, b in combinations(sorted(code_names), 2):
        union_key = f"{a}\u222a{b}"
        if union_key in available:
            triples.append((a, b, union_key))
    return triples


def enumerate_triplet_quads(
    code_names: List[str],
    llc_results: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    """Auto-generate (T1, T2, T3, T1∪T2∪T3) quads from available LLC results.

    For each triple of code names, checks if the 3-way union key exists.
    """
    available = set(llc_results.keys())
    quads = []
    for a, b, c in combinations(sorted(code_names), 3):
        union_key = f"{a}\u222a{b}\u222a{c}"
        if union_key in available:
            quads.append((a, b, c, union_key))
    return quads


def summarize_results(
    llc_results: Dict[str, Dict[str, Any]],
    defect_df: pd.DataFrame,
) -> str:
    """Format a human-readable summary of LLC and additivity results."""
    lines = ["=" * 60, "LLC Estimation Results", "=" * 60]

    for name, r in sorted(llc_results.items()):
        lines.append(f"  {name:>20s}: LLC = {r['llc_mean']:.4f} \u00b1 {r['llc_std']:.4f}")

    lines.extend(["", "=" * 60, "Additivity Defects", "=" * 60])
    for _, row in defect_df.iterrows():
        t_joint = row.get("T_joint", "")
        parts = [row["T1"], row["T2"]]
        if "T3" in row and pd.notna(row.get("T3")):
            parts.append(row["T3"])
        parts_str = " + ".join(parts)
        lines.append(
            f"  {parts_str} vs {t_joint}: "
            f"\u03b4 = {row['delta']:.4f} \u00b1 {row['delta_std']:.4f} "
            f"(z = {row['delta_zscore']:.2f}), "
            f"\u03c1 = {row['ratio']:.4f} \u00b1 {row['ratio_std']:.4f}"
        )

    return "\n".join(lines)
