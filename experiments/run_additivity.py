"""Entry point: Compute and display additivity defects from LLC estimates.

Usage:
    python experiments/run_additivity.py --model-dir results/run0 --config config/sweeps/composite_curriculum.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.additivity import (
    compute_additivity_defect,
    compute_full_additivity_defect,
    summarize_results,
)
from src.utils import load_config


def plot_defects(defect_df, save_path: str):
    """Bar plot of additivity defects with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{row['T1']}+{row['T2']}" for _, row in defect_df.iterrows()]
    deltas = defect_df["delta"].values
    errors = defect_df["delta_std"].values

    colors = ["#2196F3" if d >= 0 else "#F44336" for d in deltas]
    ax.bar(range(len(labels)), deltas, yerr=errors, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Additivity defect δ")
    ax.set_title("LLC Additivity Defects")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute LLC additivity defects")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing llc_results.pkl")
    parser.add_argument("--config", type=str, default=None,
                        help="Config with additivity triples definition")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load LLC results
    llc_path = model_dir / "llc_results.pkl"
    with open(llc_path, "rb") as f:
        llc_results = pickle.load(f)

    # Load config for additivity definitions
    config_path = args.config or str(model_dir / "config.yaml")
    config = load_config(config_path)
    additivity_cfg = config.get("additivity", {})

    # Compute pairwise defects
    triples = additivity_cfg.get("triples", [])
    triples = [tuple(t) for t in triples]

    if triples:
        defect_df = compute_additivity_defect(llc_results, triples)
        print(summarize_results(llc_results, defect_df))
        plot_defects(defect_df, str(model_dir / "additivity_defects.png"))

        # Save
        defect_df.to_csv(model_dir / "additivity_defects.csv", index=False)

    # Compute full composite defect if defined
    full_cfg = additivity_cfg.get("full", {})
    if full_cfg:
        atomics = full_cfg["atomics"]
        composite = full_cfg["composite"]
        full_result = compute_full_additivity_defect(llc_results, atomics, composite)
        print(f"\n{'='*50}")
        print("Full Composite Additivity Defect:")
        print(f"  Σ λ(atomic_i) = {full_result['sum_atomic_llc']:.4f}")
        print(f"  λ(composite)  = {full_result['composite_llc']:.4f}")
        print(f"  δ = {full_result['delta']:.4f} ± {full_result['delta_std']:.4f} "
              f"(z = {full_result['delta_zscore']:.2f})")

        with open(model_dir / "full_additivity_defect.pkl", "wb") as f:
            pickle.dump(full_result, f)


if __name__ == "__main__":
    main()
