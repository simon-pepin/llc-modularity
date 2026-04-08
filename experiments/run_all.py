"""Full pipeline: train, estimate LLCs, compute additivity defects.

Usage:
    python experiments/run_all.py --config config/default.yaml --save-dir results/full_run
    python experiments/run_all.py --config config/sweeps/composite_curriculum.yaml --seeds 0 1 2 3 4
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.additivity import (
    compute_additivity_defect,
    compute_full_additivity_defect,
    summarize_results,
)
from src.data import (
    code_to_str,
    make_subtask_dataloaders,
    make_subtask_indices,
    make_union_dataloaders,
)
from src.llc_estimation import estimate_subtask_llcs
from src.model import make_mlp
from src.train import train_cmsp
from src.utils import ensure_dir, get_device, load_config, save_config


def run_single_seed(config: dict, save_dir: Path, device: str, verbose: bool = True):
    """Run the full pipeline for a single seed."""
    seed = config["seed"]
    print(f"\n{'#'*60}")
    print(f"# Seed {seed}")
    print(f"{'#'*60}")

    # --- Phase 1: Training ---
    print("\n[Phase 1] Training...")
    train_results = train_cmsp(config, save_dir=str(save_dir), verbose=verbose)
    model = train_results["model"]

    print("Final losses:")
    for name, loss in train_results["final_subtask_losses"].items():
        print(f"  {name}: {loss:.6f}")

    # --- Phase 2: LLC Estimation ---
    print("\n[Phase 2] LLC Estimation...")
    m = config["m"]
    k = config["k"]
    n = config.get("n", m * k)
    subtask_indices = make_subtask_indices(m, k)
    task_codes = config.get("task_codes", [[i] for i in range(m)] + [list(range(m))])
    llc_cfg = config.get("llc", {})
    dtype = torch.float32 if config.get("dtype", "float32") == "float32" else torch.float64

    samples_per_code = llc_cfg.get("samples_per_code", 2000)
    batch_size = llc_cfg.get("batch_size", 256)

    # Per-code dataloaders
    per_code_loaders = make_subtask_dataloaders(
        n=n, m=m, subtask_indices=subtask_indices,
        task_codes=task_codes,
        samples_per_code=samples_per_code,
        batch_size=batch_size,
        device=device, dtype=dtype,
        seed=llc_cfg.get("seed", 42),
    )

    # Union dataloaders
    llc_measurements = config.get("llc_measurements", {})
    unions_cfg = llc_measurements.get("unions", {})
    if unions_cfg:
        union_loaders = make_union_dataloaders(
            n=n, m=m, subtask_indices=subtask_indices,
            code_groups=unions_cfg,
            samples_per_code=samples_per_code,
            batch_size=batch_size,
            device=device, dtype=dtype,
            seed=llc_cfg.get("seed", 42),
        )
        per_code_loaders.update(union_loaders)

    # If no explicit unions defined, create default pairwise unions from atomics
    atomic_codes = [c for c in task_codes if len(c) == 1]
    if not unions_cfg and len(atomic_codes) >= 2:
        # Create union of all atomics
        all_atomic_label = "∪".join(code_to_str([c[0]]) for c in atomic_codes)
        default_unions = {all_atomic_label: atomic_codes}
        # Create pairwise unions
        for i in range(len(atomic_codes)):
            for j in range(i + 1, len(atomic_codes)):
                label = f"{code_to_str(atomic_codes[i])}∪{code_to_str(atomic_codes[j])}"
                default_unions[label] = [atomic_codes[i], atomic_codes[j]]
        union_loaders = make_union_dataloaders(
            n=n, m=m, subtask_indices=subtask_indices,
            code_groups=default_unions,
            samples_per_code=samples_per_code,
            batch_size=batch_size,
            device=device, dtype=dtype,
            seed=llc_cfg.get("seed", 42),
        )
        per_code_loaders.update(union_loaders)

    llc_kwargs = {
        "num_chains": llc_cfg.get("num_chains", 10),
        "num_draws": llc_cfg.get("num_draws", 500),
        "num_burnin_steps": llc_cfg.get("num_burnin_steps", 100),
        "num_steps_bw_draws": llc_cfg.get("num_steps_bw_draws", 1),
        "learning_rate": llc_cfg.get("learning_rate", 1e-4),
        "localization": llc_cfg.get("localization", 0.0),
        "seed": llc_cfg.get("seed", 42),
    }

    llc_results = estimate_subtask_llcs(
        model=model,
        subtask_dataloaders=per_code_loaders,
        device=device,
        verbose=verbose,
        **llc_kwargs,
    )

    with open(save_dir / "llc_results.pkl", "wb") as f:
        pickle.dump(llc_results, f)

    # --- Phase 3: Additivity Defects ---
    print("\n[Phase 3] Additivity Defects...")
    additivity_cfg = config.get("additivity", {})

    # Build triples from config or defaults
    triples = [tuple(t) for t in additivity_cfg.get("triples", [])]

    # Auto-generate pairwise triples if not specified
    if not triples:
        available = set(llc_results.keys())
        for i in range(len(atomic_codes)):
            for j in range(i + 1, len(atomic_codes)):
                t1 = code_to_str(atomic_codes[i])
                t2 = code_to_str(atomic_codes[j])
                t_joint = f"{t1}∪{t2}"
                if t1 in available and t2 in available and t_joint in available:
                    triples.append((t1, t2, t_joint))

    defect_df = None
    if triples:
        defect_df = compute_additivity_defect(llc_results, triples)
        print(summarize_results(llc_results, defect_df))
        defect_df.to_csv(save_dir / "additivity_defects.csv", index=False)

    # Full composite defect
    full_cfg = additivity_cfg.get("full", {})
    full_result = None
    if full_cfg:
        full_result = compute_full_additivity_defect(
            llc_results, full_cfg["atomics"], full_cfg["composite"]
        )
        print(f"\nFull defect: δ = {full_result['delta']:.4f} ± "
              f"{full_result['delta_std']:.4f} (z = {full_result['delta_zscore']:.2f})")
        with open(save_dir / "full_additivity_defect.pkl", "wb") as f:
            pickle.dump(full_result, f)

    return {
        "train_results": {k: v for k, v in train_results.items() if k != "model"},
        "llc_results": llc_results,
        "defect_df": defect_df,
        "full_defect": full_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Full CMSP → LLC → Additivity pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seed list")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or config.get("device") or str(get_device())
    config["device"] = device

    seeds = args.seeds or config.get("seeds", [config.get("seed", 0)])
    base_dir = Path(args.save_dir)

    all_results = {}
    for seed in seeds:
        config["seed"] = seed
        seed_dir = ensure_dir(base_dir / f"seed_{seed}")
        all_results[seed] = run_single_seed(
            config, seed_dir, device, verbose=not args.quiet
        )

    # Save aggregate results
    with open(base_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\n{'='*60}")
    print(f"All results saved to {base_dir}")
    print(f"Seeds run: {seeds}")


if __name__ == "__main__":
    main()
