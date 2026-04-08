"""Entry point: Estimate LLCs on a trained CMSP model.

Usage:
    python experiments/run_llc.py --model-dir results/run0 --config config/default.yaml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import (
    code_to_str,
    make_subtask_dataloaders,
    make_subtask_indices,
    make_union_dataloaders,
)
from src.llc_estimation import estimate_subtask_llcs
from src.model import make_mlp
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description="Estimate LLCs on trained CMSP model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model.pt and config.yaml")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config (default: use model-dir/config.yaml)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--num-draws", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    config_path = args.config or str(model_dir / "config.yaml")
    config = load_config(config_path)
    llc_cfg = config.get("llc", {})

    # Apply overrides
    device = args.device or config.get("device") or str(get_device())
    if args.num_chains is not None:
        llc_cfg["num_chains"] = args.num_chains
    if args.num_draws is not None:
        llc_cfg["num_draws"] = args.num_draws
    if args.learning_rate is not None:
        llc_cfg["learning_rate"] = args.learning_rate

    # Reconstruct model
    m = config["m"]
    k = config["k"]
    n = config.get("n", m * k)
    subtask_indices = make_subtask_indices(m, k)

    model = make_mlp(
        n=n, m=m,
        width=config["width"],
        depth=config["depth"],
        activation=config.get("activation", "relu"),
        use_layernorm=config.get("use_layernorm", False),
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device)

    dtype = torch.float32 if config.get("dtype", "float32") == "float32" else torch.float64
    model = model.to(dtype)

    # Create per-code dataloaders
    task_codes = config.get("task_codes", [[i] for i in range(m)] + [list(range(m))])
    samples_per_code = llc_cfg.get("samples_per_code", 2000)
    batch_size = llc_cfg.get("batch_size", 256)

    print("Creating per-code dataloaders...")
    per_code_loaders = make_subtask_dataloaders(
        n=n, m=m, subtask_indices=subtask_indices,
        task_codes=task_codes,
        samples_per_code=samples_per_code,
        batch_size=batch_size,
        device=device, dtype=dtype,
        seed=llc_cfg.get("seed", 42),
    )

    # Create union dataloaders if specified
    llc_measurements = config.get("llc_measurements", {})
    unions_cfg = llc_measurements.get("unions", {})
    if unions_cfg:
        print("Creating union dataloaders...")
        union_loaders = make_union_dataloaders(
            n=n, m=m, subtask_indices=subtask_indices,
            code_groups=unions_cfg,
            samples_per_code=samples_per_code,
            batch_size=batch_size,
            device=device, dtype=dtype,
            seed=llc_cfg.get("seed", 42),
        )
        per_code_loaders.update(union_loaders)

    # Estimate LLCs
    llc_kwargs = {
        "num_chains": llc_cfg.get("num_chains", 10),
        "num_draws": llc_cfg.get("num_draws", 500),
        "num_burnin_steps": llc_cfg.get("num_burnin_steps", 100),
        "num_steps_bw_draws": llc_cfg.get("num_steps_bw_draws", 1),
        "learning_rate": llc_cfg.get("learning_rate", 1e-4),
        "localization": llc_cfg.get("localization", 0.0),
        "seed": llc_cfg.get("seed", 42),
    }

    print(f"\nEstimating LLCs with {llc_kwargs['num_chains']} chains, "
          f"{llc_kwargs['num_draws']} draws...")
    llc_results = estimate_subtask_llcs(
        model=model,
        subtask_dataloaders=per_code_loaders,
        device=device,
        verbose=not args.quiet,
        **llc_kwargs,
    )

    # Save results
    output_path = model_dir / "llc_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(llc_results, f)
    print(f"\nLLC results saved to {output_path}")

    # Summary
    print("\n" + "=" * 50)
    print("LLC Summary:")
    print("=" * 50)
    for name, r in sorted(llc_results.items()):
        print(f"  {name:>20s}: {r['llc_mean']:.4f} ± {r['llc_std']:.4f}")


if __name__ == "__main__":
    main()
