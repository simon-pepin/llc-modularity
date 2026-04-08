"""Entry point: Train CMSP models.

Usage:
    python experiments/run_training.py --config config/default.yaml --save-dir results/run0
    python experiments/run_training.py --config config/default.yaml --seed 5 --width 256
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_cmsp
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train CMSP model")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--width", type=int, default=None, help="Override width")
    parser.add_argument("--depth", type=int, default=None, help="Override depth")
    parser.add_argument("--steps", type=int, default=None, help="Override steps")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.width is not None:
        config["width"] = args.width
    if args.depth is not None:
        config["depth"] = args.depth
    if args.steps is not None:
        config["steps"] = args.steps
    if args.device is not None:
        config["device"] = args.device

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"results/w{config['width']}_d{config['depth']}_s{config['seed']}"

    print(f"Training config: m={config['m']}, k={config['k']}, "
          f"width={config['width']}, depth={config['depth']}, "
          f"seed={config['seed']}, steps={config['steps']}")
    print(f"Task codes: {config.get('task_codes', 'default')}")
    print(f"Saving to: {save_dir}")

    results = train_cmsp(config, save_dir=save_dir, verbose=not args.quiet)

    print(f"\nTraining complete. Parameters: {results['n_parameters']}")
    print("Final per-subtask losses:")
    for name, loss in results["final_subtask_losses"].items():
        print(f"  {name}: {loss:.6f}")


if __name__ == "__main__":
    main()
