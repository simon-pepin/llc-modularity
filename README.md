# LLC Additivity on Compositional Multitask Sparse Parity

Experimental framework to test **LLC (Local Learning Coefficient) additivity** as a signature of modularity in neural networks trained on the **Compositional Multitask Sparse Parity (CMSP)** task.

## The Question

Given subtasks $T_1, T_2$ of a compositional multitask problem, is the LLC additive?

$$\delta = \hat\lambda(T_1) + \hat\lambda(T_2) - \hat\lambda(T_1 \cup T_2)$$

- **$\delta \approx 0$**: Independent circuits (modular decomposition)
- **$\delta > 0$**: Circuit sharing (sub-additive complexity — the joint is simpler than the sum of parts)
- **$\delta < 0$**: Circuit interference (super-additive complexity)

## Background

- **CMSP** ([Michaud, Parker-Sartori & Tegmark, arXiv:2505.15811](https://arxiv.org/abs/2505.15811)): Multitask sparse parity with compositional structure. Networks exhibit strong curriculum effects — composite tasks can only be learned alongside their atomic constituents.
- **LLC** ([Singular Learning Theory](https://github.com/timaeus-research/devinterp)): The local learning coefficient $\lambda$ measures effective model complexity at a trained singularity, estimated via SGLD sampling.

## Setup

```bash
# Clone dependencies
git clone https://github.com/ejmichaud/narrow.git
git clone https://github.com/timaeus-research/devinterp.git
pip install -e ./devinterp

# Install this project
pip install -e .
```

## Experiments

### Experiment A: Independent Atomics (Calibration)

Train on two independent atomic subtasks. Expect $\delta \approx 0$.

```bash
python experiments/run_all.py --config config/sweeps/independent_atomic.yaml --save-dir results/exp_a
```

### Experiment B: Composite with Curriculum

Train on atomics + composite. Measure LLC additivity.

```bash
python experiments/run_all.py --config config/sweeps/composite_curriculum.yaml --save-dir results/exp_b
```

### Experiment C: Width Sweep

```bash
for w in 64 128 256 512; do
    python experiments/run_all.py --config config/sweeps/composite_curriculum.yaml \
        --save-dir results/width_$w --seeds 0 1 2
done
```

### Individual Steps

```bash
# Train only
python experiments/run_training.py --config config/default.yaml --save-dir results/run0

# Estimate LLCs on a trained model
python experiments/run_llc.py --model-dir results/run0

# Compute additivity defects
python experiments/run_additivity.py --model-dir results/run0
```

## Project Structure

```
llc-modularity/
├── config/              # YAML experiment configs
│   ├── default.yaml
│   └── sweeps/
├── src/                 # Core library
│   ├── data.py          # CMSP data generation
│   ├── model.py         # MLP definitions
│   ├── train.py         # Training loop
│   ├── llc_estimation.py  # Data-restricted LLC estimation
│   └── additivity.py    # Additivity defect computation
├── experiments/         # Entry points
│   ├── run_training.py
│   ├── run_llc.py
│   ├── run_additivity.py
│   └── run_all.py       # Full pipeline
├── notebooks/
│   └── analysis.ipynb   # Interactive analysis
└── tests/
```

## Key Design: Data-Restricted LLC

The central technique is estimating the LLC using **different subsets of data** corresponding to different subtasks. For each subtask $T_i$, we run SGLD using only $T_i$'s loss function, starting from the same jointly-trained weights. This gives $\hat\lambda(T_i)$ — the effective complexity of the model *as seen by subtask $T_i$'s loss landscape*.

## Tests

```bash
pytest tests/ -v
```
