from setuptools import setup, find_packages

setup(
    name="llc-modularity",
    version="0.1.0",
    description="LLC additivity experiments on Compositional Multitask Sparse Parity",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "tracking": ["wandb"],
    },
)
