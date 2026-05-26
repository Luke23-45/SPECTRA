from setuptools import setup, find_packages

setup(
    name="spectra-mtl",
    version="1.0.0",
    description=(
        "SPECTRA: Frequency-Decoupled Manifolds and Projected Bayesian "
        "Scaling for Massive Multi-Task Learning"
    ),
    author="Neryva Lab",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "vision": ["opencv-python>=4.8.0", "torchvision>=0.16.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
