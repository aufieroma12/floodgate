"""Setup file for package."""
from setuptools import find_packages, setup

setup(
    name="floodgate",
    version="0.0.0",
    packages=find_packages(include=["src*", "ishigami*", "config*", "Hymod*", "CBMZ*"]),
    description="package for running floodgate sensitivity analysis.",
    install_requires=[
        "scikit-learn<=1.1.1",
        "numpy",
        "safepython",
        "joblib",
        "matplotlib",
        "pandas",
    ],
)
