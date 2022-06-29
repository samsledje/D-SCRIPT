#!/usr/bin/env python

from setuptools import setup, find_packages
import dscript

setup(
    name="dscript",
    version=dscript.__version__,
    description="D-SCRIPT: protein-protein interaction prediction",
    author="Samuel Sledzieski",
    author_email="samsl@mit.edu",
    url="http://dscript.csail.mit.edu",
    license="GPLv3",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "dscript = dscript.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy", # mathematical operations/representations 
        "scipy", # linear algebra
        "pandas", # data analysis tools
        "torch", #nn training/creation
        "matplotlib", # visualization
        "seaborn", # matplotlib 2.0
        "tqdm", # progress bar
        "scikit-learn", # machine learning
        "h5py", # store lots of (binary) data
    ],
)
