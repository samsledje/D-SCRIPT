# D-SCRIPT
 ![D-SCRIPT Architecture](docs/source/img/dscript_architecture.png)

<!--- #![GitHub release (latest by date)](https://img.shields.io/github/v/release/samsledje/D-SCRIPT) --->
[![D-SCRIPT](https://img.shields.io/github/v/release/samsledje/D-SCRIPT?include_prereleases)](https://github.com/samsledje/D-SCRIPT/releases)
[![PyPI](https://img.shields.io/pypi/v/dscript)](https://pypi.org/project/dscript/)
[![Documentation Status](https://readthedocs.org/projects/d-script/badge/?version=main)](https://d-script.readthedocs.io/en/main/?badge=main)
[![License](https://img.shields.io/github/license/samsledje/D-SCRIPT)](https://github.com/samsledje/D-SCRIPT/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


 D-SCRIPT is a deep learning method for predicting a physical interaction between two proteins given just their sequences. It generalizes well to new species and is robust to limitations in training data size. Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in con-tact with the other. The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein contact map of the protein dimer. This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved evolutionarily than sequence, improves generalizability across species.
 
 - D-SCRIPT is described in the paper [“Sequence-based prediction of protein-protein interactions: a structure-aware interpretable deep learning model”](https://www.biorxiv.org/content/10.1101/2021.01.22.427866v1) by Sam Sledzieski, Rohit Singh, Lenore Cowen and Bonnie Berger.
 
 - [Homepage](http://dscript.csail.mit.edu)
 
 - [Documentation](https://d-script.readthedocs.io/en/main/)
