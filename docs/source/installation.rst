Installation
============

Install from pip
----------------

The simplest way to install D-SCRIPT is via pip. This will install the latest version from PyPI.

.. code-block:: bash

    pip install dscript


Build from source
-----------------

If you want to build D-SCRIPT for source or contribute to development, you can follow the below instructions.

.. code-block:: bash

    $ git clone https://github.com/samsledje/D-SCRIPT.git

    $ cd D-SCRIPT

    $ pip install -e .[dev]

This will install the following dependencies:

    - biotite == 1.2.0
    - h5py
    - huggingface_hub
    - loguru
    - matplotlib
    - numpy
    - pandas
    - pytest
    - pytest-cov
    - ruff
    - safetensors
    - scikit-learn
    - scipy
    - seaborn
    - torch >= 1.13
    - tqdm

Optional GPU support: CUDA Toolkit, cuDNN