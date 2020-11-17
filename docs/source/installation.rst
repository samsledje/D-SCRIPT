Installation
============

Requirements
------------
- numpy 1.18.5
- scipy 1.4.1
- pandas 1.0.5
- pytorch 1.2.0
- matplotlib 3.1.3
- seaborn 0.10.1
- tqdm 4.42
- scikit-learn 0.22.2

Optional GPU support: CUDA Version 10.1, cuDNN 7.6.5

- Set up a conda environment using `environment.yml <https://github.com/samsledje/D-SCRIPT/blob/main/environment.yml>`_

.. code-block:: bash
 
    $ cd D-SCRIPT

    $ conda env create --file environment.yml

    $ conda activate dscript

Install from pip
----------------

.. code-block:: bash

    [TBD] pip install dscript

Build from source
-----------------

.. code-block:: bash

    git clone https://github.com/samsledje/D-SCRIPT.git; python setup.py build; python setup.py install
