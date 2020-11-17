Installation
============

Requirements
------------
- matplotlib
- numpy
- pandas
- pytorch
- seaborn
- scikit-learn
- scipy
- tqdm

Optional GPU support: CUDA Toolkit, cuDNN

Set up environment
------------------

.. code-block:: bash
 
    $ git clone https://github.com/samsledje/D-SCRIPT.git

    $ cd D-SCRIPT

    $ conda env create --file environment.yml # Edit this file to change CUDA version if necessary

    $ conda activate dscript

Install from pip
----------------

[TBD]

.. code-block:: bash

    pip install dscript

Build from source
-----------------

.. code-block:: bash

    $ git clone https://github.com/samsledje/D-SCRIPT.git

    $ cd D-SCRIPT
    
    $ python setup.py build; python setup.py install
