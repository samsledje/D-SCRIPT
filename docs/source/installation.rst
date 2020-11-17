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

Set up environment
------------------

.. code-block:: bash
 
    $ git clone https://github.com/samsledje/D-SCRIPT.git

    $ cd D-SCRIPT

    $ conda env create --file environment.yml # Edit this file to change CUDA and cuDNN version if necessary

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
