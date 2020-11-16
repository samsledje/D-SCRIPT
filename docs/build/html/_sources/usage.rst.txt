Usage
=====

Quick Start
~~~~~~~~~~~

- Set up a conda environment using `conda-env.yml <https://github.com/samsledje/D-SCRIPT/blob/main/conda-env.yml>`_ with ``conda create --name [dscript/other env name] --file conda-env.yml``


Embed sequences with language model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``dscript embed --seqs [sequences, .fasta format] --outfile [embedding file]``

Train and save a model
^^^^^^^^^^^^^^^^^^^^^^
``dscript train --train [training data] --val [validation data] --embedding [embedding file] --save-prefix [prefix]``


Evaluate a trained model
^^^^^^^^^^^^^^^^^^^^^^^^
``dscript eval --model [model file] --test [test data] --embedding [embedding file] --outfile [result file]``


Predict a new network using a trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``dscript predict --pairs [input data] --seqs [sequences, .fasta format] --model [model file]``

Embedding
~~~~~~~~~

Training
~~~~~~~~

Evaluation
~~~~~~~~~~

Prediction
~~~~~~~~~~