Usage
=====

Quick Start
~~~~~~~~~~~

Predict a new network using a trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from [TBD].
Candidate pairs should be in tab-separated (``.tsv``) format with no header, and columns for [protein name 1], [protein name 2].
Optionally, a third column with [label] can be provided, so predictions can be made using training or test data files (but the label will not affect the predictions).

.. code-block:: bash

    dscript predict --pairs [input data] --seqs [sequences, .fasta format] --model [model file]

Embed sequences with language model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sequences should be in ``.fasta`` format.

.. code-block:: bash

    dscript embed --seqs [sequences] --outfile [embedding file]

Train and save a model
^^^^^^^^^^^^^^^^^^^^^^

Training and validation data should be in tab-separated (``.tsv``) format with no header, and columns for [protein name 1], [protein name 2], [label].

.. code-block:: bash

    dscript train --train [training data] --val [validation data] --embedding [embedding file] --save-prefix [prefix]


Evaluate a trained model
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    dscript eval --model [model file] --test [test data] --embedding [embedding file] --outfile [result file]


Prediction
~~~~~~~~~~

.. code-block:: bash

    usage: dscript predict [-h] --pairs PAIRS --model MODEL [--seqs SEQS]
                        [--embeddings EMBEDDINGS] [-o OUTFILE] [-d DEVICE]
                        [--thresh THRESH]

    Make new predictions with a pre-trained model. One of --seqs and --embeddings is required.

    optional arguments:
    -h, --help            show this help message and exit
    --pairs PAIRS         Candidate protein pairs to predict
    --model MODEL         Pretrained Model
    --seqs SEQS           Protein sequences in .fasta format
    --embeddings EMBEDDINGS
                            h5 file with embedded sequences
    -o OUTFILE, --outfile OUTFILE
                            File for predictions
    -d DEVICE, --device DEVICE
                            Compute device to use
    --thresh THRESH       Positive prediction threshold - used to store contact
                            maps and predictions in a separate file. [default:
                            0.5]

Embedding
~~~~~~~~~

.. code-block:: bash

    usage: dscript embed [-h] --seqs SEQS --outfile OUTFILE [-d DEVICE]

    Generate new embeddings using pre-trained language model

    optional arguments:
    -h, --help            show this help message and exit
    --seqs SEQS           Sequences to be embedded
    --outfile OUTFILE     h5 file to write results
    -d DEVICE, --device DEVICE
                            Compute device to use

Training
~~~~~~~~

.. code-block:: bash

    usage: dscript train [-h] --train TRAIN --val VAL --embedding EMBEDDING
                        [--augment] [--projection-dim PROJECTION_DIM]
                        [--dropout-p DROPOUT_P] [--hidden-dim HIDDEN_DIM]
                        [--kernel-width KERNEL_WIDTH] [--use-w]
                        [--pool-width POOL_WIDTH]
                        [--negative-ratio NEGATIVE_RATIO]
                        [--epoch-scale EPOCH_SCALE] [--num-epochs NUM_EPOCHS]
                        [--batch-size BATCH_SIZE] [--weight-decay WEIGHT_DECAY]
                        [--lr LR] [--lambda LAMBDA_] [-o OUTFILE]
                        [--save-prefix SAVE_PREFIX] [-d DEVICE]
                        [--checkpoint CHECKPOINT]

    Train a new model

    optional arguments:
    -h, --help            show this help message and exit

    Data:
    --train TRAIN         Training data
    --val VAL             Validation data
    --embedding EMBEDDING
                            h5 file with embedded sequences
    --augment             Set flag to augment data by adding (B A) for all pairs
                            (A B)

    Projection Module:
    --projection-dim PROJECTION_DIM
                            Dimension of embedding projection layer (default: 100)
    --dropout-p DROPOUT_P
                            Parameter p for embedding dropout layer (default: 0.5)

    Contact Module:
    --hidden-dim HIDDEN_DIM
                            Number of hidden units for comparison layer in contact
                            prediction (default: 50)
    --kernel-width KERNEL_WIDTH
                            Width of convolutional filter for contact prediction
                            (default: 7)

    Interaction Module:
    --use-w               Use weight matrix in interaction prediction model
    --pool-width POOL_WIDTH
                            Size of max-pool in interaction model (default: 9)

    Training:
    --negative-ratio NEGATIVE_RATIO
                            Number of negative training samples for each positive
                            training sample (default: 10)
    --epoch-scale EPOCH_SCALE
                            Report heldout performance every this many epochs
                            (default: 5)
    --num-epochs NUM_EPOCHS
                            Number of epochs (default: 100)
    --batch-size BATCH_SIZE
                            Minibatch size (default: 25)
    --weight-decay WEIGHT_DECAY
                            L2 regularization (default: 0)
    --lr LR               Learning rate (default: 0.001)
    --lambda LAMBDA_      Weight on the similarity objective (default: 0.35)

    Output and Device:
    -o OUTPUT, --output OUTPUT
                            Output file path (default: stdout)
    --save-prefix SAVE_PREFIX
                            Path prefix for saving models
    -d DEVICE, --device DEVICE
                            Compute device to use
    --checkpoint CHECKPOINT
                            Checkpoint model to start training from``

Evaluation
~~~~~~~~~~

.. code-block:: bash

    usage: dscript eval [-h] --model MODEL --test TEST --embedding EMBEDDING
                        [-o OUTFILE] [-d DEVICE]

    Evaluate a trained model

    optional arguments:
    -h, --help            show this help message and exit
    --model MODEL         Trained prediction model
    --test TEST           Test Data
    --embedding EMBEDDING
                            h5 file with embedded sequences
    -o OUTFILE, --outfile OUTFILE
                            Output file to write results
    -d DEVICE, --device DEVICE
                            Compute device to use
