Usage
=====

Quick Start
~~~~~~~~~~~

Predict a new network using a trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from `here <https://d-script.readthedocs.io/en/main/data.html#trained-models>`_.
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

    dscript evaluate --model [model file] --test [test data] --embedding [embedding file] --outfile [result file]


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
    --model MODEL         Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_v1]
    --seqs SEQS           Protein sequences in .fasta format
    --foldseek_fasta FOLDSEEK_FASTA
                            3di sequences in .fasta format. Can be generated using `dscript extract-3di. Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run.
    --embeddings EMBEDDINGS
                            h5 file with embedded sequences
    -o OUTFILE, --outfile OUTFILE
                            File for predictions
    --store_cmaps      Store contact maps for predicted pairs above `--thresh` in an h5 file
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

    usage: dscript train [-h] --train TRAIN --test TEST --embedding EMBEDDING
                     [--no-augment] [--input-dim INPUT_DIM]
                     [--projection-dim PROJECTION_DIM] [--dropout-p DROPOUT_P]
                     [--hidden-dim HIDDEN_DIM] [--kernel-width KERNEL_WIDTH]
                     [--no-w] [--no-sigmoid] [--do-pool]
                     [--pool-width POOL_WIDTH] [--num-epochs NUM_EPOCHS]
                     [--batch-size BATCH_SIZE] [--weight-decay WEIGHT_DECAY]
                     [--lr LR] [--lambda INTERACTION_WEIGHT] [--topsy-turvy]
                     [--glider-weight GLIDER_WEIGHT]
                     [--glider-thresh GLIDER_THRESH] [-o OUTFILE]
                     [--save-prefix SAVE_PREFIX] [-d DEVICE]
                     [--checkpoint CHECKPOINT]

    Train a new model.

    optional arguments:
      -h, --help            show this help message and exit

    Data:
      --train TRAIN         list of training pairs
      --test TEST           list of validation/testing pairs
      --embedding EMBEDDING
                            h5py path containing embedded sequences
      --no-augment          data is automatically augmented by adding (B A) for
                            all pairs (A B). Set this flag to not augment data

    Projection Module:
      --input-dim INPUT_DIM
                            dimension of input language model embedding (per amino
                            acid) (default: 6165)
      --projection-dim PROJECTION_DIM
                            dimension of embedding projection layer (default: 100)
      --dropout-p DROPOUT_P
                            parameter p for embedding dropout layer (default: 0.5)

    Contact Module:
      --hidden-dim HIDDEN_DIM
                            number of hidden units for comparison layer in contact
                            prediction (default: 50)
      --kernel-width KERNEL_WIDTH
                            width of convolutional filter for contact prediction
                            (default: 7)

    Interaction Module:
      --no-w                don't use weight matrix in interaction prediction
                            model
      --no-sigmoid          don't use sigmoid activation at end of interaction
                            model
      --do-pool             use max pool layer in interaction prediction model
      --pool-width POOL_WIDTH
                            size of max-pool in interaction model (default: 9)

    Training:
      --num-epochs NUM_EPOCHS
                            number of epochs (default: 10)
      --batch-size BATCH_SIZE
                            minibatch size (default: 25)
      --weight-decay WEIGHT_DECAY
                            L2 regularization (default: 0)
      --lr LR               learning rate (default: 0.001)
      --lambda INTERACTION_WEIGHT
                            weight on the similarity objective (default: 0.35)
      --topsy-turvy         run in Topsy-Turvy mode -- use top-down GLIDER scoring
                            to guide training (reference TBD)
      --glider-weight GLIDER_WEIGHT
                            weight on the GLIDER accuracy objective (default: 0.2)
      --glider-thresh GLIDER_THRESH
                            proportion of GLIDER scores treated as positive edges
                            (0 < gt < 1) (default: 0.925)

    Output and Device:
      -o OUTPUT, --output OUTPUT
                            output file path (default: stdout)
      --save-prefix SAVE_PREFIX
                            path prefix for saving models
      -d DEVICE, --device DEVICE
                            compute device to use
      --checkpoint CHECKPOINT
                            checkpoint model to start training from

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
