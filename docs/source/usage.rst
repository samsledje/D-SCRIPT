Usage
=====

Quick Start
~~~~~~~~~~~

Embed sequences with language model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sequences should be in ``.fasta`` format.

.. code-block:: bash

    dscript embed --seqs [sequences] --outfile [embedding file]

Predict a new network using a trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained models can be downloaded from `here <https://d-script.readthedocs.io/en/main/data.html#trained-models>`_.
Protein names should be listed one per line with no header for prediction between all pairs of proteins.
Alternatively, candidate pairs should be in tab-separated (``.tsv``) format with no header, and columns for [protein name 1], [protein name 2].
For a list of pairs, additional columns (for example, a [label] in training or test data files), can exist but are ignored.

.. code-block:: bash

    dscript predict --proteins [list of proteins] --embeddings [embedding file] --outfile [outfile] --model [model file]
    dscript predict --pairs [list of pairs] --embeddings [embedding file] --outfile [outfile] --model [model file]

Train and save a model
^^^^^^^^^^^^^^^^^^^^^^

Training and validation data should be in tab-separated (``.tsv``) format with no header, and columns for [protein name 1], [protein name 2], [label].

.. code-block:: bash

    dscript train --train [training data] --val [validation data] --embedding [embedding file] --save-prefix [prefix]


Evaluate a trained model
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    dscript evaluate --model [model file] --test [test data] --embeddings [embedding file] --outfile [result file]


Blocked, Multi-GPU Prediction
~~~~~~~~~~

.. code-block:: bash

    usage: dscript predict [-h] [--proteins PROTEINS] [--pairs PAIRS] [--model MODEL] --embeddings EMBEDDINGS [--foldseek_fasta FOLDSEEK_FASTA] [-o OUTFILE] [-d DEVICE]
                       [--store_cmaps] [--thresh THRESH] [--load_proc LOAD_PROC] [--blocks BLOCKS] [--sparse_loading]

    Make new predictions with a pre-trained model using blocked, multi-GPU pariwise inference. One of --proteins and --pairs is required.

    options:
      -h, --help            show this help message and exit
      --proteins PROTEINS   File with protein IDs for which to predict all pairs, one per line; specify one of proteins or pairs
      --pairs PAIRS         File with candidate protein pairs to predict, one pair per line; specify one of proteins or pairs
      --model MODEL         Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub
                            [default: samsl/topsy_turvy_human_v1]
      --embeddings EMBEDDINGS
                            h5 file with (a superset of) pre-embedded sequences. Generate with dscript embed.
      --foldseek_fasta FOLDSEEK_FASTA
                            3di sequences in .fasta format. Can be generated using `dscript extract-3di. Default is None. If provided, TT3D will be run, otherwise default
                            D-SCRIPT/TT will be run.
      -o OUTFILE, --outfile OUTFILE
                            File for predictions
      -d DEVICE, --device DEVICE
                            Compute device to use. Options: 'cpu', 'all' (all GPUs), or GPU index (0, 1, 2, etc.). To use specific GPUs, set CUDA_VISIBLE_DEVICES
                            beforehand and use 'all'. [default: all]
      --store_cmaps         Store contact maps for predicted pairs above `--thresh` in an h5 file
      --thresh THRESH       Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]
      --load_proc LOAD_PROC
                            Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the
                            # of CPUs are allowed.
      --blocks BLOCKS       Number of equal-sized blocks to split proteins into. In the multi-block case, maximum (embedding) memory usage should be 3 blocks' worth. When
                            multiple GPUs are used, memory usage may briefly be higher when different GPUs are working on tasks from different blocks. And, small blocks
                            may lead to occasional brief hangs with multiple GPUs. Default 1.
      --sparse_loading      Load only the proteins required from each block, but do not reuse loaded blocks in memory. Recommented when predicting with many blocks on
                            sparse pairs, such that many pairs of blocks might contain no pairs of proteins of interest. Only available when blocks > 1 and pairs
                            specified. Maximum (embedding) memory usage with this option is 4 blocks' worth.

Bipartite, Multi-GPU Prediction
~~~~~~~~~~

.. code-block:: bash

    usage: dscript predict_bipartite [-h] --protA PROTA --protB PROTB [--model MODEL] --embedA EMBEDA [--embedB EMBEDB] [--foldseekA FOLDSEEKA] [--foldseekB FOLDSEEKB] [-o OUTFILE] [-d DEVICE] [--store_cmaps] [--thresh THRESH] [--load_proc LOAD_PROC] [--blocksA BLOCKSA]
                                 [--blocksB BLOCKSB]

    Make new predictions between two protein sets using blocked, multi-GPU pariwise inference  with a pre-trained model.
    
    options:
      -h, --help            show this help message and exit
      --protA PROTA         A text file with protein IDs, one on each line. All pairs between proteins in this file and proteins in protB will be predicted
      --protB PROTB         A text file with protein IDs, one on each line. All pairs between proteins in protA and proteins in this file will be predicted
      --model MODEL         Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]
      --embedA EMBEDA       h5 file with (a superset of) pre-embedded sequences from the file protA. Generate with dscript embed. If a single file contains embeddings for both protA and protB, specify it as embedA.
      --embedB EMBEDB       h5 file with (a superset of) pre-embedded sequences from the file protB. Generate with dscript embed.
      --foldseekA FOLDSEEKA
                            3di sequences in .fasta format for proteins in protA. Can be generated using `dscript extract-3di. Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run. If a single file contains 3di sequences for both protA and protB,
                            specify it as foldseekA.
      --foldseekB FOLDSEEKB
                            3di sequences in .fasta format for proteins in protA. Can be generated using `dscript extract-3di. Default is None. If provided, TT3D will be run, otherwise default D-SCRIPT/TT will be run.
      -o OUTFILE, --outfile OUTFILE
                            File for predictions
      -d DEVICE, --device DEVICE
                            Compute device to use. Options: 'cpu', 'all' (all GPUs), or GPU index (0, 1, 2, etc.). To use specific GPUs, set CUDA_VISIBLE_DEVICES
                            beforehand and use 'all'. [default: all]
      --store_cmaps         Store contact maps for predicted pairs above `--thresh` in an h5 file
      --thresh THRESH       Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]
      --load_proc LOAD_PROC
                            Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the # of CPUs are allowed.
      --blocksA BLOCKSA     Number of equal-sized blocks to split proteins in protA into. If one set is smuch smaller, it is recommended to set the corresponding # of blocks to 1. Default 1.
      --blocksB BLOCKSB     Number of equal-sized blocks to split proteins in protB into. Default 1.


Serial Prediction
~~~~~~~~~~

.. code-block:: bash

    usage: dscript predict_serial [-h] --pairs PAIRS [--model MODEL] [--seqs SEQS] [--embeddings EMBEDDINGS] [--foldseek_fasta FOLDSEEK_FASTA] [-o OUTFILE] [-d DEVICE]
                              [--store_cmaps] [--thresh THRESH] [--load_proc LOAD_PROC]

    Make new predictions with a pre-trained model using legacy (serial) inference. One of --seqs or --embeddings is required.

    options:
      -h, --help            show this help message and exit
      --pairs PAIRS         Candidate protein pairs to predict
      --model MODEL         Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default:
                            samsl/topsy_turvy_human_v1]
      --seqs SEQS           Protein sequences in .fasta format
      --embeddings EMBEDDINGS
                            h5 file with embedded sequences
      --foldseek_fasta FOLDSEEK_FASTA
                            3di sequences in .fasta format. Can be generated using `dscript extract-3di. Default is None. If provided, TT3D will be run, otherwise default
                            D-SCRIPT/TT will be run.
      -o OUTFILE, --outfile OUTFILE
                            File for predictions
      -d DEVICE, --device DEVICE
                            Compute device to use
      --store_cmaps         Store contact maps for predicted pairs above `--thresh` in an h5 file
      --thresh THRESH       Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]
      --load_proc LOAD_PROC
                            Number of processes to use when loading embeddings (-1 = # of CPUs, default=32)


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
