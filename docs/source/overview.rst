Overview
========

D-SCRIPT is a deep learning method for predicting a physical interaction between two proteins given just their sequences. It generalizes well to new species and is robust to limitations in training data size. Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in con-tact with the other. The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein contact map of the protein dimer. This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved evolutionarily than sequence, improves generalizability across species.

Basic Usage
~~~~~~~~~~~

``dscript embed --seqs [sequences, .fasta format] --outfile [embedding file]``

``dscript train --train [training data] --val [validation data] --embedding [embedding file] --save-prefix [model file]``

``dscript eval --model [model_file]_final.sav --test [test data] --embedding [embedding file] --outfile [result file]``