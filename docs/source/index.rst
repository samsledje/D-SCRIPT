D-SCRIPT: Deep Learning PPI Prediction
=======================================

- `D-SCRIPT Home Page`_

- `Quick Start <usage.html#quick-start>`_

D-SCRIPT is a deep learning method for predicting a physical interaction between two proteins given just their sequences.
It generalizes well to new species and is robust to limitations in training data size.
Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in contact with the other.
The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein
contact map of the protein dimer. This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved
evolutionarily than sequence, improves generalizability across species.

If you use D-SCRIPT, please cite `"D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions" <https://www.cell.com/cell-systems/fulltext/S2405-4712(21)00333-1>`_
by `Sam Sledzieski`_, `Rohit Singh`_, `Lenore Cowen`_, and `Bonnie Berger`_.

If you use Topsy-Turvy, please cite `"Topsy-Turvy: integrating a global view into sequence-based PPI prediction" <https://cb.csail.mit.edu/cb/topsyturvy/>`_ by `Kapil Devkota`_, `Rohit Singh`_, `Sam Sledzieski`_, `Bonnie Berger`_, and `Lenore Cowen`_.

.. _`D-SCRIPT Home Page`: http://dscript.csail.mit.edu
.. _`Kapil Devkota`: http://www.kapildevkota.com
.. _`Sam Sledzieski`: http://samsledje.github.io/
.. _`Rohit Singh`: http://people.csail.mit.edu/rsingh/
.. _`Lenore Cowen`: http://www.cs.tufts.edu/~cowen/
.. _`Bonnie Berger`: http://people.csail.mit.edu/bab/

Table of contents
=================

.. toctree::
   :maxdepth: 1

   installation
   usage
   data
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
