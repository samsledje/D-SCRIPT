import React from 'react'

import NavBar from '../modules/NavBar'

export default function Home() {
    return (
        <div className="Home-Container">
            <NavBar></NavBar>
            <img src="dscript_architecture1.png" alt="D-SCIPRT Architecture"/><br/>

            <div class="about sans_font">
            <p class="home_body">
                D-SCRIPT is a deep learning method for predicting a physical interaction between two proteins given just their sequences.   It generalizes well to new species and is robust to limitations in training data size.  Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in con-tact with the other.  The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein contact map of the protein dimer.  This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved evolutionarily than sequence, improves generalizability across species.
            <br />
            <br />
            D-SCRIPT is described in the paper <a href="https://www.biorxiv.org/content/10.1101/2021.01.22.427866v1">&ldquo;Sequence-based prediction of protein-protein interactions: a structure-aware interpretable deep learning model&rdquo;</a> by <a href="http://people.csail.mit.edu/samsl">Sam Sledzieski</a>, <a href="http://people.csail.mit.edu/rsingh/">Rohit Singh</a>, <a href="http://www.cs.tufts.edu/~cowen/"> Lenore Cowen</a> and <a href="http://people.csail.mit.edu/bab/">Bonnie Berger</a>.
            </p>
            </div>

            <br />

            <div class="about home_body">
            <div class="sans_font">Installation:</div> <div class="title_font">pip install dscript</div>
            </div>
  
        </div>
    )
}
