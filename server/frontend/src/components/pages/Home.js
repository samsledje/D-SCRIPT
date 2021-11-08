import React from 'react'

import NavBar from '../modules/NavBar'
import '../../assets/styles_home.css'
import architecture from '../../assets/dscript_architecture1.png'


export default function Home() {
    const citation = `@article{
        Sledzieski_Sequencebased_prediction_of_2021,
        author = {Sledzieski, Samuel and Singh, Rohit and Cowen, Lenore and Berger, Bonnie},
        doi = {10.1101/2021.01.22.427866},
        journal = {bioRxiv},
        month = {1},
        title = {{Sequence-based prediction of protein-protein interactions: a structure-aware interpretable deep learning model}},
        year = {2021}
    }`
    return (
        <div className="Home-Container">
            <NavBar></NavBar>

            <div class="half_center">
                <img class="half_center" src={architecture} alt="D-SCIPRT Architecture"/><br/>

                <div class="nav sans_font">
                    <a href="./predict"><b>Make Predictions Online (InDev!)</b></a>
                </div>

                <br/>
                <br/>

                <div class="about sans_font">
                <p class="home_body">
                    D-SCRIPT is an interpretable deep learning method for predicting a physical interaction between two proteins given just their sequences.   It generalizes well to new species and is robust to limitations in training data size.  Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in con-tact with the other.  The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein contact map of the protein dimer.  This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved evolutionarily than sequence, improves generalizability across species.
                <br/>
                <br/>
                D-SCRIPT is described in the paper <a href="https://www.biorxiv.org/content/10.1101/2021.01.22.427866v1">&ldquo;Sequence-based prediction of protein-protein interactions: a structure-aware interpretable deep learning model&rdquo;</a> by <a href="http://people.csail.mit.edu/samsl">Sam Sledzieski</a>, <a href="http://people.csail.mit.edu/rsingh/">Rohit Singh</a>, <a href="http://www.cs.tufts.edu/~cowen/"> Lenore Cowen</a> and <a href="http://people.csail.mit.edu/bab/">Bonnie Berger</a>.
                </p>
                <p class="title_font">
                    {citation}
                </p>
                </div>
                <br/>

                <div class="about home_body">
                <p class="sans_font">Installation:</p> <p class="title_font">pip install dscript</p>
                </div>
            </div>
        </div>
    )
}
