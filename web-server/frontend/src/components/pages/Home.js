import React from 'react'

import SinglePair from '../modules/SinglePair'
import ManyPair from '../modules/ManyPair'
import NavBar from '../modules/NavBar'

export default function Home() {
    return (
        <div className="Home-Container">
            <NavBar></NavBar>
            <div className="Home-Description">
                <h1>D-SCRIPT</h1>
                <p>D-SCRIPT (Deep Sequence Contact Residue Interaction Prediction Transfer) is a deep learning method for predicting protein-protein interaction based on the sequences of their amino acids.</p>
            </div>
            <h1 id="Home-PredictionTypes">Prediction Types</h1>
            <div className="Home-Usages">
                <div className="Home-Usage">
                    <h2>Single Pair</h2>
                    <p>Provide the sequences of a single pair of proteins and compute the probability of these proteins interacting</p>
                </div>
                <div className="Home-Usage">
                    <h2>Many Pair</h2>
                    <p>Provide multiple specified pairs of proteins as well as their sequences and compute the probability of each pair interacting</p>
                </div>
                <div className="Home-Usage">
                    <h2>All Pair</h2>
                    <p>Provide sequences for a set of proteins and compute the probability of all pairs among these proteins interacting</p>
                </div>
            </div>
        </div>
    )
}
