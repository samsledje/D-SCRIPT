import React from 'react'

import SinglePair from '../modules/SinglePair'
import ManyPairs from '../modules/ManyPairs'
import NavBar from '../modules/NavBar'

export default function Home() {
    return (
        <div className="Home-Container">
            <NavBar></NavBar>
            <div className="Home-Description">
                <h1>D-SCRIPT</h1>
                <p>D-SCRIPT (Deep Sequence Contact Residue Interaction Prediction Transfer) is a deep learning method for predicting protein-protein interaction based on the sequences of their amino acids.</p>
            </div>
        </div>
    )
}
