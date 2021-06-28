import React from 'react'

import Protein from './Protein'

export default function Predict() {
    return (
        <div className="Predict-Container">
            <form>
                <label>Enter a job title: </label><br></br>
                <input type="text"></input><br></br>
            </form>
            <div className="Predict-Proteins">
                <Protein></Protein>
                <Protein></Protein>
            </div>
        </div>
    )
}
