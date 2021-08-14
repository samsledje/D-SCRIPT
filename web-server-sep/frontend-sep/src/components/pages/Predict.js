import React from 'react'

import NavBar from '../modules/NavBar'
import PredictInput from '../modules/PredictInput'

export default function Predict() {
    return (
        <div className="Predict-Container">
            <NavBar></NavBar>
            <PredictInput></PredictInput>
        </div>
    )
}
