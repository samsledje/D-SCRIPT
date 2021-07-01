import React from 'react'

import Predict from '../modules/Predict'
import FilePredict from '../modules/FilePredict'

export default function Home() {
    return (
        <div className="Home-Container">
            <Predict></Predict><br></br>
            <FilePredict></FilePredict>
        </div>
    )
}
