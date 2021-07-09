import React from 'react'

import SinglePair from '../modules/SinglePair'
import FilePredict from '../modules/FilePredict'

export default function Home() {
    return (
        <div className="Home-Container">
            <SinglePair></SinglePair><br></br>
            <FilePredict></FilePredict>
        </div>
    )
}
