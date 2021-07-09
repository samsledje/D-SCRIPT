import React from 'react'

import SinglePair from '../modules/SinglePair'
import ManyPairs from '../modules/ManyPairs'

export default function Home() {
    return (
        <div className="Home-Container">
            <SinglePair></SinglePair><br></br>
            <ManyPairs></ManyPairs>
        </div>
    )
}
