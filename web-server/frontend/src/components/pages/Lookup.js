import React from 'react'

import NavBar from '../modules/NavBar'
import LookupInput from '../modules/LookupInput'

export default function Lookup() {
    return (
        <div className="Lookup-Container">
            <NavBar></NavBar>
            <LookupInput></LookupInput>
        </div>
    )
}
