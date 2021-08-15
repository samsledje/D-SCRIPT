import React from 'react'

import { Link } from 'react-router-dom';

export default function NavBar() {
    return (
        <div className="NavBar-Container">
            <div className="NavBar-Home">
                <Link to='/'>D-SCRIPT Home</Link>
            </div>
            <ul className="NavBar-Links">
                <li>
                    <a href='https://github.com/samsledje/D-SCRIPT'>Code</a>
                </li>
                <li>
                    <a href='https://d-script.readthedocs.io/en/main/'>Documentation</a>
                </li>
                <li>
                    <Link to='/predict'>Predict</Link>
                </li>
                <li>
                    <Link to='/lookup'>Lookup</Link>
                </li>
            </ul>
        </div>
    )
}
