import React from 'react'

import { Link } from 'react-router-dom';

export default function NavBar() {
    return (
        <div className="NavBar-Container">
            <div className="NavBar-Home">
                <Link to='/'>D-SCRIPT</Link>
            </div>
            <ul className="NavBar-Links">
                <li>
                    <Link to='/single-pair'>Single Pair</Link>
                </li>
                <li>
                    <Link to='/many-pair'>Many Pair</Link>
                </li>
                <li>
                    <Link to='/all-pair'>All Pair</Link>
                </li>
            </ul>
        </div>
    )
}
