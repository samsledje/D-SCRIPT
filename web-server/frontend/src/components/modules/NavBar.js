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
                    <Link to='/pair'>Single Pair</Link>
                </li>
                <li>
                    <Link to='/pairs'>Many Pairs</Link>
                </li>
            </ul>
        </div>
    )
}
