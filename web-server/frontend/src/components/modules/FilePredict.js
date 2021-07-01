import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function FilePredict() {
    const [item, setItem] = useState({
        'title': '',
        'pairs': null,
        'sequences': null
    });

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handlePairsChange = (e) => {
        setItem({...item, 'pairs': e.target.value});
    }

    const handleSequencesChange = (e) => {
        setItem({...item, 'sequences': e.target.value})
    }

    return (
        <div className="FilePredict-Container">
            <h1>Predict Interaction Between Protein Pairs</h1>
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text" onChange={handleTitleChange}></input><br></br>
                <label>Protein candidate pairs (.tsv)</label><br></br>
                <input type="file" accept=".tsv" onChange={handlePairsChange}></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta" onChange={handleSequencesChange}></input><br></br>
            </form>
            <button onClick={() => console.log(item)}>Compute Interaction Probability</button>
        </div>
    )
}
