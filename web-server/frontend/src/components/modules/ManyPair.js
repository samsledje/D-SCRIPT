import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function ManyPair() {
    const [item, setItem] = useState({
        'title': '',
        'pairs': null,
        'sequences': null
    });

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handlePairsChange = (e) => {
        setItem({...item, 'pairs': e.target.files[0]});
    }

    const handleSequencesChange = (e) => {
        setItem({...item, 'sequences': e.target.files[0]})
    }

    const handleSubmit = () => {
        console.log(item)
        const uploadData = new FormData()
        uploadData.append('title', item.title)
        uploadData.append('pairs', item.pairs)
        uploadData.append('sequences', item.sequences)
        axios
            .post("http://localhost:8000/api/manypair/", uploadData)
            .then((res) => console.log(res))
            .catch((err) => console.log(err))
    }

    return (
        <div className="ManyPair-Container">
            <h1>Predict Interaction Between Protein Pairs</h1>
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text" onChange={handleTitleChange}></input><br></br>
                <label>Protein candidate pairs (.tsv)</label><br></br>
                <input type="file" accept=".tsv" onChange={handlePairsChange}></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta" onChange={handleSequencesChange}></input><br></br>
            </form>
            <button onClick={handleSubmit}>Compute Interaction Probability</button>
        </div>
    )
}
