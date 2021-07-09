import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function SinglePair() {
    const [item, setItem] = useState({
        'title': '',
        'protein1': '',
        'protein2': '',
        'sequence1': '',
        'sequence2': ''
    });

    const logPredictions = () => {
        axios
            .get("http://localhost:8000/api/predictions/")
            .then((res) => console.log(res.data))
            .catch((err) => console.log(err))
    }

    useEffect(() => {
        logPredictions();
    }, []);

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handleProtein1Change = (e) => {
        setItem({...item, 'protein1': e.target.value});
    }

    const handleProtein2Change = (e) => {
        setItem({...item, 'protein2': e.target.value});
    }

    const handleSequence1Change = (e) => {
        setItem({...item, 'sequence1': e.target.value});
    }

    const handleSequence2Change = (e) => {
        setItem({...item, 'sequence2': e.target.value});
    }

    const handleSubmit = () => {
        console.log(item)
        axios
            .post("http://localhost:8000/api/predictions/", item)
            .then((res) => console.log(res.data.probability))
            .catch((err) => console.log(err))
    }

    return (
        <div className="SinglePair-Container">
            <h1>Predict Interaction Between Two Proteins</h1>
            <form>
                <label>Enter a job title: </label><br></br>
                <input type="text" onChange={handleTitleChange}></input><br></br>
            </form>
            <div className="SinglePair-Proteins">
                <form className="SinglePair-Protein">
                    <label>Protein #1: </label><br></br>
                    <input type="text" onChange={handleProtein1Change}></input><br></br>
                    <label>Sequence #1: </label><br></br>
                    <textarea onChange={handleSequence1Change}></textarea><br></br>
                </form>
                <form className="SinglePair-Protein">
                    <label>Protein #2: </label><br></br>
                    <input type="text" onChange={handleProtein2Change}></input><br></br>
                    <label>Sequence #2: </label><br></br>
                    <textarea onChange={handleSequence2Change}></textarea><br></br>
                </form>
            </div>
            <button onClick={handleSubmit}>Compute Interaction Probability</button>
        </div>
    )
}
