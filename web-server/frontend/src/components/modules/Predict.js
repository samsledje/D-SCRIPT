import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function Predict() {

    const logPredictions = () => {
        axios
            .get("http://localhost:8000/api/predictions/")
            .then((res) => console.log(res.data))
            .catch((err) => console.log(err))
    }

    useEffect(() => {
        logPredictions();
    }, []);

    return (
        <div className="Predict-Container">
            <form>
                <label>Enter a job title: </label><br></br>
                <input type="text"></input><br></br>
            </form>
            <div className="Predict-Proteins">
                <form className="Predict-Protein">
                    <label>Protein #1: </label><br></br>
                    <input type="text"></input><br></br>
                    <label>Sequence #1: </label><br></br>
                    <textarea></textarea><br></br>
                </form>
                <form className="Predict-Protein">
                    <label>Protein #2: </label><br></br>
                    <input type="text"></input><br></br>
                    <label>Sequence #2: </label><br></br>
                    <textarea></textarea><br></br>
                </form>
            </div>
            <button>Compute Interaction Probability</button>
        </div>
    )
}
