import React from 'react'

export default function Predict() {
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
                    <label>Protein #1: </label><br></br>
                    <input type="text"></input><br></br>
                    <label>Sequence #1: </label><br></br>
                    <textarea></textarea><br></br>
                </form>
            </div>
        </div>
    )
}
