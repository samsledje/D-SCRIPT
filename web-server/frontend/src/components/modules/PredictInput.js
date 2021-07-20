import React from 'react'

export default function PredictInput() {
    return (
        <div className="PredictInput-Container">
            <h1>Predict Interaction Between Many Pairs</h1>
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text"></input><br></br>
                <label>Protein candidate pairs (.tsv)</label><br></br>
                <input type="file" accept=".tsv"></input>
                or
                <textarea></textarea>
                or
                <input type="checkbox"></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta"></input>
                or
                <textarea></textarea><br></br>
            </form>
            <button>Compute Interaction Probability</button>
        </div>
    )
}
