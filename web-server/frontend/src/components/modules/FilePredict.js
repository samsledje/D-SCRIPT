import React from 'react'

export default function FilePredict() {
    return (
        <div className="FilePredict-Container">
            <h1>Predict Interaction Between Protein Pairs</h1>
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text"></input><br></br>
                <label>Protein candidate pairs (.tsv)</label><br></br>
                <input type="file" accept=".tsv"></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta"></input><br></br>
            </form>
            <button>Compute Interaction Probability</button>
        </div>
    )
}
