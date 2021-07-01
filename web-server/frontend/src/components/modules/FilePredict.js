import React from 'react'

export default function FilePredict() {
    return (
        <div className="FilePredict-Container">
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text"></input><br></br>
                <label>Protein candidate pairs (.tsv)</label><br></br>
                <input type="file" accept=".tsv"></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta"></input><br></br>
            </form>
        </div>
    )
}
