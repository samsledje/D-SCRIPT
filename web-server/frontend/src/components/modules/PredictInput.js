import React from 'react'
import { TextField, Button } from '@material-ui/core'

import PairInput from './PairInput'
import SequenceInput from './SequenceInput'

export default function PredictInput() {
    return (
        <div className="PredictInput-Container">
            <h1>Predict Protein Interactions</h1>
            <form autoComplete="off">
                <TextField
                    label='Enter a job title'
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}>
                </TextField><br></br>
                <TextField
                    label='Enter an email address'
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}>
                </TextField>
                <PairInput></PairInput>
                <SequenceInput></SequenceInput>
                <Button variant='contained'>Compute Interaction Probability</Button>
                <br></br>
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
