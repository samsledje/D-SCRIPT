import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'

import PairInput from './PairInput'
import SequenceInput from './SequenceInput'

export default function PredictInput() {
    const [item, setItem] = useState({
        'title': '',
        'email': '',
        'pairsIndex': '1',
        'seqsIndex': '1',
        'pairsUpload': null,
        'pairsInput': '',
        'pairsAll': false,
        'seqsUpload': null,
        'seqsInput': ''
    });

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handleEmailChange = (e) => {
        setItem({...item, 'email': e.target.value})
    }

    const handlePairsIndexChange = (e, newIndex) => {
        setItem({...item, 'pairsIndex': newIndex})
    }

    const handleSeqsIndexChange = (e, newIndex) => {
        setItem({...item, 'seqsIndex': newIndex})
    }

    const handleSubmit = () => {
        console.log(item)
    }
    
    return (
        <div className="PredictInput-Container">
            <h1>Predict Protein Interactions</h1>
            <form autoComplete="off">
                <TextField
                    label='Enter a job title'
                    value={item.title}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleTitleChange}>
                </TextField><br></br>
                <TextField
                    label='Enter an email address'
                    value={item.email}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleEmailChange}>
                </TextField>
                <PairInput index={item.pairsIndex} handleIndexChange={handlePairsIndexChange}></PairInput>
                <SequenceInput index={item.seqsIndex} handleIndexChange={handleSeqsIndexChange}></SequenceInput>
                <Button variant='contained' onClick={handleSubmit}>Compute Interaction Probability</Button>
            </form>
        </div>
    )
}
