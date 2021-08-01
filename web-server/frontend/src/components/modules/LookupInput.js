import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'

import SubmissionModal from './SubmissionModal'
import LookupResult from './LookupResult'

export default function LookupInput() {
    const [input, setInput] = useState('');
    const [jobPosition, setJobPosition] = useState('');
    const [lookupValid, setLookupValid] = useState(false)

    const handleInputChange = (e) => {
        setInput(e.target.value);
    }

    const handleLookup = () => {
        axios
            .get(`http://localhost:8000/api/position/${input}/`)
            .then((res) => {
                if (res.data.inQueue) {
                    setLookupValid(true)
                    setJobPosition(res.data.position)
                } else {
                    setLookupValid(false)
                    setJobPosition('None')
                }
            })
            .catch((err) => console.log(err))
    }

    return (
        <div className="LookupInput-Container">
            <TextField
                    label='Look up a job by ID'
                    value={input}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleInputChange}>
            </TextField>
            <Button variant='contained' onClick={handleLookup}>Look up</Button>
            {lookupValid ? <LookupResult position={jobPosition}></LookupResult> : ''}
        </div>
    )
}
