import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'

import LookupResult from './LookupResult'
import LookupModal from './LookupModal'

export default function LookupInput() {
    const [input, setInput] = useState('');
    const [jobPosition, setJobPosition] = useState(null);
    const [lookupValid, setLookupValid] = useState(false)

    const handleInputChange = (e) => {
        setInput(e.target.value);
    }

    const handleModalClose = () => {
        setLookupValid(false);
    }

    const handleLookup = () => {
        setJobPosition(null)
        axios
            .get(`http://localhost:8000/api/position/${input}/`)
            .then((res) => {
                if (res.data.inQueue) {
                    setLookupValid(true)
                    setJobPosition(res.data.position)
                } else if (res.data.position === -1 ) {
                    setLookupValid(true)
                    setJobPosition(res.data.position)
                } else {
                    setLookupValid(false)
                    setJobPosition(null)
                }
            })
            .catch((err) => {
                console.log(err)
                setLookupValid(false)
                setJobPosition(null)
            })
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
                    spellCheck={false}
                    onChange={handleInputChange}>
            </TextField>
            <Button variant='contained' onClick={handleLookup}>Look up</Button>
            { (lookupValid && jobPosition != null)  ? 
            <LookupModal open={lookupValid} id={input} position={jobPosition} handleClose={handleModalClose}></LookupModal>
             : <div></div>}
        </div>
    )
}
