import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'
import LookupModal from './LookupModal'

axios.defaults.xsrfCookieName = 'csrftoken'
axios.defaults.xsrfHeaderName = 'X-CSRFToken'

export default function LookupInput() {
    const [input, setInput] = useState('');
    const [jobStatus, setJobStatus] = useState(null);
    const [lookupValid, setLookupValid] = useState(false)

    const handleInputChange = (e) => {
        setInput(e.target.value);
    }

    const handleModalClose = () => {
        setLookupValid(false);
    }

    const handleLookup = () => {
        setJobStatus(null)
        axios
            .get(`http://dscript-predict.csail.mit.edu:8000/api/position/${input}/`)
            .then((res) => {
                if (res.status === 200) {
                    setLookupValid(true)
                    setJobStatus(res.data.status)
                } else {
                    setLookupValid(false)
                    setJobStatus(null)
                }
            })
            .catch((err) => {
                alert('The job you attempted to look up does not exist!')
                console.log(err)
                setLookupValid(false)
                setJobStatus(null)
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
            { (lookupValid && jobStatus != null)  ?
            <LookupModal open={lookupValid} id={input} status={jobStatus} handleClose={handleModalClose}></LookupModal>
             : <div></div>}
        </div>
    )
}
