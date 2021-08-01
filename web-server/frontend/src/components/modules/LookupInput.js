import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'

export default function LookupInput() {
    const [input, setInput] = useState('');

    const handleInputChange = (e) => {
        setInput(e.target.value);
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
            <Button variant='contained'>Look up</Button>
        </div>
    )
}
