import React from 'react'

import { TextField } from '@material-ui/core'

export default function LookupInput() {
    return (
        <div className="LookupInput-Container">
            <TextField
                    label='Lookup a job by ID'
                    // value={item.email}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}>
                </TextField>
        </div>
    )
}
