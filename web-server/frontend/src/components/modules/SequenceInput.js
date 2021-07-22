import React from 'react'
import { AppBar, Tab, Button, TextField } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function SequenceInput(props) {
    // index: Current index of Tab
    // handleIndexChange: Changes index of current Tab
    // filename: Current uploaded filename
    // handleUploadChange: Changes uploaded file
    // handleInputChange: Changes pair input

    return (
        <TabContext value={props.index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={props.handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload sequences' value='1'/>
                    <Tab label='Enter sequences' value='2'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>
                <input id='upload-seqs' type="file" accept=".fasta" onChange={props.handleUploadChange} hidden/>
                <label htmlFor='upload-seqs' className='SequenceInput-Upload'>
                    <Button variant="contained" color="primary" component="span">
                        Upload .fasta
                    </Button>
                    <em>{props.filename}</em>
                </label>
            </TabPanel>
            <TabPanel value='2'>
                <TextField
                    multiline
                    label='Sequences in .fasta format'
                    rows={4}
                    fullWidth={true}
                    variant='outlined'
                    spellCheck='false'
                    onChange={props.handleInputChange}
                />
            </TabPanel>
        </TabContext>
    )
}
