import React, { useState } from 'react'
import { AppBar, Tab, Button, TextField } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function SequenceInput() {
    const [index, setIndex] = useState('1');
    const [file, setFile] = useState('No file chosen')

    const handleIndexChange = (e, newIndex) => {
        setIndex(newIndex)
    }

    const handleFileChange = (e) => {
        if (typeof e.target.files[0] != 'undefined') {
            setFile(e.target.files[0].name)
        } else {
            setFile('No file chosen')
        }
    }

    return (
        <TabContext value={index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload sequences' value='1'/>
                    <Tab label='Input sequences' value='2'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>
                <input id='upload-seqs' type="file" accept=".fasta" onChange={handleFileChange} hidden/>
                <label htmlFor='upload-seqs' className='SequenceInput-Upload'>
                    <Button variant="contained" color="primary" component="span">
                        Upload .fasta
                    </Button>
                    <em>{file}</em>
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
                />
            </TabPanel>
        </TabContext>
    )
}
