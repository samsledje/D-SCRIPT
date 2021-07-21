import React, { useState } from 'react'
import { AppBar, Tab, Button, TextField } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function PairInput(props) {
    // index: Current index of Tab
    // handleIndexChange: Changes index of current Tab

    const [filename, setFilename] = useState('No file chosen')

    const handleFilenameChange = (e) => {
        if (typeof e.target.files[0] != 'undefined') {
            setFilename(e.target.files[0].name)
        } else {
            setFilename('No file chosen')
        }
    }

    return (
        <TabContext value={props.index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={props.handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload pairs' value='1'/>
                    <Tab label='Input pairs' value='2'/>
                    <Tab label='All pairs' value='3'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>
                <input id='upload-pairs' type="file" accept=".tsv" hidden onChange={handleFilenameChange}/>
                <label htmlFor='upload-pairs' className='PairInput-Upload'>
                    <Button variant="contained" color="primary" component="span">
                        Upload .tsv
                    </Button>
                    <em>{filename}</em>
                </label>
            </TabPanel>
            <TabPanel value='2'>
                <TextField
                    multiline
                    label='Proteins in .csv format'
                    rows={4}
                    fullWidth={true}
                    variant='outlined'
                    spellCheck='false'
                />
            </TabPanel>
            <TabPanel value='3'>
                By selecting this tab, predictions will be run across all distinct pairs of proteins whose sequences you provide below
            </TabPanel>
        </TabContext>
    )
}
