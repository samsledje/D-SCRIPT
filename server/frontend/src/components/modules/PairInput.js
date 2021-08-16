import React from 'react'
import { AppBar, Tab, Button, TextField } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function PairInput(props) {
    // index: Current index of Tab
    // handleIndexChange: Changes index of current Tab
    // filename: Current uploaded filename
    // handleUploadChange: Changes uploaded file
    // handleInputChange: Changes pair input

    return (
        <TabContext value={props.index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={props.handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload pairs' value='1'/>
                    <Tab label='Enter pairs' value='2'/>
                    <Tab label='All pairs' value='3'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>
                <input id='upload-pairs' type="file" accept=".tsv" hidden onChange={props.handleUploadChange}/>
                <label htmlFor='upload-pairs' className='PairInput-Upload'>
                    <Button variant="contained" color="primary" component="span">
                        Upload .tsv
                    </Button>
                    <em>{props.filename}</em>
                </label>
            </TabPanel>
            <TabPanel value='2'>
                <TextField
                    multiline
                    label='Proteins in comma-separated format'
                    rows={6}
                    fullWidth={true}
                    variant='outlined'
                    spellCheck={false}
                    onChange={props.handleInputChange}
                />
            </TabPanel>
            <TabPanel value='3'>
                By selecting this tab, predictions will be run across all distinct pairs of proteins whose sequences you provide above
            </TabPanel>
        </TabContext>
    )
}
