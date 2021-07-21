import React, { useState } from 'react'
import { AppBar, Tab, Button, TextField } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function PairInput() {
    const [index, setIndex] = useState('1');
    const [file, setFile] = useState('No file chosen')

    const handleIndexChange = (e, newIndex) => {
        setIndex(newIndex)
    }

    const handleFileChange = (e) => {
        console.log(e.target.files[0])
        setFile(e.target.files[0].name)
    }

    return (
        <TabContext value={index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload pairs' value='1'/>
                    <Tab label='Input pairs' value='2'/>
                    <Tab label='All pairs' value='3'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>
                <input id='upload-pairs' type="file" accept=".tsv" onChange={handleFileChange}/>
                <label htmlFor='upload-pairs'>
                    <Button variant="contained" color="primary" component="span">
                        Upload .tsv
                    </Button><br></br>
                    {file}
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
