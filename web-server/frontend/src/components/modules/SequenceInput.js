import React, { useState } from 'react'
import { AppBar, Tab } from '@material-ui/core'
import { TabContext, TabList, TabPanel } from '@material-ui/lab'

export default function SequenceInput() {
    const [index, setIndex] = useState('1');

    const handleIndexChange = (e, newIndex) => {
        setIndex(newIndex)
    }

    return (
        <TabContext value={index}>
            <AppBar position='static' color='primary'>
                <TabList onChange={handleIndexChange} variant='fullWidth'>
                    <Tab label='Upload sequences' value='1'/>
                    <Tab label='Input sequences' value='2'/>
                </TabList>
            </AppBar>
            <TabPanel value='1'>Item One</TabPanel>
            <TabPanel value='2'>Item Two</TabPanel>
        </TabContext>
    )
}
