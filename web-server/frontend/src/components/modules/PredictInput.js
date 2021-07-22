import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'

import PairInput from './PairInput'
import SequenceInput from './SequenceInput'

export default function PredictInput() {
    const [item, setItem] = useState({
        'title': '',
        'email': '',
        'pairsIndex': '1',
        'seqsIndex': '1',
        'pairsUpload': null,
        'pairsFilename': 'No file chosen',
        'pairsInput': '',
        'seqsUpload': null,
        'seqsFilename': 'No file chosen',
        'seqsInput': ''
    });

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handleEmailChange = (e) => {
        setItem({...item, 'email': e.target.value})
    }

    const handlePairsIndexChange = (e, newIndex) => {
        setItem({...item, 'pairsIndex': newIndex})
    }

    const handleSeqsIndexChange = (e, newIndex) => {
        setItem({...item, 'seqsIndex': newIndex})
    }

    const handlePairsUploadChange = (e) => {
        if (typeof e.target.files[0] != 'undefined') {
            setItem({...item, 'pairsUpload': e.target.files[0], 'pairsFilename': e.target.files[0].name});
        } else {
            setItem({...item, 'pairsUpload': null, 'pairsFilename': 'No file chosen'})
        }
    }

    const handlePairsInputChange = (e) => {
        setItem({...item, 'pairsInput': e.target.value});
    }

    const handleSeqsUploadChange = (e) => {
        if (typeof e.target.files[0] != 'undefined') {
            setItem({...item, 'seqsUpload': e.target.files[0], 'seqsFilename': e.target.files[0].name});
        } else {
            setItem({...item, 'seqsUpload': null, 'seqsFilename': 'No file chosen'})
        }
    }

    const handleSeqsInputChange = (e) => {
        setItem({...item, 'seqsInput': e.target.value});
    }

    const handleSubmit = () => {
        console.log(item)
        const uploadData = new FormData()
        uploadData.append('title', item.title)
        uploadData.append('email', item.email)
        uploadData.append('pairsIndex', item.pairsIndex)
        uploadData.append('seqsIndex', item.seqsIndex)

        // Handling pairs submission
        if (item.pairsIndex === '1') {
            if (item.pairsUpload != null) {
                uploadData.append('pairs', item.pairsUpload)
            } else {
                alert('Upload a file of protein pairs!')
            }
        } else if (item.pairsIndex === '2') {
            if (item.pairsInput !== '') {
                uploadData.append('pairs', item.pairsInput)
            } else {
                alert('Enter protein pairs!')
            }
        }

        // Handling sequences submission
        if (item.seqsIndex === '1') {
            if (item.seqsUpload != null) {
                uploadData.append('seqs', item.seqsUpload)
            } else {
                alert('Upload a file of protein sequences!')
            }
        } else if (item.seqsIndex === '2') {
            if (item.seqsInput !== '') {
                uploadData.append('seqs', item.seqsInput)
            } else {
                alert('Enter protein sequences!')
            }
        }

        console.log(uploadData.data)
    }
    
    return (
        <div className="PredictInput-Container">
            <h1>Predict Protein Interactions</h1>
            <form autoComplete="off">
                <PairInput
                    index={item.pairsIndex}
                    handleIndexChange={handlePairsIndexChange}
                    filename={item.pairsFilename}
                    handleUploadChange={handlePairsUploadChange}
                    handleInputChange={handlePairsInputChange}
                ></PairInput>
                <SequenceInput
                    index={item.seqsIndex}
                    handleIndexChange={handleSeqsIndexChange}
                    filename={item.seqsFilename}
                    handleUploadChange={handleSeqsUploadChange}
                    handleInputChange={handleSeqsInputChange}
                ></SequenceInput>
                <TextField
                    label='Enter a job title'
                    value={item.title}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleTitleChange}>
                </TextField><br></br>
                <TextField
                    label='Enter an email address'
                    value={item.email}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleEmailChange}>
                </TextField>
                <Button variant='contained' onClick={handleSubmit}>Compute Interaction Probability</Button>
            </form>
        </div>
    )
}
