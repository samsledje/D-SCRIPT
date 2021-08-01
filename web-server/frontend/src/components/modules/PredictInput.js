import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'

import PairInput from './PairInput'
import SequenceInput from './SequenceInput'
import SubmissionModal from './SubmissionModal'

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
    const [modalOpen, setModalOpen] = useState(false);
    const [jobId, setJobId] = useState('');
    const [jobPosition, setJobPosition] = useState('');

    // const handleModalClose = () => {
    //     setModalOpen(false)
    // }

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
        } else {
            uploadData.append('pairs', '')
        }

        axios
            .post("http://localhost:8000/api/predict/", uploadData)
            .then((res) => {
                console.log(res)
                setJobId(res.data.id)
                if (res.data.first) {
                    setJobPosition(1)
                    setModalOpen(true)
                    axios
                        .post("http://localhost:8000/api/process")
                } else {
                    // axios
                    //     .post("http://localhost:8000/api/position", {'id': res.data.id})
                    //     .then((res) => {
                    //         if (res.data.inQueue) {
                    //             setJobPosition(res.data.position)
                    //         } else {
                    //             setJobPosition('None')
                    //         }
                    //     })
                    //     .catch((err) => console.log(err))
                    axios
                        .get(`http://localhost:8000/api/position/${res.data.id}/`)
                        .then((res) => {
                            if (res.data.inQueue) {
                                setModalOpen(true)
                                setJobPosition(res.data.position)
                            } else {
                                setJobPosition('None')
                                setModalOpen(false)
                            }
                        })
                        .catch((err) => console.log(err))

                }
            })
            .catch((err) => console.log(err))

    }

    const testSubmit = () => {
        axios
            .post("http://localhost:8000/api/test", {})
            .then((res) => console.log(res))
            .catch((err) => console.log(err))
    }
    
    return (
        <div className="PredictInput-Container">
            {/* <h1>Predict Protein Interactions</h1> */}
            <h2>PREDICT PROTEIN INTERACTIONS</h2>
            <form autoComplete="off">
                <SequenceInput
                    index={item.seqsIndex}
                    handleIndexChange={handleSeqsIndexChange}
                    filename={item.seqsFilename}
                    handleUploadChange={handleSeqsUploadChange}
                    handleInputChange={handleSeqsInputChange}
                ></SequenceInput>
                <PairInput
                    index={item.pairsIndex}
                    handleIndexChange={handlePairsIndexChange}
                    filename={item.pairsFilename}
                    handleUploadChange={handlePairsUploadChange}
                    handleInputChange={handlePairsInputChange}
                ></PairInput>
                <TextField
                    label='Enter an email address'
                    value={item.email}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    onChange={handleEmailChange}>
                </TextField>
                <TextField
                    label='Enter a job title'
                    value={item.title}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    onChange={handleTitleChange}>
                </TextField>
                <Button variant='contained' onClick={handleSubmit}>Compute Interaction Probability</Button>
                {/* <Button variant='contained' onClick={testSubmit}>Submit</Button> */}
            </form>
            <SubmissionModal open={modalOpen} id={jobId} position={jobPosition}></SubmissionModal>
        </div>
    )
}
