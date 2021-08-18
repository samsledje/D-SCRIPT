import React, { useState } from 'react'
import { TextField, Button } from '@material-ui/core'
import axios from 'axios'
import Cookies from 'js-cookie';
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
    const [jobId, setJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null);

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
        const csrftoken = Cookies.get('csrftoken');
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
            .post(
                "http://localhost:8000/api/predict/",
                uploadData,
                {
                    headers: {'X-CSRFToken': csrftoken}
                }
                )
            .then((res) => {
                console.log(res)
                setJobId(res.data.id)
                axios
                    .get(`http://localhost:8000/api/position/${res.data.id}/`)
                    .then((res) => {
                        console.log(res)
                        setJobId(res.data.id)
                        if (res.data.status === 'PENDING') {
                            setModalOpen(true)
                            setJobStatus('PENDING')
                        } else if (res.data.status === 'STARTED') {
                            setModalOpen(true)
                            setJobStatus('STARTED')
                        } else if (res.data.status === 'SUCCESS') {
                            setModalOpen(false)
                            setJobStatus('SUCCESS')
                        } else if (res.data.status === 'FAILURE') {
                            setModalOpen(true)
                            setJobStatus('FAILURE')
                        }
                    })
                    .catch((err) => console.log(err))

            })
            .catch((err) => console.log(err))

    }

    return (
        <div className="PredictInput-Container">
            {/* <h1>Predict Protein Interactions</h1> */}
            <h2>PREDICT PROTEIN INTERACTIONS</h2>
            <form autoComplete="off">
                <h3>1. Provide Protein Sequences</h3>
                <SequenceInput
                    index={item.seqsIndex}
                    handleIndexChange={handleSeqsIndexChange}
                    filename={item.seqsFilename}
                    handleUploadChange={handleSeqsUploadChange}
                    handleInputChange={handleSeqsInputChange}
                ></SequenceInput>
                <h3>2. Specify Protein Pairs</h3>
                <PairInput
                    index={item.pairsIndex}
                    handleIndexChange={handlePairsIndexChange}
                    filename={item.pairsFilename}
                    handleUploadChange={handlePairsUploadChange}
                    handleInputChange={handlePairsInputChange}
                ></PairInput>
                <h3>3. Specify a receiving email</h3>
                <TextField
                    label='Enter an email address'
                    value={item.email}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    required={true}
                    spellCheck={false}
                    onChange={handleEmailChange}>
                </TextField>
                <h3>4. Provide a job title (optional)</h3>
                <TextField
                    label='Enter a job title'
                    value={item.title}
                    variant='outlined'
                    margin='dense'
                    fullWidth={true}
                    spellCheck={false}
                    onChange={handleTitleChange}>
                </TextField>
                <Button variant='contained' onClick={handleSubmit}>Compute Interaction Probability</Button>
                {/* <Button variant='contained' onClick={testSubmit}>Submit</Button> */}
            </form>
            { (modalOpen && jobStatus != null && jobId != null)  && <SubmissionModal open={modalOpen} id={jobId} status={jobStatus} email={item.email}></SubmissionModal>}
        </div>
    )
}
