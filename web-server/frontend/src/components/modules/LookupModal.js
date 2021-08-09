import React, { useState, useEffect } from 'react'
import { makeStyles, Modal, Backdrop, Fade, LinearProgress } from '@material-ui/core'

import axios from 'axios'

const useStyles = makeStyles((theme) => ({
    modal: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    paper: {
      backgroundColor: theme.palette.background.paper,
      border: '2px solid #000',
      boxShadow: theme.shadows[5],
      padding: theme.spacing(2, 4, 3),
    },
  }));

export default function LookupModal(props) {
    const classes = useStyles();

    const [counter, setCounter] = useState(15);
    const [position, setPosition] = useState(props.position);
    const [processed, setProcessed] = useState(false);

    useEffect(() => {
        if (props.position === -1) {
            setProcessed(true)
        }
    }, [props.position])

    useEffect(() => {
      if (counter > 0) {
        setTimeout(() => {
          setCounter(counter - 1)
        }, 1000);
      } else {
        axios
          .get(`http://localhost:8000/api/position/${props.id}/`)
          .then((res) => {
              if (res.data.inQueue) {
                  setPosition(res.data.position)
                  setCounter(15);
              } else {
                  if (res.data.position === -1) {
                    setProcessed(true);
                  }
                  setPosition('None')
              }
          })
          .catch((err) => console.log(err))
      }
    }, [counter, props.id]);

    return (
        <div className='LookupModal-Container'>
            <Modal
                className={classes.modal}
                open={props.open}
                onClose={props.handleClose}
                closeAfterTransition
                BackdropComponent={Backdrop}
                BackdropProps={{
                  timeout: 500,
                }}
              >
                <Fade in={props.open}>
                  <div className={classes.paper}>
                      { processed ?
                      <div className='LookupModal-Info'> 
                        <p><em>Finished refreshing</em></p>
                        <LinearProgress variant='determinate' color='secondary' value='primary'></LinearProgress>
                        <h2>Your job has finished processing</h2>
                        <p>Job id: {props.id}</p>
                        <p><em>Check your email to see the results of your prediction</em></p>
                      </div> :
                      <div className='LookupModal-Info'> 
                        <p><em>Refreshing in {counter} seconds...</em></p>
                        <LinearProgress></LinearProgress>
                        <h2>Your job is queued in position {position}</h2>
                        <p>Job id: {props.id}</p>
                        <p><em>Make note of this job id for the purpose of tracking your job.</em></p>
                      </div>
                      }
                  </div>
                </Fade>
              </Modal>
        </div>
    )
}