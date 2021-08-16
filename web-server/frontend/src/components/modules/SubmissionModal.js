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

export default function SubmissionModal(props) {
    const classes = useStyles();

    const [counter, setCounter] = useState(15);
    const [position, setPosition] = useState(props.position);
    const [processed, setProcessed] = useState(false);

    const protectEmail = (email) => {
        let avg, splitted, p1, p2;
        splitted = email.split("@");
        p1 = splitted[0];
        avg = p1.length / 2;
        p1 = p1.substring(0, (p1.length - avg));
        p2 = splitted[1];
        return p1 + "...@" + p2;
    }

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
        <div className='SubmissionModal-Container'>
            <Modal
                className={classes.modal}
                open={props.open}
                // onClose={props.handleClose}
                // closeAfterTransition
                BackdropComponent={Backdrop}
                BackdropProps={{
                  timeout: 500,
                }}
              >
                <Fade in={props.open}>
                  <div className={classes.paper}>
                      { processed ?
                      <div className='SubmissionModal-Info'> 
                        <p><em>Finished refreshing</em></p>
                        <LinearProgress variant='determinate' color='secondary' value='primary'></LinearProgress>
                        <h2>Your job has finished processing</h2>
                        <p>Job id: {props.id}</p>
                        <p><em>The results of your prediction have been emailed to {protectEmail(props.email)}</em></p>
                      </div> :
                      <div className='SubmissionModal-Info'> 
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
