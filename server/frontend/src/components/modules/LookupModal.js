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

    const [counter, setCounter] = useState(0);
    const [status, setJobStatus] = useState(props.status);
    const [processed, setProcessed] = useState(false);
    const [_, setLookupValid] = useState(true);
    const [viewPath, setViewPath] = useState(null);

    // const protectEmail = (email) => {
    //     let avg, splitted, p1, p2;
    //     splitted = email.split("@");
    //     p1 = splitted[0];
    //     avg = p1.length / 2;
    //     p1 = p1.substring(0, (p1.length - avg));
    //     p2 = splitted[1];
    //     return p1 + "...@" + p2;
    // }

    useEffect(() => {
        if (props.status === 'SUCCESS') {
            setJobStatus(true)
        }
    }, [props.status])

    useEffect(() => {
      if (counter > 0) {
        setTimeout(() => {
          setCounter(counter - 1)
        }, 1000);
      } else {
        axios
          .get(`http://localhost:8000/api/position/${props.id}/`)
          .then((res) => {
            if (res.status === 200) {
                setLookupValid(true)
                setJobStatus(res.data.status)
                if (res.data.status == 'PENDING') {
                  setProcessed(false)
                  setCounter(10)
                } else if (res.data.status == 'STARTED') {
                  setProcessed(false)
                  setCounter(10)
                } else if (res.data.status == 'SUCCESS') {
                  setProcessed(true)
                  setViewPath(`http://localhost:8000/view/${props.id}`)
                } else if (res.data.status == 'FAILED') {
                  setProcessed(true)
                }
            } else {
                setLookupValid(false)
                setJobStatus(null)
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
                        <p><em>The results of your prediction have been emailed.</em></p>
                        <p><em>View and analyze results <a href={viewPath}>here</a></em></p>
                      </div> :
                      <div className='LookupModal-Info'>
                        <p><em>Refreshing in {counter} seconds...</em></p>
                        <LinearProgress></LinearProgress>
                        <h2>Your job is queued with status {status}</h2>
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
