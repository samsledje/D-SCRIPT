import React from 'react'
import { makeStyles, Modal, Backdrop, Fade, LinearProgress } from '@material-ui/core'

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
                    <div className='SubmissionModal-Info'>
                      <p>Refreshing in 120 seconds...</p>
                      <LinearProgress></LinearProgress>
                      <h2>Your job is queued in position {props.position}</h2>
                      <p>Job id: {props.id}</p>
                      <p><em>Make note of this job id for the purpose of tracking your job.</em></p>
                    </div>
                  </div>
                </Fade>
              </Modal>
        </div>
    )
}
