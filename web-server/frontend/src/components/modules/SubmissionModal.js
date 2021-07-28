import React from 'react'
import { makeStyles, Modal, Backdrop, Fade } from '@material-ui/core'

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
                // onClose={handleClose}
                // closeAfterTransition
                BackdropComponent={Backdrop}
                BackdropProps={{
                  timeout: 500,
                }}
              >
                <Fade in={props.open}>
                  <div className={classes.paper}>
                    <h2>Basic transition modal</h2>
                    <p>This transitional modal is a placeholder right now</p>
                  </div>
                </Fade>
              </Modal>
        </div>
    )
}