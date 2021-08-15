import React, { useState } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';

export default function AllPair() {

    const [item, setItem] = useState({
        'title': '',
        'sequences': null
    });

    const handleTitleChange = (e) => {
        setItem({...item, 'title': e.target.value});
    }

    const handleSequencesChange = (e) => {
        setItem({...item, 'sequences': e.target.files[0]})
    }

    const handleSubmit = () => {
        console.log(item)
        const csrftoken = Cookies.get('csrftoken');
        const uploadData = new FormData()
        uploadData.append('title', item.title)
        uploadData.append('sequences', item.sequences)
        axios
            .post(
                "http://localhost:8000/api/all-pair/", 
                uploadData,
                {
                    headers: {'X-CSRFToken': csrftoken}
                }
                )
            .then((res) => console.log(res))
            .catch((err) => console.log(err))
    }

    return (
        <div className="AllPair-Container">
            <h1>Predict Interaction Between All Pairs</h1>
            <form>
                <label>Enter a job title:</label><br></br>
                <input type="text" onChange={handleTitleChange}></input><br></br>
                <label>Protein sequences (.fasta)</label><br></br>
                <input type="file" accept=".fasta" onChange={handleSequencesChange}></input><br></br>
            </form>
            <button onClick={handleSubmit}>Compute Interaction Probability</button>
        </div>
    )
}
