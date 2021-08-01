import React from 'react';
import {BrowserRouter as Router, Route } from 'react-router-dom';

import Home from './components/pages/Home';
import SinglePairPredict from './components/pages/SinglePairPredict';
import ManyPairPredict from './components/pages/ManyPairPredict';
import AllPairPredict from './components/pages/AllPairPredict';
import Predict from './components/pages/Predict';
import Lookup from './components/pages/Lookup';
import './styles.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Route path='/' exact component={Home}/>
        <Route path='/single-pair' exact component={SinglePairPredict}/>
        <Route path='/many-pair' exact component={ManyPairPredict}/>
        <Route path='/all-pair' exact component={AllPairPredict}/>
        <Route path='/predict' exact component={Predict}/>
        <Route path='/lookup' exact component={Lookup}/>
      </div>
    </Router>
  );
}

export default App;
