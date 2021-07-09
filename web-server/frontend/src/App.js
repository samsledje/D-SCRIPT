import React from 'react';
import {BrowserRouter as Router, Route } from 'react-router-dom';

import Home from './components/pages/Home';
import SinglePairPredict from './components/pages/SinglePairPredict';
import ManyPairsPredict from './components/pages/ManyPairsPredict';
import './styles.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Route path='/' exact component={Home}/>
        <Route path='/pair' exact component={SinglePairPredict}/>
        <Route path='/pairs' exact component={ManyPairsPredict}/>
      </div>
    </Router>
  );
}

export default App;
