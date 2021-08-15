import React from 'react';
import {BrowserRouter as Router, Route } from 'react-router-dom';

import Home from './components/pages/Home';
import Predict from './components/pages/Predict';
import Lookup from './components/pages/Lookup';
import './styles.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Route path='/' exact component={Home}/>
        <Route path='/predict' exact component={Predict}/>
        <Route path='/lookup' exact component={Lookup}/>
      </div>
    </Router>
  );
}

export default App;
