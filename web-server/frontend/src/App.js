import React from 'react';
import {BrowserRouter as Router, Route } from 'react-router-dom';

import Home from './components/pages/Home'
import './styles.css'

function App() {
  return (
    <Router>
      <div className="App">
        <Route path='/' exact component={Home}/>
      </div>
    </Router>
  );
}

export default App;
