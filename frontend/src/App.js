import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: 'Failed to get prediction' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sentiment Analysis</h1>
        <div className="container">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze sentiment..."
            rows={6}
            cols={50}
          />
          <br />
          <button 
            onClick={handlePredict} 
            disabled={loading || !text.trim()}
            className="predict-btn"
          >
            {loading ? 'Analyzing...' : 'Predict'}
          </button>
          
          {result && (
            <div className="result">
              {result.error ? (
                <p className="error">{result.error}</p>
              ) : (
                <div>
                  <h3>Result:</h3>
                  <p className={`sentiment ${result.label}`}>
                    <strong>Label:</strong> {result.label}
                  </p>
                  <p>
                    <strong>Confidence:</strong> {(result.score * 100).toFixed(2)}%
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;