import React, { useState } from 'react'
import ParameterEditor from './components/ParameterEditor'
import VisualizationDashboard from './components/VisualizationDashboard'
import './App.css'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{padding: '20px', color: '#ff6b6b'}}>
          <h1>Something went wrong.</h1>
          <pre>{this.state.error && this.state.error.toString()}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [status, setStatus] = useState("Idle");
  const [activeLambdas, setActiveLambdas] = useState([]); 
  const [activeFermi, setActiveFermi] = useState(0.0);
  const [activeSigma, setActiveSigma] = useState(2.0);
  const [runTrigger, setRunTrigger] = useState(0);

  const runFit = async (data) => {
    setStatus("Submitting...");
    setActiveFermi(parseFloat(data.Efermi) || 0.0);
    setActiveSigma(parseFloat(data.weight_sigma) || 2.0);
    try {
        const res = await fetch('/api/fit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        if (res.ok) {
            setStatus("Fitting Started");
            pollStatus();
        } else {
            const err = await res.json();
            setStatus("Error: " + err.detail);
        }
    } catch (e) {
        setStatus("Network Error: " + e.message);
    }
  };

  const pollStatus = () => {
      const interval = setInterval(async () => {
          try {
              const res = await fetch('/api/status');
              const data = await res.json();
              
              if (data.status === 'completed') {
                  clearInterval(interval);
                  setStatus("Fitting Completed");
                  if (data.result && data.result.optimized_lambdas) {
                      const newLambdas = data.result.optimized_lambdas;
                      setActiveLambdas(newLambdas);
                      setRunTrigger(prev => prev + 1);
                      // ParameterEditor updates via externalLambdas
                  }
              } else if (data.status === 'error') {
                  clearInterval(interval);
                  setStatus("Error: " + data.message);
              } else {
                  setStatus(`Fitting... ${Math.round(data.progress * 100)}%`);
              }
          } catch (e) {
              console.error(e);
          }
      }, 1000);
  };

  const handlePreview = (formData) => {
      // Called when user clicks "Preview Bands"
      console.log("Previewing with:", formData.lambdas);
      setActiveLambdas(formData.lambdas);
      setActiveFermi(parseFloat(formData.Efermi) || 0.0);
      setActiveSigma(parseFloat(formData.weight_sigma) || 2.0);
      setRunTrigger(prev => prev + 1); // Trigger refresh
      setStatus("Previewing...");
  };

  const handleStopFit = async () => {
      setStatus("Stopping...");
      try {
          await fetch('/api/fit/stop', { method: 'POST' });
          // Poll will eventually see "cancelled" or "error"
      } catch (e) {
          console.error("Failed to stop fitting:", e);
      }
  };

  return (
    <ErrorBoundary>
    <div className="container" style={{maxWidth: '100%', padding: '20px', height: '100vh', boxSizing: 'border-box', display: 'flex', flexDirection: 'column'}}>
      <header style={{textAlign: 'center', marginBottom: '1em', flexShrink: 0}}>
        <h1 style={{margin: '0.2em 0', fontSize: '2em'}}>TBSOC Desktop</h1>
        <div style={{display: 'flex', justifyContent: 'center', gap: '20px', alignItems: 'center'}}>
            <p style={{color: '#888', margin: 0}}>Tight-Binding SOC Fitter</p>
            <div style={{padding: '2px 8px', background: '#333', borderRadius: '4px', fontSize: '0.8rem'}}>
                Status: <span style={{color: status.includes('Error') ? '#ff6b6b' : '#42d392'}}>{status}</span>
            </div>
        </div>
      </header>
      
      <div style={{display: 'flex', gap: '20px', flex: 1, minHeight: 0}}>
          <div style={{flex: '0 0 400px', minWidth: '350px'}}>
            <ParameterEditor 
                onRunFit={runFit} 
                onPreview={handlePreview} 
                externalLambdas={activeLambdas} 
                isFitting={status.startsWith("Fitting") || status === "Submitting..."}
                onStopFit={handleStopFit}
            />
          </div>
          <div style={{flex: 1, minWidth: 0}}>
            <VisualizationDashboard 
                lambdas={activeLambdas} 
                runTrigger={runTrigger} 
                fermiLevel={activeFermi} 
                weightSigma={activeSigma}
            />
          </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}

export default App
