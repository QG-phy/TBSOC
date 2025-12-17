import React, { useState, useEffect } from 'react';
import Plot from './PlotlyComponent';

export default function BandPlot({ lambdas, runTrigger, fermiLevel = 0.0 }) {
    const [dftData, setDftData] = useState(null);
    const [tbData, setTbData] = useState(null);
    const [loading, setLoading] = useState(false);

    // ... (useEffect for fetching remains same, simplified here for brevity if no changes needed)
    // Actually I should just modify the render part and keep logic.
    // I'll rewrite the component function signature and render block.

    // Load DFT bands on mount or refresh
    useEffect(() => {
        fetch('/api/bands/dft')
            .then(res => {
                if (res.ok) return res.json();
                throw new Error("No DFT data");
            })
            .then(data => setDftData(data))
            .catch(err => console.log("DFT Bands not loaded yet"));
    }, [runTrigger]);

    // Calculate TB bands
    useEffect(() => {
        if (!dftData) return;
        
        setLoading(true);
        fetch('/api/bands/tb', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lambdas: lambdas })
        })
        .then(res => res.json())
        .then(data => {
            setTbData(data.bands);
            setLoading(false);
        })
        .catch(err => {
            console.error(err);
            setLoading(false);
        });

    }, [lambdas, dftData, runTrigger]);

    if (!dftData) return <div style={{padding: '20px', color: '#666'}}>Load data to see band structure</div>;

    const traces = [];

    // DFT Bands (Blue Dots/Lines)
    if (dftData) {
        const xData = dftData.k_distance || listRange(dftData.bands[0].length);
        
        dftData.bands.forEach((band, i) => {
             traces.push({
                 x: xData, 
                 y: band,
                 type: 'scatter',
                 mode: 'lines',
                 line: { color: '#1f77b4', width: 1.5, dash: 'solid' }, // Blue for DFT as requested
                 name: i === 0 ? 'DFT' : '',
                 showlegend: i === 0,
                 hoverinfo: 'y'
             });
        });
    }

    // TB Bands (Red Lines)
    if (tbData && dftData) {
         const xData = dftData.k_distance || listRange(tbData[0].length);
         tbData.forEach((band, i) => {
             traces.push({
                 x: xData,
                 y: band,
                 type: 'scatter',
                 mode: 'lines',
                 line: { color: '#e63946', width: 1 }, // Thinner red line
                 name: i === 0 ? 'TB SOC' : '',
                 showlegend: i === 0,
                 hoverinfo: 'y'
             });
        });
    }
    
    // Vertical lines for high symmetry points
    const shapes = [];
    if (dftData && dftData.k_ticks) {
        dftData.k_ticks.forEach(tick => {
            shapes.push({
                type: 'line',
                x0: tick, x1: tick,
                y0: 0, y1: 1, yref: 'paper',
                line: { color: '#ccc', width: 1, dash: 'solid' }
            });
        });
    }
    // E_Fermi line
    shapes.push({
        type: 'line',
        x0: 0, x1: 1, xref: 'paper',
        y0: fermiLevel, y1: fermiLevel,
        line: { color: '#aaa', width: 1, dash: 'dash' }
    });

    return (
        <div style={{width: '100%', height: '100%'}}>
            <Plot
                data={traces}
                layout={{
                    title: 'Electronic Band Structure',
                    autosize: true,
                    showlegend: true,
                    legend: {x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255,255,255,0.8)'},
                    xaxis: {
                        zeroline: false,
                        showgrid: false,
                        tickvals: dftData ? dftData.k_ticks : [],
                        ticktext: dftData ? dftData.k_labels : [],
                        mirror: true,
                        ticks: 'inside',
                        showline: true,
                        linecolor: 'black'
                    },
                    yaxis: {
                        title: {
                            text: 'E (eV)',
                            font: { size: 16 },
                            standoff: 0
                        },
                        automargin: true,
                        zeroline: false,
                        showgrid: false,
                        mirror: true,
                        ticks: 'inside',
                        showline: true,
                        linecolor: 'black'
                    },
                    shapes: shapes,
                    template: 'plotly_white',
                    margin: {l: 50, r: 10, t: 40, b: 40},
                    paper_bgcolor: 'white',
                    plot_bgcolor: 'white',
                    font: { family: 'Arial, sans-serif', size: 14, color: '#333' }
                }}
                useResizeHandler={true}
                style={{width: '100%', height: '100%', minHeight: '300px'}}
                config={{displayModeBar: true, displaylogo: false}}
            />
            {loading && <div style={{position: 'absolute', top: 50, left: '50%', transform: 'translateX(-50%)', background: 'rgba(255,255,255,0.9)', padding: '5px 10px', borderRadius: '4px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)'}}>Calculating...</div>}
        </div>
    );
}

function listRange(n) {
    return Array.from({length: n}, (_, i) => i);
}
