import React, { useState, useEffect } from 'react';
import Plot from './PlotlyComponent';

export default function BandPlot({ lambdas, runTrigger, fermiLevel = 0.0, weightSigma = 2.0 }) {
    const [dftData, setDftData] = useState(null);
    const [tbData, setTbData] = useState(null);
    const [mae, setMae] = useState(null);
    const [loading, setLoading] = useState(false);

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
            setMae(data.mae);
            setLoading(false);
        })
        .catch(err => {
            console.error(err);
            setLoading(false);
        });

    }, [lambdas, dftData, runTrigger]);

    if (!dftData) return <div style={{padding: '20px', color: '#666'}}>Load data to see band structure</div>;

    const traces = [];

    // --- Band Plot Traces (Left Subplot) ---
    // xaxis: 'x', yaxis: 'y'

    // DFT Bands (Blue Dashed)
    if (dftData) {
        const xData = dftData.k_distance || listRange(dftData.bands[0].length);
        
        dftData.bands.forEach((band, i) => {
             traces.push({
                 x: xData, 
                 y: band,
                 xaxis: 'x',
                 yaxis: 'y',
                 type: 'scatter',
                 mode: 'lines',
                 line: { color: '#89B4FA', width: 2, dash: 'dash' },
                 name: i === 0 ? 'DFT (VASP)' : '',
                 showlegend: i === 0,
                 hoverinfo: 'y',
                 opacity: 0.6
             });
        });
    }

    // TB Bands (Red Solid)
    if (tbData && dftData) {
         const xData = dftData.k_distance || listRange(tbData[0].length);
         tbData.forEach((band, i) => {
             traces.push({
                 x: xData,
                 y: band,
                 xaxis: 'x',
                 yaxis: 'y',
                 type: 'scatter',
                 mode: 'lines',
                 line: { color: '#F38BA8', width: 1.5, dash: 'solid' },
                 name: i === 0 ? 'TB+SOC' : '',
                 showlegend: i === 0,
                 hoverinfo: 'y'
             });
        });
    }
    
    // --- Vertical Lines (Shapes) ---
    const shapes = [];
    if (dftData && dftData.k_ticks) {
        dftData.k_ticks.forEach(tick => {
            shapes.push({
                type: 'line',
                x0: tick, x1: tick,
                y0: 0, y1: 1, yref: 'paper', // Reference entire plotting area usually, but with subplot?
                // For subplots, shapes need xref/yref specific to the axis or paper.
                // Using 'paper' for y means 0-1 covers the subplot height if subplot takes full height.
                // x0/x1 are in data coordinates of xaxis.
                xref: 'x', 
                line: { color: 'grey', width: 1, dash: 'solid' }
            });
        });
    }
    
    // E_Fermi line (Left Plot)
    shapes.push({
        type: 'line',
        xref: 'x',
        x0: dftData.k_distance ? dftData.k_distance[0] : 0, 
        x1: dftData.k_distance ? dftData.k_distance[dftData.k_distance.length - 1] : 1, 
        y0: fermiLevel, y1: fermiLevel,
        line: { color: '#aaa', width: 1, dash: 'dash' }
    });
    
    // --- Weight Function Trace (Right Subplot) ---
    // xaxis: 'x2', yaxis: 'y' (Shared Y axis)
    
    // Determine Y range from DFT data min/max
    let minY = -10, maxY = 10;
    if (dftData && dftData.bands) {
        const allY = dftData.bands.flat();
        minY = Math.min(...allY);
        maxY = Math.max(...allY);
    }
    
    // Generate Energy points for weight curve (vertical resolution)
    const numPoints = 200;
    const eStep = (maxY - minY) / numPoints;
    const yWeight = [];
    const xWeight = [];
    
    for (let i = 0; i <= numPoints; i++) {
        const E = minY + i * eStep;
        // Gaussian: exp(-(x-u)^2 / (2*s^2))
        const sigma = weightSigma || 1e-6;
        const W = Math.exp(-Math.pow(E - fermiLevel, 2) / (2 * sigma * sigma));
        yWeight.push(E);
        xWeight.push(W);
    }
    
    traces.push({
        x: xWeight,
        y: yWeight,
        xaxis: 'x2',
        yaxis: 'y', // Share Y axis
        type: 'scatter',
        mode: 'lines',
        fill: 'tozerox', // Fill towards x=0
        line: { color: '#42d392', width: 1.5 },
        fillcolor: 'rgba(66, 211, 146, 0.2)',
        name: 'Weight',
        showlegend: false,
        hoverinfo: 'x+y'
    });
    
    // E_Fermi line (Right Plot)
    shapes.push({
        type: 'line',
        xref: 'x2',
        x0: 0, x1: 1,
        y0: fermiLevel, y1: fermiLevel,
        line: { color: '#aaa', width: 1, dash: 'dash' }
    });

    return (
        <div style={{position: 'relative', width: '100%', height: '100%'}}>
            <Plot
                data={traces}
                layout={{
                    title: { text: null }, 
                    autosize: true,
                    showlegend: true,
                    legend: {x: 0, xanchor: 'left', y: 1, bgcolor: 'rgba(30,30,46,0.8)', bordercolor: '#313244', borderwidth: 1, font: {color: '#D9E0EE'}},
                    
                    // --- Axis Definitions ---
                    xaxis: {
                        domain: [0, 0.78],
                        range: dftData ? [dftData.k_distance[0], dftData.k_distance[dftData.k_distance.length - 1]] : undefined,
                        zeroline: false,
                        showgrid: false,
                        tickvals: dftData ? dftData.k_ticks : [],
                        ticktext: dftData ? dftData.k_labels : [],
                        mirror: true,
                        ticks: 'inside',
                        showline: true,
                        linecolor: '#6E738D',
                        tickcolor: '#6E738D',
                        linewidth: 1.5,
                        tickfont: { size: 14, color: '#D9E0EE' }
                    },
                    yaxis: {
                        title: { text: 'E (eV)', font: { size: 16, color: '#D9E0EE' }, standoff: 10 },
                        zeroline: false,
                        showgrid: false,
                        mirror: true,
                        ticks: 'inside',
                        showline: true,
                        linecolor: '#6E738D',
                        tickcolor: '#6E738D',
                        linewidth: 1.5,
                        tickfont: { size: 14, color: '#D9E0EE' }
                    },
                    
                    xaxis2: {
                        domain: [0.82, 1.0],
                        title: { text: 'Weight', font: { size: 12, color: '#D9E0EE' }, standoff: 0 },
                        range: [0, 1.05],
                        zeroline: false,
                        showgrid: true,
                        gridcolor: '#313244',
                        dtick: 0.5,
                        mirror: true,
                        ticks: 'inside',
                        showline: true,
                        linecolor: '#6E738D',
                        tickcolor: '#6E738D',
                        linewidth: 1.5,
                        tickfont: { size: 12, color: '#D9E0EE' },
                        side: 'bottom'
                    },
                    
                    shapes: shapes,
                    template: 'plotly_dark',
                    margin: {l: 50, r: 10, t: 20, b: 30},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { family: 'Inter, sans-serif', size: 14, color: '#D9E0EE' },
                    annotations: mae !== null ? [{
                        xref: 'paper', yref: 'paper',
                        x: 0.98, y: 0.98, 
                        xanchor: 'right',
                        text: `MAE: ${mae.toFixed(4)} eV`,
                        showarrow: false,
                        font: { size: 14, color: '#42D392', family: 'JetBrains Mono, monospace' },
                        bgcolor: 'rgba(30,30,46,0.9)',
                        borderpad: 6,
                        bordercolor: '#42D392',
                        borderwidth: 1,
                        align: 'right'
                    }] : []
                }}
                useResizeHandler={true}
                style={{width: '100%', height: '100%', minHeight: '300px'}}
                config={{displayModeBar: true, displaylogo: false}}
            />
            {loading && <div style={{position: 'absolute', top: 50, left: '40%', background: 'rgba(255,255,255,0.9)', padding: '5px 10px', borderRadius: '4px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)'}}>Calculating...</div>}
        </div>
    );
}

function listRange(n) {
    return Array.from({length: n}, (_, i) => i);
}
