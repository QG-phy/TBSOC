import React, { useEffect, useRef, useState } from 'react';

export default function PlotlyComponent(props) {
    const plotRef = useRef(null);
    const [PlotlyLib, setPlotlyLib] = useState(null);
    const [error, setError] = useState(null);

    // Load Plotly dist once
    useEffect(() => {
        import('plotly.js-dist-min')
            .then(module => {
                setPlotlyLib(module.default || module);
            })
            .catch(err => {
                console.error("Failed to load Plotly:", err);
                setError(err.toString());
            });
    }, []);

    // React to props changes to update plot
    useEffect(() => {
        if (!PlotlyLib || !plotRef.current) return;

        const { data, layout, config } = props;

        // Use react-plotly.js style props mapping manually
        // Or simply Plotly.react which is efficient for updates
        try {
            PlotlyLib.react(plotRef.current, data, layout, config);
        } catch (e) {
            console.error("Plotly Error:", e);
            setError(e.toString());
        }

        // Cleanup on unmount? Plotly.purge is good practice but might flicker strictly on react re-renders
        // We generally leave it unless component unmounts
        return () => {
           // PlotlyLib.purge(plotRef.current); // Optional, avoid if causing issues
        };
    }, [PlotlyLib, props.data, props.layout, props.config, props.revision]); // Add revision if used

    // Resize handler
    useEffect(() => {
        if (!PlotlyLib || !plotRef.current) return;
        
        const handleResize = () => {
             PlotlyLib.Plots.resize(plotRef.current);
        };
        window.addEventListener('resize', handleResize);
        
        return () => window.removeEventListener('resize', handleResize);
    }, [PlotlyLib]);

    if (error) return <div style={{color: 'red', padding: '10px'}}>Chart Error: {error}</div>;
    if (!PlotlyLib) return <div style={{color: '#888', padding: '10px'}}>Loading Chart...</div>;

    return (
        <div 
            ref={plotRef} 
            style={{width: '100%', height: '100%', ...props.style}} 
            className={props.className}
        />
    );
}
