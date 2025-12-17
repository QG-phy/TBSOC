import React from 'react';
import BandPlot from './BandPlot';

function VisualizationDashboard({ 
    lambdas, 
    runTrigger, 
    fermiLevel, 
    weightSigma 
}) {
    return (
        <div style={{height: '100%', width: '100%', display: 'flex', flexDirection: 'column'}}>
            <div style={{flex: 1, minHeight: 0, background: 'white', overflow: 'hidden', boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)', borderRadius: '0.5rem'}}>
                <BandPlot 
                    lambdas={lambdas}
                    runTrigger={runTrigger}
                    fermiLevel={fermiLevel}
                    weightSigma={weightSigma}
                />
            </div>
        </div>
    );
}

export default VisualizationDashboard;
