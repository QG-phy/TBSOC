import React from 'react';
import BandPlot from './BandPlot';

function VisualizationDashboard({ 
    lambdas, 
    runTrigger,
    dataVersion,
    fermiLevel, 
    weightSigma,
    themeConfig 
}) {
    return (
        <div style={{height: '100%', width: '100%', display: 'flex', flexDirection: 'column'}}>
            <div className="card" style={{flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', padding: '10px'}}>
                <BandPlot 
                    lambdas={lambdas}
                    runTrigger={runTrigger}
                    dataVersion={dataVersion}
                    fermiLevel={fermiLevel}
                    weightSigma={weightSigma}
                    themeConfig={themeConfig}
                />
            </div>
        </div>
    );
}

export default VisualizationDashboard;
