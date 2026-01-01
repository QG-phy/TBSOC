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
            <div style={{flex: 1, minHeight: 0, background: 'var(--bg-card)', overflow: 'hidden', boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)', borderRadius: '0.5rem'}}>
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
