import React from 'react';
import BandPlot from './BandPlot';

function VisualizationDashboard({ 
    lambdas, 
    runTrigger, 
    fermiLevel, 
    weightSigma 
}) {
    return (
        <div className="h-full w-full flex flex-col p-4">
            <div className="flex-1 min-h-0 bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
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
