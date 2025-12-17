import React from 'react';
import BandPlot from './BandPlot';
import StructureViewer from './StructureViewer';

export default function VisualizationDashboard({ lambdas, runTrigger, fermiLevel }) {
    return (
        <div style={{display: 'flex', flexDirection: 'column', height: '100%', gap: '15px'}}>
            <div className="card" style={{flex: 1, minHeight: '300px', display: 'flex', flexDirection: 'column', padding: '10px'}}>
                 {/* Band Plot */}
                 <BandPlot lambdas={lambdas} runTrigger={runTrigger} fermiLevel={fermiLevel} />
            </div>
            
            <div className="card" style={{flex: 1, minHeight: '300px', display: 'flex', flexDirection: 'column', padding: '10px'}}>
                 {/* Structure Viewer */}
                 <StructureViewer runTrigger={runTrigger} />
            </div>
        </div>
    );
}
