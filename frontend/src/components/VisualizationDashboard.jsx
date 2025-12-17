import React from 'react';
import BandPlot from './BandPlot';
import StructureViewer from './StructureViewer';

export default function VisualizationDashboard({ lambdas, runTrigger, fermiLevel }) {
    const [showStructure, setShowStructure] = React.useState(false);

    return (
        <div style={{display: 'flex', flexDirection: 'column', height: '100%', gap: '15px'}}>
            <div className="card" style={{flex: 1, minHeight: '300px', display: 'flex', flexDirection: 'column', padding: '10px'}}>
                 {/* Band Plot */}
                 <BandPlot lambdas={lambdas} runTrigger={runTrigger} fermiLevel={fermiLevel} />
            </div>
            
            <div className="card" style={{flex: showStructure ? 1 : '0 0 auto', minHeight: showStructure ? '300px' : 'auto', display: 'flex', flexDirection: 'column', padding: '10px', transition: 'all 0.3s ease'}}>
                 <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showStructure ? '10px' : '0'}}>
                    <h3 style={{margin: 0, fontSize: '1rem', color: '#aaa'}}>Structure Viewer</h3>
                    <button onClick={() => setShowStructure(!showStructure)} style={{background: 'none', border: 'none', color: '#42d392', cursor: 'pointer'}}>
                        {showStructure ? 'Hide' : 'Show'}
                    </button>
                 </div>
                 {showStructure && (
                    <div style={{flex: 1, minHeight: 0}}>
                        <StructureViewer runTrigger={runTrigger} />
                    </div>
                 )}
            </div>
        </div>
    );
}
