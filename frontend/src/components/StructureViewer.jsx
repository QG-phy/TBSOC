import React, { useEffect, useRef, useState } from 'react';
import * as $3Dmol from '3dmol/build/3Dmol.js';

export default function StructureViewer({ runTrigger }) {
    const [structure, setStructure] = useState(null);
    const viewerRef = useRef(null);
    const elementRef = useRef(null);

    // Fetch Structure Data
    useEffect(() => {
        fetch('/api/structure')
            .then(res => res.json())
            .then(data => {
                // If API returns valid structure
                if (data && data.lattice) {
                     setStructure(data);
                }
            })
            .catch(err => console.error("Failed to fetch structure:", err));
    }, [runTrigger]);

    // Initialize & Update 3Dmol Viewer
    useEffect(() => {
        if (!structure || !elementRef.current || !structure.poscar_content) return;

        // Initialize viewer if not exists
        if (!viewerRef.current) {
            const config = { backgroundColor: 'white' };
            viewerRef.current = $3Dmol.createViewer(elementRef.current, config);
        }
        
        const viewer = viewerRef.current;
        viewer.clear();
        
        // Add model from POSCAR string
        viewer.addModel(structure.poscar_content, "vasp");
        
        // Style atoms
        viewer.setStyle({}, {sphere: {scale: 0.3}, stick: {radius: 0.1}});
        
        // Add unit cell box
        viewer.addUnitCell();

        // Zoom to fit
        viewer.zoomTo();
        viewer.render();

    }, [structure]);

    return (
        <div style={{width: '100%', height: '100%', position: 'relative', background: 'white', borderRadius: '8px', overflow: 'hidden'}}>
            <div ref={elementRef} style={{width: '100%', height: '100%'}}></div>
            {!structure && <div style={{position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#666'}}>No Structure Loaded</div>}
            {structure && !structure.poscar_content && <div style={{position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#666'}}>POSCAR Content Missing</div>}
        </div>
    );
}
