import numpy as np
import time
import jax.numpy as jnp
from tbsoc.lib.json_loader import j_loader
from tbsoc.entrypoints.loadall import load_all_data
from tbsoc.lib.soc_mat import get_Hsoc
from tbsoc.lib.cal_tools import hr2hk
from tbsoc.lib.plot_tools import band_plot
from tbsoc.entrypoints.fitsoc import build_soc_basis_matrices, find_best_alignment

def plotsoc(INPUT, outdir='./', **kwargs):
    """
    Calculates SOC bands with given parameters and plots them vs DFT.
    """
    print("--- Starting Band Plotting ---")
    
    # 1. Load Data
    if isinstance(INPUT, dict):
        jdata = INPUT
    else:
        jdata = j_loader(INPUT)
    data_dict = load_all_data(**jdata)
    
    vasp_bands = data_dict['vasp_bands']
    
    # 2. Config
    efermi = jdata.get('Efermi', 0.0)
    print(f"DFT Fermi Level: {efermi} eV")

    raw_lambdas = jdata.get('lambdas')
    orb_labels = data_dict.get('orb_labels', [])

    if isinstance(raw_lambdas, dict):
        if not orb_labels:
            raise ValueError("Orbital labels missing in data, but lambdas provided as dict.")
        
        full_lambdas = np.zeros(len(orb_labels))
        for i, label in enumerate(orb_labels):
            val = raw_lambdas.get(label)
            if val is None:
                 orbital_type = label.split(':')[1] if ':' in label else label
                 if orbital_type == 's':
                     val = 0.0
                 else:
                     print(f"Warning: No value for {label}. Defaulting to 0.")
                     val = 0.0
            full_lambdas[i] = val
    elif isinstance(raw_lambdas, list):
        full_lambdas = np.array(raw_lambdas)
    else:
        raise ValueError("Invalid format for 'lambdas'. Must be list or dict.")

    # 3. Calculation
    print("Calculating TB bands...")
    
    # Build Basis Matrices (re-using logic from fitsoc logic effectively)
    # We treat all non-zero/zero params same way for plotting
    fit_indices = np.arange(len(full_lambdas)) # Use all indices to build full matrix
    
    soc_basis = build_soc_basis_matrices(
        full_lambdas, fit_indices, 
        data_dict['orbitals'], data_dict['orb_type'], 
        data_dict['orb_num'], data_dict['Msoc']
    ) 
    
    hk_tb = hr2hk(
        data_dict['hop_spinor'], data_dict['Rlatt'], 
        data_dict['kpath'], data_dict['num_wan']
    )

    hk_tb_jax = jnp.array(hk_tb)
    soc_basis_jax = jnp.array(soc_basis)
    
    # Calculate H_tot = Hk + sum(lambda * M)
    # Note: build_soc_basis_matrices returns M_i for the indices passed.
    # Here fit_indices is all indices 0..N.
    # So we just dot product full_lambdas with soc_basis
    
    h_soc = jnp.tensordot(full_lambdas, soc_basis_jax, axes=1)
    h_tot = hk_tb_jax + h_soc
    eigvals = np.array(jnp.linalg.eigvalsh(h_tot)) # (n_k, n_wan)
    
    n_wan = eigvals.shape[1]
    n_dft = vasp_bands.shape[1]
    n_k = vasp_bands.shape[0]
    
    # 4. Alignment
    print("Aligning bands...")
    best_offset, min_mse = find_best_alignment(eigvals, vasp_bands, n_wan, n_dft, n_k)
    print(f"Alignment Found: TB Bands correspond to DFT Bands {best_offset} - {best_offset + n_wan - 1}")
    
    # 5. Plotting
    print(f"Plotting to {outdir}/band.pdf ...")
    
    # Select corresponding DFT bands
    dft_subset = vasp_bands[:, best_offset : best_offset + n_wan]
    
    # band_plot args: Efermi, EMIN, EMAX, xpath, xsymm, plot_sbol, bndtb, pl_tb, pl_vasp, bndvasp, savedir
    # We should shift bands relative to Fermi? 
    # Usually band_plot takes Efermi and shifts internally (val - EFermi).
    # dft_subset is raw energy.
    # eigvals is raw energy (aligned to dft).
    
    # EMIN/EMAX for plot view
    # Prioritize EMIN/EMAX, fall back to fit_emin/max (legacy) or defaults
    emin = jdata.get('EMIN', jdata.get('fit_emin', -5)) 
    emax = jdata.get('EMAX', jdata.get('fit_emax', 5))

    band_plot(
        Efermi=efermi,
        EMIN=emin, 
        EMAX=emax, 
        xpath=data_dict['xpath'], 
        xsymm=data_dict['xsymm'], 
        plot_sbol=data_dict['plot_sbol'], 
        bndtb=eigvals, 
        pl_tb=True, 
        pl_vasp=True, 
        bndvasp=dft_subset, 
        savedir=outdir
    )
    print("Done.")
