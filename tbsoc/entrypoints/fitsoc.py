import numpy as np
from scipy.optimize import minimize
from tbsoc.lib.json_loader import j_loader
from tbsoc.entrypoints.loadall import load_all_data
from tbsoc.lib.soc_mat import get_Hsoc
from tbsoc.lib.cal_tools import hr2hk

def find_best_match(target_band, candidate_bands):
    """
    Finds the best matching candidate band for a target band.

    Args:
        target_band (np.array): A 1D array of energies for the target band.
        candidate_bands (np.array): A 2D array (n_bands, n_kpoints) of candidate bands.

    Returns:
        tuple: The index of the best match and the minimum RMSE.
    """
    min_rmse = float('inf')
    best_match_idx = -1
    
    target_avg = np.mean(target_band)

    for i, candidate_band in enumerate(candidate_bands):
        candidate_avg = np.mean(candidate_band)
        # Align the candidate band to the target band's average energy
        aligned_candidate = candidate_band + (target_avg - candidate_avg)
        rmse = np.sqrt(np.mean((target_band - aligned_candidate)**2))
        
        if rmse < min_rmse:
            min_rmse = rmse
            best_match_idx = i
            
    return best_match_idx, min_rmse

def fitsoc(INPUT, outdir='./', **kwargs):
    """
    This function automatically fits the SOC parameters (lambdas) by comparing
    the tight-binding band structure with DFT calculations.
    """
    print("--- Starting automatic SOC fitting ---")
    
    # 1. Load all data and fitting parameters
    jdata = j_loader(INPUT)
    data_dict = load_all_data(**jdata)
    
    vasp_bands = data_dict['vasp_bands']
    efermi = jdata.get('Efermi')
    if efermi is None:
        raise ValueError("Efermi must be provided in the input JSON for fitting.")

    fit_emin = jdata.get('fit_emin', -5.0)
    fit_emax = jdata.get('fit_emax', 5.0)
    
    # Identify which lambdas to optimize (the non-zero ones)
    initial_full_lambdas = np.array(jdata.get('lambdas'))
    fit_indices = np.where(initial_full_lambdas != 0)[0]
    initial_fit_lambdas = initial_full_lambdas[fit_indices]

    print(f"Energy window for fitting: [{fit_emin}, {fit_emax}] eV around Efermi.")
    print(f"Initial guess for lambdas to be fitted: {initial_fit_lambdas}")
    print(f"Fitting will optimize the lambdas at indices: {fit_indices}")

    # 2. Filter DFT bands to get the "target bands"
    e_min_abs = efermi + fit_emin
    e_max_abs = efermi + fit_emax
    
    target_dft_indices = [
        i for i, band in enumerate(vasp_bands.T) 
        if np.any((band > e_min_abs) & (band < e_max_abs))
    ]
    target_dft_bands = vasp_bands[:, target_dft_indices]
    print(f"Found {len(target_dft_indices)} DFT bands in the energy window.")

    # 3. Find the DFT anchor band (closest to Efermi)
    avg_energies = np.mean(target_dft_bands, axis=0)
    anchor_dft_local_idx = np.argmin(np.abs(avg_energies - efermi))
    anchor_dft_global_idx = target_dft_indices[anchor_dft_local_idx]
    anchor_dft_band = vasp_bands[:, anchor_dft_global_idx]
    print(f"Using DFT band {anchor_dft_global_idx} as the anchor band.")

    # 4. Define the objective function for the optimizer
    def objective_function(trial_fit_lambdas):
        # a. Reconstruct the full lambda array
        full_lambdas = np.copy(initial_full_lambdas)
        full_lambdas[fit_indices] = trial_fit_lambdas

        # b. Construct candidate TB+SOC bands
        hsoc = get_Hsoc(full_lambdas, data_dict['orbitals'], data_dict['orb_type'], data_dict['orb_num'], data_dict['Msoc'])
        hksoc = hr2hk(data_dict['hop_spinor'], data_dict['Rlatt'], data_dict['kpath'], data_dict['num_wan']) + hsoc
        bands_tb_soc = np.linalg.eigvalsh(hksoc)

        # c. Find the TB band that best matches the DFT anchor band
        best_tb_anchor_idx, _ = find_best_match(anchor_dft_band, bands_tb_soc.T)
        
        # d. Establish the index mapping
        index_offset = best_tb_anchor_idx - anchor_dft_global_idx
        
        # e. Calculate total loss using the fixed mapping
        total_loss = 0.0
        num_bands_in_loss = 0
        for i, dft_idx in enumerate(target_dft_indices):
            tb_idx = dft_idx + index_offset
            if 0 <= tb_idx < bands_tb_soc.shape[1]:
                dft_band = target_dft_bands[:, i]
                tb_band = bands_tb_soc[:, tb_idx]
                
                dft_avg = np.mean(dft_band)
                tb_avg = np.mean(tb_band)
                aligned_tb_band = tb_band + (dft_avg - tb_avg)
                
                total_loss += np.sqrt(np.mean((dft_band - aligned_tb_band)**2))
                num_bands_in_loss += 1
        
        return total_loss / num_bands_in_loss if num_bands_in_loss > 0 else float('inf')

    # 5. Run the optimization
    print("Starting optimization...")
    res = minimize(objective_function, initial_fit_lambdas, method='Nelder-Mead', options={'disp': True})

    # 6. Print the results
    final_full_lambdas = np.copy(initial_full_lambdas)
    final_full_lambdas[fit_indices] = res.x
    print("\n--- Optimization Finished ---")
    print(f"Final Loss: {res.fun}")
    print(f"Optimized Lambdas: {final_full_lambdas}")
    
    return res