import numpy as np
import time
from scipy.optimize import minimize
import jax
import jax.numpy as jnp

from tbsoc.lib.json_loader import j_loader
from tbsoc.entrypoints.loadall import load_all_data
from tbsoc.lib.soc_mat import get_Hsoc
from tbsoc.lib.cal_tools import hr2hk
from tbsoc.lib.jax_core import loss_fn_jax

def build_soc_basis_matrices(full_lambdas, fit_indices, orbitals, orb_type, orb_num, Msoc):
    """
    Construct the constant matrices M_i such that H_soc = sum_i lambda_i * M_i.
    Returns: Stacked array (n_fit_params, n_wan, n_wan)
    """
    basis_matrices = []
    
    # We want to isolate the effect of each fitted lambda.
    # We do this by calling get_Hsoc with a "one-hot" lambda vector.
    # Note: get_Hsoc logic is linear in lambdas.
    
    n_params = len(full_lambdas)
    
    # Check if there is a 'background' SOC from non-fitted parameters?
    # Current assumption: non-fitted parameters are zero. 
    # If they are non-zero but fixed, we should calculate a "static_soc" matrix.
    # But usually fitsoc optimizes all non-zero values.
    # We will strictly follow the indices.
    
    for idx_in_full in fit_indices:
        # Create a dummy lambda vector with 1.0 at the current index
        temp_lambdas = np.zeros(n_params)
        temp_lambdas[idx_in_full] = 1.0
        
        # Calculate H_soc for this single unit parameter
        h_basis = get_Hsoc(temp_lambdas, orbitals, orb_type, orb_num, Msoc)
        basis_matrices.append(h_basis)
        
    return np.array(basis_matrices)

def find_best_alignment(tb_energies_flat, dft_bands_flat, n_tb, n_dft_bands, n_kpoints):
    """
    Finds the best integer offset N such that TB[0] aligns with DFT[N].
    Assumes bands are sorted.
    
    Args:
        tb_energies_flat: (nk * n_tb) or (nk, n_tb)
        dft_bands_flat: (nk, n_dft)
    """
    # Reshape for easier broadcasting if needed, but mean squared error is robust
    tb_mean = np.mean(tb_energies_flat)
    
    best_offset = -1
    min_mse = float('inf')
    
    max_offset = n_dft_bands - n_tb
    
    if max_offset < 0:
        raise ValueError(f"DFT bands count ({n_dft_bands}) is smaller than TB bands ({n_tb}). Cannot fit.")
    
    print(f"Scanning offsets 0 to {max_offset}...")
    
    for offset in range(max_offset + 1):
        # Extract the window of DFT bands
        target_window = dft_bands_flat[:, offset : offset + n_tb]
        
        target_mean = np.mean(target_window)
        
        # Center them for fair comparison (Shift-Invariant alignment)
        # We align the centers of mass of the band structures
        diff = (tb_energies_flat - tb_mean) - (target_window - target_mean)
        
        mse = np.mean(diff**2)
        
        if mse < min_mse:
            min_mse = mse
            best_offset = offset
            
    return best_offset, min_mse

def fitsoc(INPUT, outdir='./', **kwargs):
    """
    Fits SOC parameters using JAX-accelerated gradient descent.
    """
    start_time = time.time()
    print("--- Starting JAX-accelerated SOC fitting ---")
    
    # 1. Load Data
    jdata = j_loader(INPUT)
    data_dict = load_all_data(**jdata)
    
    vasp_bands = data_dict['vasp_bands'] # Shape: (n_k, n_dft_bands) or similar? 
    # Usually vasp_bands in tbsoc is (n_dft, n_k)? No, let's check `read_in.py`.
    # `read_EIGENVAL` returns `k_bands` = `np.array(k_bands)`.
    # `k_bands` logic: `k_bands.append(sorted(kb_temp))`.
    # `kb_temp` is one k-point. So `k_bands` is (n_k, n_bands).
    # Correct.
    
    # 2. Config & Pre-calculation
    efermi = jdata.get('Efermi')
    sigma = jdata.get('weight_sigma', 2.0) # Energy weighting width
    print(f"DFT Fermi Level: {efermi} eV. Weighting sigma: {sigma} eV")

    initial_full_lambdas = np.array(jdata.get('lambdas'))
    fit_indices = np.where(initial_full_lambdas != 0)[0]
    initial_params = initial_full_lambdas[fit_indices]
    
    if len(initial_params) == 0:
        print("No non-zero lambdas found. Nothing to fit.")
        return

    print(f"Optimizing {len(initial_params)} parameters at indices: {fit_indices}")
    print("Pre-calculating Hamiltonian components...")
    
    # A. Build Basis Matrices
    soc_basis = build_soc_basis_matrices(
        initial_full_lambdas, fit_indices, 
        data_dict['orbitals'], data_dict['orb_type'], 
        data_dict['orb_num'], data_dict['Msoc']
    ) # (n_params, n_wan, n_wan)
    
    # B. Build Non-SOC Hk
    hk_tb = hr2hk(
        data_dict['hop_spinor'], data_dict['Rlatt'], 
        data_dict['kpath'], data_dict['num_wan']
    ) # (n_k, n_wan, n_wan)

    # Convert to JAX arrays
    hk_tb_jax = jnp.array(hk_tb)
    soc_basis_jax = jnp.array(soc_basis)
    
    # 3. Phase 1: Alignment (using Initial Guess)
    print("Phase 1: Band Alignment...")
    # Calculate initial TB bands
    # We can use the JAX function (compiled) for this
    initial_loss_dummy = loss_fn_jax(
        initial_params, soc_basis_jax, hk_tb_jax, 
        np.zeros((hk_tb.shape[0], hk_tb.shape[1])), np.zeros((hk_tb.shape[0], hk_tb.shape[1]))
    ) # Compile trigger (optional)
    
    h_soc_init = jnp.tensordot(initial_params, soc_basis_jax, axes=1)
    h_tot_init = hk_tb_jax + h_soc_init
    eigvals_init = np.array(jnp.linalg.eigvalsh(h_tot_init)) # (n_k, n_wan)
    
    n_wan = eigvals_init.shape[1]
    n_dft = vasp_bands.shape[1]
    n_k = vasp_bands.shape[0]
    
    # Perform Alignment Logic
    best_offset, min_mse = find_best_alignment(eigvals_init, vasp_bands, n_wan, n_dft, n_k)
    print(f"Alignment Found: TB Bands correspond to DFT Bands {best_offset} - {best_offset + n_wan - 1}")
    print(f"Initial MSE (Unweighted): {min_mse:.6f}")
    
    # 4. Phase 2: Optimization
    print("Phase 2: Optimization...")
    
    # Prepare Target and Weights
    target_bands = vasp_bands[:, best_offset : best_offset + n_wan]
    
    # Calculate Weights: exp(-|E - Ef| / sigma)
    # Efermi is relative to what? vasp_bands were shifted by read_EIGENVAL if EFERMI was 0?
    # read_in.py had EFERMI=0. So vasp_bands are raw.
    # We should assume 'efermi' provided in JSON is the absolute value in VASP energy scale.
    weights_np = np.exp(-np.abs(target_bands - efermi) / sigma)
    
    # Convert constraints to JAX
    target_bands_jax = jnp.array(target_bands)
    weights_jax = jnp.array(weights_np)
    
    # Define Gradient Function
    val_and_grad_fn = jax.value_and_grad(loss_fn_jax)
    
    def scipy_fun(x):
        # Wrapper to bridge JAX and Scipy
        # x is float64 usually
        v, g = val_and_grad_fn(x, soc_basis_jax, hk_tb_jax, target_bands_jax, weights_jax)
        return float(v), np.array(g, dtype=np.float64)

    # Run Optimization
    res = minimize(
        scipy_fun, 
        initial_params, 
        method='L-BFGS-B', 
        jac=True,
        options={'disp': True, 'maxiter': 200}
    )
    
    # 5. Output Results
    final_params = res.x
    final_full_lambdas = np.copy(initial_full_lambdas)
    final_full_lambdas[fit_indices] = final_params
    
    print("\n--- Optimization Finished ---")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Final Weighted MSE: {res.fun:.6f}")
    print(f"Optimized Lambdas: {final_full_lambdas}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    return res