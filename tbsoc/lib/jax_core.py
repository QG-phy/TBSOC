import jax
import jax.numpy as jnp
from functools import partial

# Enable 64-bit precision which is often required for scientific calculation
jax.config.update("jax_enable_x64", True)

@jax.jit
def get_Hsoc_jax(lambdas, soc_basis_matrices):
    """
    Calculate Hsoc using pre-computed basis matrices.
    Hsoc = sum_i lambda_i * Basis_i
    
    Args:
        lambdas: Parameter array (n_params,)
        soc_basis_matrices: Pre-computed constant matrices for each parameter (n_params, n_wan, n_wan)
        
    Returns:
        Hsoc: (n_wan, n_wan) complex matrix
    """
    # Use tensordot to sum: lambdas[i] * basis[i, :, :]
    return jnp.tensordot(lambdas, soc_basis_matrices, axes=1)

@jax.jit
def loss_fn_jax(lambdas, soc_basis_matrices, hk_tb, target_bands, weights):
    """
    Weighted MSE loss function for band fitting.
    Includes automatic energy shift alignment.
    
    Args:
        lambdas: optimization parameters (n_params,)
        soc_basis_matrices: basis matrices (n_params, n_wan, n_wan)
        hk_tb: Non-SOC Hamiltonian at all k-points (nk, n_wan, n_wan)
        target_bands: Sub-set of DFT bands to match against (nk, n_wan)
        weights: Weights for each band energy point (nk, n_wan)
        
    Returns:
        loss: Scalar weighted Mean Squared Error
    """
    # 1. Construct SOC matrix
    h_soc = get_Hsoc_jax(lambdas, soc_basis_matrices)
    
    # 2. Add to non-SOC Hamiltonian (broadcast over k-points)
    h_total = hk_tb + h_soc 
    
    # 3. Diagonalize to get eigenvalues
    # eigvalsh returns eigenvalues in ascending order
    eigvals = jnp.linalg.eigvalsh(h_total) # Shape: (nk, n_wan)
    
    # 4. Energy Shift Alignment (Shift-Invariant Loss)
    # Calculate the weighted mean difference
    # shift = mean( (E_tb - E_target) * weights ) / mean(weights) ??
    # Or just simple mean difference? Simple mean is safer to avoid weight bias towards specific features.
    mean_diff = jnp.mean(eigvals - target_bands)
    eigvals_aligned = eigvals - mean_diff
    
    # 5. Calculate Difference
    diff = eigvals_aligned - target_bands
    
    # 6. Weighted MSE
    # Loss = mean( weight * (E_tb - E_dft)^2 )
    loss = jnp.mean(weights * (diff**2))
    
    return loss
