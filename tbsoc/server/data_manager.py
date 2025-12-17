from tbsoc.entrypoints.loadall import load_all_data
from tbsoc.lib.soc_mat import get_Hsoc
from tbsoc.lib.cal_tools import hr2hk
from tbsoc.entrypoints.fitsoc import build_soc_basis_matrices
import numpy as np
import jax.numpy as jnp
import os

class DataManager:
    def __init__(self):
        self.data_dict = None
        self.config_hash = None
        self.hk_tb_jax = None
        self.soc_basis_jax = None
        self.kpath_flat = None # Could be useful for plotting x-axis
        
    def load_if_needed(self, config_dict):
        # simple caching based on file paths
        # In real app, might want to check file mtimes
        current_hash = str(config_dict)
        if self.data_dict is not None and self.config_hash == current_hash:
            return

        print("DataManager: Loading data...")
        self.data_dict = load_all_data(**config_dict)
        self.config_hash = current_hash
        
        # Pre-calculate similar to fitsoc
        self._precalculate(config_dict)
        print("DataManager: Data loaded and pre-calculated.")

    def _precalculate(self, config_dict):
        initial_full_lambdas = np.array(config_dict.get('lambdas', []))
        
        fit_indices = np.where(initial_full_lambdas != 0)[0]
        
        if len(fit_indices) == 0:
             # Just build TB
             self.soc_basis = np.array([])
        else:
             self.soc_basis = build_soc_basis_matrices(
                initial_full_lambdas, fit_indices, 
                self.data_dict['orbitals'], self.data_dict['orb_type'], 
                self.data_dict['orb_num'], self.data_dict['Msoc']
            )

        self.hk_tb = hr2hk(
            self.data_dict['hop_spinor'], self.data_dict['Rlatt'], 
            self.data_dict['kpath'], self.data_dict['num_wan']
        )
        
        self.hk_tb_jax = jnp.array(self.hk_tb)
        self.soc_basis_jax = jnp.array(self.soc_basis)
        self.fit_indices = fit_indices

        # --- Automatic Alignment (Same as plotsoc) ---
        # Calculate initial bands to find alignment
        if len(fit_indices) > 0:
            current_params = initial_full_lambdas[fit_indices]
            h_soc = jnp.tensordot(current_params, self.soc_basis_jax, axes=1)
            h_total = self.hk_tb_jax + h_soc
        else:
            h_total = self.hk_tb_jax
            
        eigvals = np.array(jnp.linalg.eigvalsh(h_total)) # (n_k, n_wan)
        
        vasp_bands = self.data_dict['vasp_bands']
        n_wan = eigvals.shape[1]
        n_dft = vasp_bands.shape[1]
        n_k = vasp_bands.shape[0]
        
        # We need find_best_alignment here. 
        # Note: Importing inside method to avoid circular import if needed, or top level.
        # Assuming top level import is safe (loadall imports fitsoc, but fitsoc imports soc_mat...)
        # find_best_alignment is in fitsoc.
        
        print("DataManager: Aligning bands...")
        from tbsoc.entrypoints.fitsoc import find_best_alignment
        best_offset, min_mse = find_best_alignment(eigvals, vasp_bands, n_wan, n_dft, n_k)
        self.best_offset = best_offset
        print(f"DataManager: Alignment Found. Offset: {best_offset}")


    @property
    def orb_labels(self):
        return self.data_dict.get('orb_labels', [])

    def get_structure(self):
        if not self.data_dict: return None

        # Read raw POSCAR file as backup, but generate XYZ for reliable 3Dmol visualization
        poscar_content = ""
        try:
             with open(self.data_dict.get('posfile', 'POSCAR'), 'r') as f:
                 poscar_content = f.read()
        except:
             pass
             
        return {
            "lattice": self.data_dict['Lattice'].tolist(),
            "atoms": self.data_dict['atoms'], 
            "coords": self.data_dict.get('coords', []).tolist(), 
            "atom_proj": self.data_dict.get('atom_proj', []),
            "poscar_content": poscar_content
        }

    def get_dft_bands(self):
        if not self.data_dict: return None
        
        # Use aligned subset of DFT bands
        vasp_bands = np.array(self.data_dict['vasp_bands']) # (nk, nbands)
        n_wan = self.data_dict['num_wan']
        offset = getattr(self, 'best_offset', 0)
        
        dft_subset = vasp_bands[:, offset : offset + n_wan]

        return {
            "bands": dft_subset.T.tolist(), 
            "kpath": self.data_dict['kpath'].tolist(),
            "k_distance": self.data_dict['xpath'].tolist(),
            "k_ticks": self.data_dict['xsymm'].tolist(),
            "k_labels": self.data_dict['plot_sbol']
        }

    def calculate_tb_bands(self, lambdas_list):
        # lambdas_list is the full list of lambdas from UI
        if self.hk_tb_jax is None: return None
        
        full_lambdas = np.array(lambdas_list)
        
        # We need to construct H_soc
        # If we use the pre-calculated basis, we only have matrices for 'fit_indices'.
        # So we must extract the values for those indices.
        
        current_params = full_lambdas[self.fit_indices]
        
        # Reconstruct H_total
        # H = H_tb + sum(lambda_i * M_i)
        # Using jax logic:
        # h_soc = jnp.tensordot(lambdas, soc_basis_matrices, axes=1)
        
        if len(self.fit_indices) > 0:
            h_soc = jnp.tensordot(current_params, self.soc_basis_jax, axes=1)
            h_total = self.hk_tb_jax + h_soc
        else:
            h_total = self.hk_tb_jax
            
        eigvals = jnp.linalg.eigvalsh(h_total)
        return eigvals.T.tolist() # (n_bands, nk) for plotting convenience
