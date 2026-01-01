from tbsoc.entrypoints.loadall import load_all_data
from tbsoc.lib.soc_mat import get_Hsoc
from tbsoc.lib.cal_tools import hr2hk
from tbsoc.entrypoints.fitsoc import build_soc_basis_matrices, find_best_alignment
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
        
        print("DataManager: Aligning bands...")
        best_offset, min_mse = find_best_alignment(eigvals, vasp_bands, n_wan, n_dft, n_k)
        self.best_offset = best_offset
        
        # Store alignment details for plotting
        self.n_dft = n_dft
        self.n_wan = n_wan
        self.n_compare = min(n_wan, n_dft - best_offset)
        
        dft_target_window = vasp_bands[:, best_offset : best_offset + self.n_compare]
        self.dft_window_mean = np.mean(dft_target_window)
        
        print(f"DataManager: Alignment Found. Offset: {best_offset}, Overlap: {self.n_compare}")


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
        offset = getattr(self, 'best_offset', 0)
        n_compare = getattr(self, 'n_compare', self.data_dict['num_wan'])
        
        # Handle case where n_dft < offset + n_compare (shouldn't happen with correct logic)
        max_idx = min(vasp_bands.shape[1], offset + n_compare)
        dft_subset = vasp_bands[:, offset : max_idx]

        return {
            "bands": dft_subset.T.tolist(), 
            "kpath": self.data_dict['kpath'].tolist(),
            "k_distance": self.data_dict['xpath'].tolist(),
            "k_ticks": self.data_dict['xsymm'].tolist(),
            "k_labels": self.data_dict['plot_sbol'],
            "offset": offset
        }

    def calculate_tb_bands(self, lambdas_list):
        # lambdas_list is the full list of lambdas from UI
        if self.hk_tb_jax is None: return None
        
        full_lambdas = np.array(lambdas_list)
        
        current_params = full_lambdas[self.fit_indices]
        
        if len(self.fit_indices) > 0:
            h_soc = jnp.tensordot(current_params, self.soc_basis_jax, axes=1)
            h_total = self.hk_tb_jax + h_soc
        else:
            h_total = self.hk_tb_jax
            
        eigvals = jnp.linalg.eigvalsh(h_total) # (nk, n_wan)
        
        # Create numpy copy for alignment search
        eigvals_np = np.array(eigvals)

        # --- Dynamic Re-Alignment ---
        # The user requested that we re-search the window when lambdas change.
        if self.data_dict:
            vasp_bands = self.data_dict['vasp_bands']
            n_dft = self.n_dft
            n_wan = self.n_wan
            n_k = vasp_bands.shape[0]
            
            # Re-scan for best index
            best_offset, min_mse = find_best_alignment(eigvals_np, vasp_bands, n_wan, n_dft, n_k)
            
            # Update state if changed
            if best_offset != self.best_offset:
                # print(f"DataManager: Re-alignment changed offset from {self.best_offset} to {best_offset}")
                self.best_offset = best_offset
                self.n_compare = min(n_wan, n_dft - best_offset)

        # --- Apply Alignment Shift ---
        try:
            vasp_bands = self.data_dict['vasp_bands']
            offset = getattr(self, 'best_offset', 0)
            n_compare = getattr(self, 'n_compare', 0)
            
            if n_compare > 0:
                dft_subset = vasp_bands[:, offset : offset + n_compare]
                tb_subset = eigvals[:, :n_compare]
                
                mean_diff = np.mean(tb_subset - dft_subset)
                eigvals_shifted = eigvals - mean_diff
                
                mae = float(np.mean(np.abs(eigvals_shifted[:, :n_compare] - dft_subset)))
                
                return {
                    "bands": eigvals_shifted.T.tolist(),
                    "mae": mae,
                    "offset": int(offset)
                }
        except Exception as e:
            print(f"Warning: Alignment/MAE calculation failed: {e}")
            
        return {"bands": eigvals.T.tolist(), "mae": None, "offset": getattr(self, 'best_offset', 0)}
