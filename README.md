# TBSOC (Wannier + On-site SOC)

**TBSOC** is a lightweight, high-performance Python package designed to add On-Site Spin-Orbit Coupling (SOC) to Wannier90 Tight-Binding (TB) models. It allows users to estimate accurate SOC strengths ($\lambda$) by automatically fitting TB band structures to DFT calculations.

## Key Features
- **High Performance**: Powered by **JAX** and **Just-In-Time (JIT)** compilation for lightning-fast fitting (<1s for typical systems).
- **Auto-Alignment**: Automatically detects the correct matching between TB and DFT bands (handles band index offsets).
- **Physically-Aware**: Uses **Fermi-weighted loss** to prioritize accuracy near the Fermi level.
- **Differentiable**: Uses exact gradients (Automatic Differentiation) for robust convergence, replacing slow derivative-free methods.
- **Flexible**: Supports custom local axes and spin quantization axes.

## Installation

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (Recommended for package management)

### Install via uv (Recommended)
```bash
git clone https://github.com/qqgu/TBSOC.git
cd TBSOC
uv sync
```

### Install via pip
```bash
pip install .
```

## Usage

TBSOC provides a command-line interface (`tbsoc`) with three main modes:

### 1. Pre-calculate (`precalc`)
Converts `wannier90_hr.dat` to a more efficient format.
```bash
tbsoc precalc input.json
```

### 2. Fit SOC (`fit`)
Automatically fits the SOC parameters ($\lambda$) to match a DFT band structure (VASP `EIGENVAL`).
```bash
tbsoc fit input.json
```
**Key `input.json` parameters:**
- `lambdas`: Initial guess for SOC strengths (e.g., `[0, 0.1, 0]`). Zero values are fixed; non-zero are optimized.
- `Efermi`: DFT Fermi Energy (eV).
- `weight_sigma`: (Optional) Width of the Fermi weighting window (default: 2.0 eV).

### 3. Add SOC (`addsoc`)
Calculates the final bands with specific SOC parameters and plots the result.
```bash
tbsoc addsoc input.json
```

## Input File Format (`input.json`)
See `example/` directory for complete examples (GaAs, TaAs, etc.).
```json
{
    "lambdas": [0, 0.1, 0],
    "Efermi": 4.0815,
    "vasp_bands_file": "EIGENVAL",
    "orbitals": ["Ga", "As"],
    "orb_type": [1, 1],
    "orb_num": [3, 3],
    ...
}
```

## Cite
If you use this code, please cite our paper:
```bibtex
@article{GU2023112090,
  title = {A computational method to estimate spin–orbital interaction strength in solid state systems},
  journal = {Computational Materials Science},
  volume = {221},
  pages = {112090},
  year = {2023},
  issn = {0927-0256},
  doi = {https://doi.org/10.1016/j.commatsci.2023.112090},
  url = {https://www.sciencedirect.com/science/article/pii/S0927025623000848},
  author = {Qiangqiang Gu and Shishir Kumar Pandey and Rajarshi Tiwari}
}
```
